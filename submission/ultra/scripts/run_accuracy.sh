#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
init_ultra_submission_env

PORT=${PORT:-8091}
ACCURACY_SUITE=${ACCURACY_SUITE:-all}
EXPECTED_DAILY_ITEMS=${EXPECTED_DAILY_ITEMS:-1197}
EXPECTED_VIDEOMME_ITEMS=${EXPECTED_VIDEOMME_ITEMS:-2700}
EXPECTED_VIDEOMME_MP4=${EXPECTED_VIDEOMME_MP4:-900}
EXPECTED_SEED_TTS_ITEMS=${EXPECTED_SEED_TTS_ITEMS:-2020}
DAILY_QA_JSON=${DAILY_QA_JSON:-$DATA_ROOT/Daily-Omni/qa.json}
DAILY_VIDEO_DIR=${DAILY_VIDEO_DIR:-$DATA_ROOT/Daily-Omni/Videos}
VIDEOMME_ROOT=${VIDEOMME_ROOT:-$DATA_ROOT/Video-MME}
SEED_TTS_DATASET=${SEED_TTS_DATASET:-$DATA_ROOT/seed-tts/seedtts_testset}
SEED_TTS_META=${SEED_TTS_META:-$SEED_TTS_DATASET/zh/meta.lst}
SEED_TTS_WAVLM_MODEL=${SEED_TTS_WAVLM_MODEL:-microsoft/wavlm-base-plus}
SEED_TTS_SIM_DEVICE=${SEED_TTS_SIM_DEVICE:-cpu}
RUN_DIR=${RUN_DIR:-$COMP_ROOT/results/$CANDIDATE_LABEL/accuracy/$ACCURACY_SUITE/$RUN_ID}

RUN_DAILY=0
RUN_VIDEO=0
RUN_SEED=0
case "$ACCURACY_SUITE" in
  all) RUN_DAILY=1; RUN_VIDEO=1; RUN_SEED=1 ;;
  daily_omni) RUN_DAILY=1 ;;
  videomme) RUN_VIDEO=1 ;;
  seed_tts) RUN_SEED=1 ;;
  *) die "ACCURACY_SUITE must be all, daily_omni, videomme, or seed_tts" ;;
esac

require_new_dir "$RUN_DIR"
verify_candidate_source
verify_contract_files
if test "$RUN_DAILY" = "1"; then
  require_file "$DAILY_QA_JSON"
  require_dir "$DAILY_VIDEO_DIR"
fi
if test "$RUN_VIDEO" = "1"; then
  require_dir "$VIDEOMME_ROOT"
fi
if test "$RUN_SEED" = "1"; then
  require_dir "$SEED_TTS_DATASET"
  require_file "$SEED_TTS_META"
fi
write_run_manifest "$RUN_DIR" "$ACCURACY_SUITE-accuracy"

EXPECTED_DAILY_ITEMS="$EXPECTED_DAILY_ITEMS" \
EXPECTED_VIDEOMME_MP4="$EXPECTED_VIDEOMME_MP4" \
EXPECTED_SEED_TTS_ITEMS="$EXPECTED_SEED_TTS_ITEMS" \
ACCURACY_SUITE="$ACCURACY_SUITE" \
DAILY_QA_JSON="$DAILY_QA_JSON" DAILY_VIDEO_DIR="$DAILY_VIDEO_DIR" \
VIDEOMME_ROOT="$VIDEOMME_ROOT" SEED_TTS_META="$SEED_TTS_META" \
"$TEST_PY" - "$RUN_DIR/dataset-preflight.json" <<'PY'
import json
import os
import sys
from pathlib import Path

expected_daily = int(os.environ["EXPECTED_DAILY_ITEMS"])
expected_vm_mp4 = int(os.environ["EXPECTED_VIDEOMME_MP4"])
expected_seed = int(os.environ["EXPECTED_SEED_TTS_ITEMS"])
suite = os.environ["ACCURACY_SUITE"]
checks = {}
payload = {"scope": suite, "checks": checks}

if suite in ("all", "daily_omni"):
    qa = json.loads(Path(os.environ["DAILY_QA_JSON"]).read_text(encoding="utf-8"))
    if isinstance(qa, list):
        daily_rows = len(qa)
    elif isinstance(qa, dict):
        for key in ("questions", "data", "items", "qa"):
            if isinstance(qa.get(key), list):
                daily_rows = len(qa[key])
                break
        else:
            daily_rows = len(qa)
    else:
        raise SystemExit("Daily-Omni qa.json must contain a list or object")
    daily_mp4 = sum(1 for p in Path(os.environ["DAILY_VIDEO_DIR"]).rglob("*.mp4") if p.is_file())
    daily_wav = sum(1 for p in Path(os.environ["DAILY_VIDEO_DIR"]).rglob("*.wav") if p.is_file())
    checks["daily_rows_match"] = daily_rows == expected_daily
    checks["daily_media_nonempty"] = daily_mp4 > 0 and daily_wav > 0
    payload["daily_omni"] = {"qa_rows": daily_rows, "expected": expected_daily, "mp4": daily_mp4, "wav": daily_wav}

if suite in ("all", "videomme"):
    videomme_mp4 = sum(1 for p in Path(os.environ["VIDEOMME_ROOT"]).rglob("*.mp4") if p.is_file())
    checks["videomme_mp4_match"] = videomme_mp4 == expected_vm_mp4
    payload["videomme"] = {"mp4": videomme_mp4, "expected_mp4": expected_vm_mp4}

if suite in ("all", "seed_tts"):
    seed_rows = sum(1 for line in Path(os.environ["SEED_TTS_META"]).read_text(encoding="utf-8").splitlines() if line.strip())
    checks["seed_tts_rows_match"] = seed_rows == expected_seed
    payload["seed_tts"] = {"meta_rows": seed_rows, "expected": expected_seed}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if not all(checks.values()):
    raise SystemExit("dataset preflight failed; repair coverage before running accuracy")
PY

SERVICE_PID=$(start_candidate_service "$RUN_DIR" "$PORT")
cleanup() {
  stop_process_tree "$SERVICE_PID"
}
trap cleanup EXIT INT TERM
wait_for_health "$SERVICE_PID" "$PORT" "$RUN_DIR/service.log" 90

ACC_DRIVER="$CANDIDATE_SRC/tests/e2e/accuracy/qwen3_omni/run_qwen_omni_acc_benchmark.py"
require_file "$ACC_DRIVER"
mkdir -p "$RUN_DIR/daily_omni" "$RUN_DIR/videomme" "$RUN_DIR/seed_tts"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export SEED_TTS_SIM_EVAL=1
export SEED_TTS_SIM_DEVICE
export SEED_TTS_WAVLM_MODEL

run_suite() {
  local name=$1
  shift
  record_command "$RUN_DIR/$name/command.sh" "$TEST_PY" "$ACC_DRIVER" "$@"
  set +e
  "$TEST_PY" "$ACC_DRIVER" "$@" 2>&1 | tee "$RUN_DIR/$name/benchmark.log"
  local rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$RUN_DIR/$name/exit-code.txt"
}

COMMON=(
  --host 127.0.0.1
  --port "$PORT"
  --model "$SERVED_MODEL_NAME"
  --trust-remote-code
  --ready-check-timeout-sec 180
)

if test "$RUN_DAILY" = "1"; then
run_suite daily_omni \
  "${COMMON[@]}" \
  --num-prompts "$EXPECTED_DAILY_ITEMS" \
  --max-concurrency 10 \
  --num-warmups 0 \
  --result-dir "$RUN_DIR/daily_omni" \
  --skip-seed-tts \
  --skip-videomme \
  --temperature 0 \
  --output-len 512 \
  --daily-omni-qa-json "$DAILY_QA_JSON" \
  --daily-omni-video-dir "$DAILY_VIDEO_DIR" \
  --daily-omni-input-mode all \
  --daily-omni-pack-mode minicpm-interleave \
  --daily-omni-save-eval-items \
  --min-daily-omni-accuracy 0.775 \
  --daily-extra-body-json '{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false}}'
fi

if test "$RUN_VIDEO" = "1"; then
run_suite videomme \
  "${COMMON[@]}" \
  --num-prompts "$EXPECTED_VIDEOMME_ITEMS" \
  --max-concurrency 4 \
  --num-warmups 0 \
  --result-dir "$RUN_DIR/videomme" \
  --skip-daily-omni \
  --skip-seed-tts \
  --run-videomme \
  --temperature 0 \
  --output-len 128 \
  --videomme-dataset-path "$VIDEOMME_ROOT" \
  --videomme-pack-mode minicpm-frames \
  --videomme-max-frames 96 \
  --videomme-duration all \
  --videomme-save-eval-items \
  --min-videomme-accuracy 0.67 \
  --videomme-extra-body-json '{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false}}'
fi

if test "$RUN_SEED" = "1"; then
run_suite seed_tts \
  "${COMMON[@]}" \
  --num-prompts "$EXPECTED_SEED_TTS_ITEMS" \
  --max-concurrency 1 \
  --num-warmups 0 \
  --result-dir "$RUN_DIR/seed_tts" \
  --skip-daily-omni \
  --skip-videomme \
  --temperature 0 \
  --seed-tts-dataset-path "$SEED_TTS_DATASET" \
  --seed-tts-root "$SEED_TTS_DATASET" \
  --seed-tts-locale zh \
  --seed-tts-file-ref-audio \
  --seed-tts-wer-save-items \
  --seed-tts-eval-device cpu \
  --max-seed-tts-mean-wer 0.0156 \
  --min-seed-tts-mean-sim 0.689 \
  --seed-extra-body-json '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}'
fi

EXPECTED_DAILY_ITEMS="$EXPECTED_DAILY_ITEMS" \
EXPECTED_VIDEOMME_ITEMS="$EXPECTED_VIDEOMME_ITEMS" \
EXPECTED_SEED_TTS_ITEMS="$EXPECTED_SEED_TTS_ITEMS" \
SEED_TTS_WAVLM_MODEL="$SEED_TTS_WAVLM_MODEL" \
ACCURACY_SUITE="$ACCURACY_SUITE" \
"$TEST_PY" - "$RUN_DIR" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
scope = os.environ["ACCURACY_SUITE"]

def one_json(name: str) -> tuple[Path | None, dict]:
    files = sorted((root / name).glob("*.json"))
    if len(files) != 1:
        return None, {}
    return files[0], json.loads(files[0].read_text(encoding="utf-8"))

daily_path, daily = one_json("daily_omni")
video_path, video = one_json("videomme")
seed_path, seed = one_json("seed_tts")
daily_n = int(os.environ["EXPECTED_DAILY_ITEMS"])
video_n = int(os.environ["EXPECTED_VIDEOMME_ITEMS"])
seed_n = int(os.environ["EXPECTED_SEED_TTS_ITEMS"])

daily_acc = daily.get("daily_omni_accuracy")
video_acc = video.get("videomme_accuracy")
wer = seed.get("seed_tts_content_error_mean")
sim = seed.get("seed_tts_sim_mean")

daily_coverage = (
    int(daily.get("daily_omni_evaluated", 0) or 0) == daily_n
    and int(daily.get("daily_omni_evaluated_ok", 0) or 0) == daily_n
    and int(daily.get("daily_omni_request_failed", 0) or 0) == 0
)
video_coverage = (
    int(video.get("videomme_evaluated", 0) or 0) == video_n
    and int(video.get("videomme_evaluated_ok", 0) or 0) == video_n
    and int(video.get("videomme_request_failed", 0) or 0) == 0
)
seed_coverage = (
    int(seed.get("seed_tts_content_evaluated", 0) or 0) == seed_n
    and int(seed.get("seed_tts_sim_evaluated", 0) or 0) == seed_n
    and all(int(seed.get(key, 0) or 0) == 0 for key in (
        "seed_tts_request_failed", "seed_tts_no_pcm", "seed_tts_asr_failed",
        "seed_tts_sim_failed", "seed_tts_sim_skipped_no_ref",
    ))
)
daily_pass = daily_acc is not None and float(daily_acc) >= 0.775
video_pass = video_acc is not None and float(video_acc) >= 0.67
wer_pass = wer is not None and float(wer) <= 0.0156
sim_pass = sim is not None and float(sim) >= 0.689
selected = {
    "daily_omni": daily_coverage and daily_pass,
    "videomme": video_coverage and video_pass,
    "seed_tts": seed_coverage and wer_pass and sim_pass,
}
coverage_ok = all((daily_coverage, video_coverage, seed_coverage)) if scope == "all" else {
    "daily_omni": daily_coverage,
    "videomme": video_coverage,
    "seed_tts": seed_coverage,
}[scope]
passed = all(selected.values()) if scope == "all" else selected[scope]

summary = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "candidate": os.environ["CANDIDATE_LABEL"],
    "candidate_head": json.loads((root / "manifest.json").read_text())["candidate_head"],
    "scope": scope,
    "status": ("PASS" if passed else "FAIL") if scope == "all" else ("PASS_PARTIAL" if passed else "FAIL_PARTIAL"),
    "daily_omni": {
        "result_file": str(daily_path) if daily_path else None,
        "accuracy": daily_acc,
        "evaluated": daily.get("daily_omni_evaluated"),
        "evaluated_ok": daily.get("daily_omni_evaluated_ok"),
        "request_failed": daily.get("daily_omni_request_failed"),
        "gate": 0.775,
        "coverage_ok": daily_coverage,
        "gate_pass": daily_pass,
    },
    "videomme": {
        "result_file": str(video_path) if video_path else None,
        "accuracy": video_acc,
        "evaluated": video.get("videomme_evaluated"),
        "evaluated_ok": video.get("videomme_evaluated_ok"),
        "request_failed": video.get("videomme_request_failed"),
        "gate": 0.67,
        "coverage_ok": video_coverage,
        "gate_pass": video_pass,
    },
    "seed_tts": {
        "result_file": str(seed_path) if seed_path else None,
        "wer_mean": wer,
        "wer_evaluated": seed.get("seed_tts_content_evaluated"),
        "wer_gate": 0.0156,
        "wer_gate_pass": wer_pass,
        "asv_sim_mean": sim,
        "sim_evaluated": seed.get("seed_tts_sim_evaluated"),
        "sim_failed": seed.get("seed_tts_sim_failed"),
        "asv_sim_gate": 0.689,
        "asv_sim_gate_pass": sim_pass,
        "embedding_model": os.environ["SEED_TTS_WAVLM_MODEL"],
        "evaluator_note": "Official-branch WavLM embedding cosine; source labels the default base-plus model a proxy for fine-tuned Seed-TTS UniSpeech/WavLM-SV.",
        "coverage_ok": seed_coverage,
    },
    "coverage_ok": coverage_ok,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
raise SystemExit(0 if passed else 2)
PY

printf 'ACCURACY_RESULT=%s\nRUN_DIR=%s\n' "$([ "$ACCURACY_SUITE" = all ] && printf PASS || printf PASS_PARTIAL)" "$RUN_DIR"
