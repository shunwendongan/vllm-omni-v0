#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
init_ultra_submission_env

PORT=${PORT:-8091}
PERFORMANCE_MATRIX=${PERFORMANCE_MATRIX:-all}
SEED_TTS_DATASET=${SEED_TTS_DATASET:-$DATA_ROOT/seed-tts/seedtts_testset}
ACCURACY_SUMMARY_PATH=${ACCURACY_SUMMARY_PATH:-}
BYPASS_ACCURACY_GATE=${BYPASS_ACCURACY_GATE:-0}
RUN_DIR=${RUN_DIR:-$COMP_ROOT/results/$CANDIDATE_LABEL/performance/official-matrix/$RUN_ID}

require_new_dir "$RUN_DIR"
verify_candidate_source
verify_contract_files
require_dir "$SEED_TTS_DATASET"
write_run_manifest "$RUN_DIR" performance

ELIGIBILITY=RANKING_ELIGIBLE
if test -z "$ACCURACY_SUMMARY_PATH" || test ! -f "$ACCURACY_SUMMARY_PATH"; then
  if test "$BYPASS_ACCURACY_GATE" != "1"; then
    die "set ACCURACY_SUMMARY_PATH to the passing Ultra full-accuracy summary"
  fi
  ELIGIBILITY=NOT_RANKING_ELIGIBLE
else
  set +e
  "$TEST_PY" - "$ACCURACY_SUMMARY_PATH" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = (
    data.get("status") == "PASS",
    data.get("coverage_ok") is True,
    data.get("daily_omni", {}).get("gate_pass") is True,
    data.get("videomme", {}).get("gate_pass") is True,
    data.get("seed_tts", {}).get("wer_gate_pass") is True,
    data.get("seed_tts", {}).get("asv_sim_gate_pass") is True,
)
if not all(required):
    raise SystemExit("Ultra accuracy summary does not pass every gate")
print("Ultra accuracy prerequisite: PASS")
PY
  accuracy_rc=$?
  set -e
  if test "$accuracy_rc" -ne 0; then
    if test "$BYPASS_ACCURACY_GATE" != "1"; then
      die "accuracy gate failed; performance ranking run is blocked"
    fi
    ELIGIBILITY=NOT_RANKING_ELIGIBLE
  fi
fi
printf '%s\n' "$ELIGIBILITY" > "$RUN_DIR/eligibility.txt"

SERVICE_PID=$(start_candidate_service "$RUN_DIR" "$PORT")
MONITOR_PID=
cleanup() {
  if test -n "$MONITOR_PID" && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill -TERM "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  stop_process_tree "$SERVICE_PID"
}
trap cleanup EXIT INT TERM
wait_for_health "$SERVICE_PID" "$PORT" "$RUN_DIR/service.log" 90

(
  while kill -0 "$SERVICE_PID" 2>/dev/null; do
    date -Is
    if command -v npu-smi >/dev/null 2>&1; then
      npu-smi info || true
    fi
    ps -p "$SERVICE_PID" -o pid,ppid,stat,etime,%cpu,%mem,rss,vsz,cmd || true
    sleep 10
  done
) > "$RUN_DIR/resource-monitor.log" 2>&1 &
MONITOR_PID=$!

mkdir -p "$RUN_DIR/c1-p32" "$RUN_DIR/c4-p64" "$RUN_DIR/c8-p128"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

run_bench() {
  local label=$1
  local prompts=$2
  local concurrency=$3
  local warmups=$4
  local repeat=$5
  local result_dir="$RUN_DIR/$label"
  local filename="${label}-run${repeat}.json"
  local log_file="$result_dir/run${repeat}.log"
  local -a command=(
    "$VLLM_BIN" bench serve
    --omni
    --host 127.0.0.1
    --port "$PORT"
    --backend openai-chat-omni
    --endpoint /v1/chat/completions
    --model "$SERVED_MODEL_NAME"
    --tokenizer "$MODEL_PATH"
    --trust-remote-code
    --dataset-name seed-tts
    --dataset-path "$SEED_TTS_DATASET"
    --seed-tts-root "$SEED_TTS_DATASET"
    --seed-tts-locale zh
    --seed-tts-file-ref-audio
    --num-prompts "$prompts"
    --max-concurrency "$concurrency"
    --num-warmups "$warmups"
    --no-oversample
    --disable-shuffle
    --temperature 0
    --percentile-metrics ttft,e2el,audio_ttfp,audio_rtf,audio_duration
    --metric-percentiles 50,90,95,99
    --extra-body '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}'
    --save-result
    --result-dir "$result_dir"
    --result-filename "$filename"
  )
  record_command "$result_dir/run${repeat}-command.sh" "${command[@]}"
  set +e
  (
    cd "$CANDIDATE_SRC"
    "${command[@]}"
  ) 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$result_dir/run${repeat}-exit-code.txt"
}

for repeat in 1 2 3; do
  run_bench c1-p32 32 1 2 "$repeat"
done

if test "$PERFORMANCE_MATRIX" = "all"; then
  run_bench c4-p64 64 4 2 1
  run_bench c8-p128 128 8 2 1
elif test "$PERFORMANCE_MATRIX" != "c1"; then
  die "PERFORMANCE_MATRIX must be all or c1"
fi

ELIGIBILITY="$ELIGIBILITY" PERFORMANCE_MATRIX="$PERFORMANCE_MATRIX" \
"$TEST_PY" - "$RUN_DIR" <<'PY'
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])

def load_case(label: str, prompts: int, expected_runs: int) -> tuple[list[dict], list[str]]:
    rows = []
    errors = []
    for idx in range(1, expected_runs + 1):
        path = root / label / f"{label}-run{idx}.json"
        exit_path = root / label / f"run{idx}-exit-code.txt"
        rc = int(exit_path.read_text().strip()) if exit_path.is_file() else None
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "run": idx,
            "result_file": str(path),
            "exit_code": rc,
            "completed": data.get("completed"),
            "failed": data.get("failed"),
            "request_throughput": data.get("request_throughput"),
            "mean_ttft_ms": data.get("mean_ttft_ms"),
            "mean_e2el_ms": data.get("mean_e2el_ms"),
            "mean_audio_ttfp_ms": data.get("mean_audio_ttfp_ms"),
            "mean_audio_rtf": data.get("mean_audio_rtf"),
            "total_audio_duration_s": data.get("total_audio_duration_s"),
            "total_audio_frames": data.get("total_audio_frames"),
            "audio_continuity_ok_rate": data.get("audio_continuity_ok_rate"),
        }
        numeric = ("mean_ttft_ms", "mean_e2el_ms", "mean_audio_ttfp_ms", "mean_audio_rtf")
        valid = (
            rc == 0
            and int(row["completed"] or 0) == prompts
            and int(row["failed"] or 0) == 0
            and all(isinstance(row[key], (int, float)) and math.isfinite(float(row[key])) for key in numeric)
            and float(row["total_audio_duration_s"] or 0) > 0
            and int(row["total_audio_frames"] or 0) > 0
        )
        row["valid"] = valid
        row["audio_output_success_rate_inferred"] = (float(row["completed"] or 0) / prompts) if valid else None
        if not valid:
            errors.append(f"invalid result: {path}")
        rows.append(row)
    return rows, errors

def distribution(rows: list[dict]) -> dict:
    metrics = ("mean_audio_rtf", "mean_audio_ttfp_ms", "mean_ttft_ms", "mean_e2el_ms")
    out = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get("valid")]
        out[metric] = {
            "values": values,
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "range": (max(values) - min(values)) if values else None,
        }
    return out

ranking, errors = load_case("c1-p32", 32, 3)
guardrails = {}
if os.environ["PERFORMANCE_MATRIX"] == "all":
    c4, c4_errors = load_case("c4-p64", 64, 1)
    c8, c8_errors = load_case("c8-p128", 128, 1)
    errors.extend(c4_errors)
    errors.extend(c8_errors)
    guardrails = {"c4-p64": c4, "c8-p128": c8}

eligible = os.environ["ELIGIBILITY"] == "RANKING_ELIGIBLE"
all_valid = not errors and len(ranking) == 3 and all(row["valid"] for row in ranking)
if os.environ["PERFORMANCE_MATRIX"] == "all":
    all_valid = all_valid and all(rows and rows[0]["valid"] for rows in guardrails.values())
status = "PASS" if all_valid and eligible else ("NOT_RANKING_ELIGIBLE" if all_valid else "FAIL")
summary = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "candidate": os.environ["CANDIDATE_LABEL"],
    "candidate_head": json.loads((root / "manifest.json").read_text())["candidate_head"],
    "hardware": os.environ["LOCAL_HARDWARE"],
    "status": status,
    "eligibility": os.environ["ELIGIBILITY"],
    "matrix": os.environ["PERFORMANCE_MATRIX"],
    "ranking_unit": {
        "dataset": "Seed-TTS zh",
        "num_prompts": 32,
        "max_concurrency": 1,
        "num_warmups": 2,
        "formal_runs": ranking,
        "distribution": distribution(ranking),
    },
    "guardrails_non_ranking": guardrails,
    "errors": errors,
    "resource_monitor": str(root / "resource-monitor.log"),
    "audio_decode_rate_note": "Aggregate output lacks a per-request decoded-audio count; completed/expected with nonzero total audio frames is recorded as an inferred output-success rate.",
    "official_a3_reference_cross_hardware_only": {
        "mean_audio_rtf": 0.4423,
        "mean_audio_ttfp_ms": 986.4666,
        "mean_ttft_ms": 333.2633,
        "mean_e2el_ms": 1857.2154,
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
raise SystemExit(0 if status == "PASS" else 2)
PY

printf 'PERFORMANCE_RESULT=PASS\nRUN_DIR=%s\n' "$RUN_DIR"
