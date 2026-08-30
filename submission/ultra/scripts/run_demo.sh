#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
init_ultra_submission_env

DEMO_ACTION=${DEMO_ACTION:-start}
BACKEND_PORT=${BACKEND_PORT:-8099}
GRADIO_PORT=${GRADIO_PORT:-7862}

finalize_demo() {
  : "${DEMO_RUN_DIR:?set DEMO_RUN_DIR to the existing Demo start run}"
  : "${DEMO_VIDEO_PATH:?copy the Mac recording to the remote host and set DEMO_VIDEO_PATH}"
  : "${DEMO_SCREENSHOT_DIR:?set DEMO_SCREENSHOT_DIR to one or more Demo screenshots}"
  require_dir "$DEMO_RUN_DIR"
  require_file "$DEMO_RUN_DIR/interactions.jsonl"
  require_file "$DEMO_VIDEO_PATH"
  require_dir "$DEMO_SCREENSHOT_DIR"
  test ! -e "$DEMO_RUN_DIR/finalization" || \
    die "Demo run is already finalized: $DEMO_RUN_DIR/finalization"
  mkdir -p "$DEMO_RUN_DIR/finalization/screenshots" "$DEMO_RUN_DIR/finalization/outputs"
  cp -a "$DEMO_VIDEO_PATH" "$DEMO_RUN_DIR/finalization/demo-recording.mp4"
  find "$DEMO_SCREENSHOT_DIR" -maxdepth 1 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    -exec cp -a {} "$DEMO_RUN_DIR/finalization/screenshots/" \;

  DEMO_RUN_DIR="$DEMO_RUN_DIR" "$TEST_PY" - <<'PY'
import hashlib
import json
import os
import shutil
import wave
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["DEMO_RUN_DIR"])
lines = [line for line in (root / "interactions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
rows = [json.loads(line) for line in lines]
errors = []
required_modalities = {"text", "image", "audio", "video"}
seen_modalities = {str(row.get("input_modality", "")) for row in rows}
if len(rows) < 10:
    errors.append(f"interaction count {len(rows)} is below 10")
if not required_modalities.issubset(seen_modalities):
    errors.append(f"missing modalities: {sorted(required_modalities - seen_modalities)}")

audio_checks = []
copied = 0
for index, row in enumerate(rows, 1):
    for key in ("output_text_nonempty", "streaming_audio", "service_healthy_after"):
        if row.get(key) is not True:
            errors.append(f"interaction {index}: {key} is not true")
    audio_raw = row.get("output_audio_path")
    if not audio_raw:
        errors.append(f"interaction {index}: missing output_audio_path")
        continue
    audio = Path(str(audio_raw)).expanduser()
    if not audio.is_file():
        errors.append(f"interaction {index}: audio file not found: {audio}")
        continue
    try:
        with wave.open(str(audio), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
    except (wave.Error, EOFError) as exc:
        errors.append(f"interaction {index}: WAV decode failed: {exc}")
        continue
    valid = channels == 1 and sample_rate == 24000 and frames > 0
    if not valid:
        errors.append(
            f"interaction {index}: expected non-empty 24 kHz mono WAV, got "
            f"channels={channels}, sample_rate={sample_rate}, frames={frames}"
        )
    audio_checks.append({
        "interaction_id": row.get("interaction_id", index),
        "path": str(audio.resolve()),
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "frames": frames,
        "valid": valid,
    })
    if copied < 5:
        target = root / "finalization" / "outputs" / f"interaction-{index:02d}.wav"
        shutil.copy2(audio, target)
        copied += 1

video = root / "finalization" / "demo-recording.mp4"
if not video.is_file() or video.stat().st_size == 0:
    errors.append("Demo recording is missing or empty")
screenshots = sorted((root / "finalization" / "screenshots").glob("*"))
if not screenshots:
    errors.append("no Demo screenshots were copied")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

artifacts = []
for path in sorted((root / "finalization").rglob("*")):
    if path.is_file():
        artifacts.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

summary = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "candidate": os.environ["CANDIDATE_LABEL"],
    "candidate_head": json.loads((root / "manifest.json").read_text(encoding="utf-8"))["candidate_head"],
    "status": "PASS" if not errors else "FAIL",
    "standard_demo": "examples/online_serving/minicpmo/gradio_demo.py",
    "interaction_count": len(rows),
    "modalities_seen": sorted(seen_modalities),
    "required_modalities": sorted(required_modalities),
    "audio_contract": {"sample_rate_hz": 24000, "channels": 1},
    "audio_checks": audio_checks,
    "service_stable": all(row.get("service_healthy_after") is True for row in rows),
    "recording": str(video),
    "artifacts": artifacts,
    "errors": errors,
}
start_summary = root / "demo-summary.json"
if start_summary.is_file():
    shutil.copy2(start_summary, root / "demo-summary-start.json")
start_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
raise SystemExit(0 if not errors else 2)
PY
  printf 'DEMO_RESULT=PASS\nDEMO_RUN_DIR=%s\n' "$DEMO_RUN_DIR"
}

if test "$DEMO_ACTION" = "finalize"; then
  finalize_demo
  exit 0
fi
test "$DEMO_ACTION" = "start" || die "DEMO_ACTION must be start or finalize"

RUN_DIR=${RUN_DIR:-$COMP_ROOT/results/$CANDIDATE_LABEL/demo/$RUN_ID}
require_new_dir "$RUN_DIR"
verify_candidate_source
verify_contract_files
write_run_manifest "$RUN_DIR" demo-start

SERVICE_PID=$(start_candidate_service "$RUN_DIR" "$BACKEND_PORT")
GRADIO_PID=
cleanup() {
  if test -n "$GRADIO_PID"; then
    stop_process_tree "$GRADIO_PID"
  fi
  stop_process_tree "$SERVICE_PID"
}
trap cleanup EXIT INT TERM
wait_for_health "$SERVICE_PID" "$BACKEND_PORT" "$RUN_DIR/service.log" 90

GRADIO_SCRIPT="$CANDIDATE_SRC/examples/online_serving/minicpmo/gradio_demo.py"
require_file "$GRADIO_SCRIPT"
GRADIO_COMMAND=(
  "$TEST_PY" "$GRADIO_SCRIPT"
  --minicpmo45-api-base "http://127.0.0.1:$BACKEND_PORT/v1"
  --minicpmo45-model "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$GRADIO_PORT"
)
record_command "$RUN_DIR/gradio-command.sh" "${GRADIO_COMMAND[@]}"
(
  cd "$CANDIDATE_SRC"
  nohup "${GRADIO_COMMAND[@]}" > "$RUN_DIR/gradio.log" 2>&1 &
  printf '%s\n' "$!" > "$RUN_DIR/gradio.pid"
)
GRADIO_PID=$(cat "$RUN_DIR/gradio.pid")

for attempt in $(seq 1 60); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$GRADIO_PORT/" >/dev/null 2>&1; then
    break
  fi
  kill -0 "$GRADIO_PID" 2>/dev/null || {
    tail -n 120 "$RUN_DIR/gradio.log" >&2 || true
    die "Gradio process exited"
  }
  test "$attempt" -lt 60 || die "Gradio did not become ready"
  sleep 2
done

"$TEST_PY" - "$RUN_DIR" "$BACKEND_PORT" "$GRADIO_PORT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
summary = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "status": "IN_PROGRESS",
    "backend": f"http://127.0.0.1:{sys.argv[2]}/v1",
    "gradio": f"http://127.0.0.1:{sys.argv[3]}",
    "interaction_count": 0,
    "instructions": "Complete the ledger, stop services with Ctrl-C, then run DEMO_ACTION=finalize.",
}
(root / "demo-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
example = {
    "interaction_id": 1,
    "input_modality": "text",
    "output_text_nonempty": True,
    "streaming_audio": True,
    "output_audio_path": "/absolute/path/output.wav",
    "service_healthy_after": True,
    "notes": "",
}
(root / "interactions.jsonl.example").write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8")
PY

printf '\nDemo is ready.\n'
printf 'RUN_DIR=%s\n' "$RUN_DIR"
printf 'Open http://127.0.0.1:%s after port forwarding.\n' "$GRADIO_PORT"
printf 'Record at least 10 rows in %s/interactions.jsonl using the example schema.\n' "$RUN_DIR"
printf 'Press Ctrl-C after interactions and recording are complete; then run finalize.\n\n'

while kill -0 "$SERVICE_PID" 2>/dev/null && kill -0 "$GRADIO_PID" 2>/dev/null; do
  sleep 5
done
die "a Demo process exited; inspect $RUN_DIR/service.log and $RUN_DIR/gradio.log"
