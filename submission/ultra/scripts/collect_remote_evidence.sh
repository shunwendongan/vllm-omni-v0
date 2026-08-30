#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
init_ultra_submission_env

: "${ACCURACY_RESULT_DIR:?set ACCURACY_RESULT_DIR to the Ultra full-accuracy run}"
: "${PERFORMANCE_RESULT_DIR:?set PERFORMANCE_RESULT_DIR to the Ultra performance run}"
: "${DEMO_RESULT_DIR:?set DEMO_RESULT_DIR to the finalized Ultra Demo run}"
EVIDENCE_ROOT=${EVIDENCE_ROOT:-$COMP_ROOT/results/$CANDIDATE_LABEL/submission-evidence/$RUN_ID}

require_new_dir "$EVIDENCE_ROOT"
verify_candidate_source
verify_contract_files
for result_dir in "$ACCURACY_RESULT_DIR" "$PERFORMANCE_RESULT_DIR" "$DEMO_RESULT_DIR"; do
  require_dir "$result_dir"
done
require_file "$ACCURACY_RESULT_DIR/summary.json"
require_file "$PERFORMANCE_RESULT_DIR/summary.json"
require_file "$DEMO_RESULT_DIR/demo-summary.json"
write_run_manifest "$EVIDENCE_ROOT" collect-submission-evidence

copy_tree_without_accuracy_wavs() {
  local source_dir=$1
  local target_dir=$2
  mkdir -p "$target_dir"
  tar -C "$source_dir" \
    --exclude='./audio' \
    --exclude='./audio/*' \
    --exclude='*.wav' \
    -cf - . | tar -C "$target_dir" -xf -
}

copy_tree_without_accuracy_wavs "$ACCURACY_RESULT_DIR" "$EVIDENCE_ROOT/accuracy"
mkdir -p "$EVIDENCE_ROOT/performance" "$EVIDENCE_ROOT/demo"
tar -C "$PERFORMANCE_RESULT_DIR" -cf - . | tar -C "$EVIDENCE_ROOT/performance" -xf -
tar -C "$DEMO_RESULT_DIR" -cf - . | tar -C "$EVIDENCE_ROOT/demo" -xf -

git -C "$CANDIDATE_SRC" status --short > "$EVIDENCE_ROOT/candidate-git-status.txt"
git -C "$CANDIDATE_SRC" diff --binary > "$EVIDENCE_ROOT/candidate-worktree.diff"
git -C "$CANDIDATE_SRC" log -20 --decorate --oneline > "$EVIDENCE_ROOT/candidate-log.txt"
uname -a > "$EVIDENCE_ROOT/uname.txt"
"$TEST_PY" -m pip freeze > "$EVIDENCE_ROOT/pip-freeze.txt"
if command -v lscpu >/dev/null 2>&1; then
  lscpu > "$EVIDENCE_ROOT/lscpu.txt"
fi
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info > "$EVIDENCE_ROOT/npu-smi-info.txt" 2>&1 || true
fi

EVIDENCE_ROOT="$EVIDENCE_ROOT" "$TEST_PY" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["EVIDENCE_ROOT"])

def load(path: str) -> dict:
    return json.loads((root / path).read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "evidence-index.json":
        files.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

accuracy = load("accuracy/summary.json")
performance = load("performance/summary.json")
demo = load("demo/demo-summary.json")
ready = accuracy.get("status") == "PASS" and performance.get("status") == "PASS" and demo.get("status") == "PASS"
index = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "candidate": os.environ["CANDIDATE_LABEL"],
    "status": "READY_FOR_ARCHIVE" if ready else "NOT_READY",
    "summaries": {
        "accuracy": accuracy,
        "performance": performance,
        "demo": demo,
    },
    "files": files,
}
(root / "evidence-index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"status": index["status"], "file_count": len(files)}, indent=2))
PY

printf 'EVIDENCE_ROOT=%s\n' "$EVIDENCE_ROOT"
