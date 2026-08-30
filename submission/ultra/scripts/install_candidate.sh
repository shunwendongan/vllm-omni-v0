#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
init_ultra_submission_env

RUN_DIR=${RUN_DIR:-$COMP_ROOT/results/$CANDIDATE_LABEL/install/$RUN_ID}
require_new_dir "$RUN_DIR"
verify_candidate_repository
write_run_manifest "$RUN_DIR" install-candidate

git -C "$CANDIDATE_SRC" status --short > "$RUN_DIR/git-status.txt"
git -C "$CANDIDATE_SRC" diff --binary > "$RUN_DIR/worktree.diff"
git -C "$CANDIDATE_SRC" diff --cached --binary > "$RUN_DIR/index.diff"
"$TEST_PY" -m pip --version > "$RUN_DIR/pip-version.txt"
"$TEST_PY" -m pip freeze > "$RUN_DIR/pip-freeze-before.txt"

INSTALL_COMMAND=(
  "$TEST_PY" -m pip install
  --editable "$CANDIDATE_SRC"
  --no-build-isolation
  --no-deps
)
record_command "$RUN_DIR/install-command.sh" "${INSTALL_COMMAND[@]}"
"${INSTALL_COMMAND[@]}" 2>&1 | tee "$RUN_DIR/pip-install.log"

(
  cd "$CANDIDATE_SRC"
  "$TEST_PY" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import vllm_omni

source = Path(vllm_omni.__file__).resolve()
expected = Path.cwd().resolve()
print("vllm-omni version:", version("vllm-omni"))
print("vllm_omni source:", source)
if expected not in source.parents:
    raise SystemExit(f"candidate import mismatch: {source} is not under {expected}")
print("candidate editable import: OK")
PY
) | tee "$RUN_DIR/import-check.txt"

"$TEST_PY" -m pip check > "$RUN_DIR/pip-check.txt" 2>&1 || true
"$TEST_PY" -m pip freeze > "$RUN_DIR/pip-freeze-after.txt"
printf 'INSTALL_RESULT=PASS\nRUN_DIR=%s\n' "$RUN_DIR"
