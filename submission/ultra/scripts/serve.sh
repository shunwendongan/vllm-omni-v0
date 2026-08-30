#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
init_ultra_submission_env

PORT=${PORT:-8091}
RUN_DIR=${RUN_DIR:-$COMP_ROOT/results/$CANDIDATE_LABEL/service/$RUN_ID}
require_new_dir "$RUN_DIR"
verify_candidate_source
verify_contract_files
write_run_manifest "$RUN_DIR" serve

COMMAND=(
  "$VLLM_BIN" serve "$MODEL_PATH"
  --omni
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --deploy-config "$OFFICIAL_DEPLOY_CONFIG"
  --stage-init-timeout 900
  --host 0.0.0.0
  --port "$PORT"
  --allowed-local-media-path "$DATA_ROOT"
)
record_command "$RUN_DIR/service-command.sh" env VLLM_WORKER_MULTIPROC_METHOD=spawn "${COMMAND[@]}"
printf 'RUN_DIR=%s\n' "$RUN_DIR"
cd "$CANDIDATE_SRC"
exec env VLLM_WORKER_MULTIPROC_METHOD=spawn "${COMMAND[@]}"
