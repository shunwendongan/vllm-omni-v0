#!/usr/bin/env bash

set -euo pipefail

init_ultra_submission_env() {
  : "${COMP_ROOT:=/workspace/vllm-competition}"
  : "${MODEL_PATH:=/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5}"
  : "${DATA_ROOT:=$COMP_ROOT/data}"
  : "${OFFICIAL_SRC:=$COMP_ROOT/src/official-minicpm-challenge}"
  : "${CANDIDATE_SRC:=$COMP_ROOT/src/ultra}"
  : "${RUN_ID:=$(date +%Y%m%d-%H%M%S)}"
  : "${CANDIDATE_LABEL:=ultra}"
  : "${TEST_PY:=$COMP_ROOT/venv-a2/bin/python}"
  : "${VLLM_BIN:=$COMP_ROOT/venv-a2/bin/vllm}"
  : "${EXPECTED_CODE_BASE_COMMIT:=687af3ad5c66425ab77072cd4964164192897579}"
  : "${EXPECTED_OFFICIAL_COMMIT:=ecd9d99da0c124331861890e0371e66a01cddaa5}"
  : "${SERVED_MODEL_NAME:=openbmb/MiniCPM-o-4_5}"
  : "${LOCAL_HARDWARE:=single Atlas A2 / Ascend 910B3 NPU}"

  OFFICIAL_DEPLOY_CONFIG="$OFFICIAL_SRC/vllm_omni/deploy/minicpmo_4_5.yaml"
  FROZEN_DEPLOY_CONFIG="$CANDIDATE_SRC/submission/ultra/configs/official-deploy-config.yaml"
  export COMP_ROOT MODEL_PATH DATA_ROOT OFFICIAL_SRC CANDIDATE_SRC RUN_ID
  export CANDIDATE_LABEL TEST_PY VLLM_BIN EXPECTED_CODE_BASE_COMMIT
  export EXPECTED_OFFICIAL_COMMIT SERVED_MODEL_NAME LOCAL_HARDWARE
  export OFFICIAL_DEPLOY_CONFIG FROZEN_DEPLOY_CONFIG
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  test -f "$1" || die "required file not found: $1"
}

require_dir() {
  test -d "$1" || die "required directory not found: $1"
}

require_executable() {
  test -x "$1" || die "required executable not found: $1"
}

require_new_dir() {
  test ! -e "$1" || die "refusing to overwrite existing evidence: $1"
  mkdir -p "$1"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_contract_files() {
  require_dir "$OFFICIAL_SRC/.git"
  require_file "$OFFICIAL_DEPLOY_CONFIG"
  require_file "$FROZEN_DEPLOY_CONFIG"

  local official_head
  official_head=$(git -C "$OFFICIAL_SRC" rev-parse HEAD)
  test "$official_head" = "$EXPECTED_OFFICIAL_COMMIT" || \
    die "official source is $official_head, expected $EXPECTED_OFFICIAL_COMMIT"
  cmp -s "$OFFICIAL_DEPLOY_CONFIG" "$FROZEN_DEPLOY_CONFIG" || \
    die "official deploy config differs from the frozen submission copy"
}

verify_candidate_repository() {
  require_dir "$CANDIDATE_SRC/.git"
  require_executable "$TEST_PY"
  require_executable "$VLLM_BIN"

  git -C "$CANDIDATE_SRC" merge-base --is-ancestor \
    "$EXPECTED_CODE_BASE_COMMIT" HEAD || \
    die "candidate HEAD does not derive from $EXPECTED_CODE_BASE_COMMIT"

  if test "${ALLOW_DIRTY:-0}" != "1" && \
     test -n "$(git -C "$CANDIDATE_SRC" status --porcelain)"; then
    git -C "$CANDIDATE_SRC" status --short >&2
    die "candidate worktree is dirty; collect the diff and clean it before a formal run"
  fi

}

verify_candidate_source() {
  verify_candidate_repository
  local imported
  imported=$(
    cd "$CANDIDATE_SRC"
    "$TEST_PY" -c 'from pathlib import Path; import vllm_omni; print(Path(vllm_omni.__file__).resolve())'
  )
  imported=$(printf '%s\n' "$imported" | tail -n 1)
  case "$imported" in
    "$CANDIDATE_SRC"/*) ;;
    *) die "vllm_omni imports from $imported, not $CANDIDATE_SRC" ;;
  esac
}

write_run_manifest() {
  local run_dir=$1
  local action=$2
  local candidate_head official_head deploy_sha
  candidate_head=$(git -C "$CANDIDATE_SRC" rev-parse HEAD)
  official_head=$(git -C "$OFFICIAL_SRC" rev-parse HEAD)
  deploy_sha=$(sha256_file "$OFFICIAL_DEPLOY_CONFIG")

  ACTION_NAME="$action" CANDIDATE_HEAD="$candidate_head" \
  OFFICIAL_HEAD="$official_head" DEPLOY_SHA="$deploy_sha" \
  "$TEST_PY" - "$run_dir/manifest.json" <<'PY'
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "action": os.environ["ACTION_NAME"],
    "run_id": os.environ["RUN_ID"],
    "candidate_label": os.environ["CANDIDATE_LABEL"],
    "candidate_source": os.environ["CANDIDATE_SRC"],
    "candidate_head": os.environ["CANDIDATE_HEAD"],
    "candidate_code_base": os.environ["EXPECTED_CODE_BASE_COMMIT"],
    "official_source": os.environ["OFFICIAL_SRC"],
    "official_head": os.environ["OFFICIAL_HEAD"],
    "official_deploy_config": os.environ["OFFICIAL_DEPLOY_CONFIG"],
    "official_deploy_config_sha256": os.environ["DEPLOY_SHA"],
    "model_path": os.environ["MODEL_PATH"],
    "data_root": os.environ["DATA_ROOT"],
    "hardware_label": os.environ["LOCAL_HARDWARE"],
    "python": sys.version,
    "platform": platform.platform(),
}
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

record_command() {
  local output=$1
  shift
  printf '%q ' "$@" > "$output"
  printf '\n' >> "$output"
}

wait_for_health() {
  local pid=$1
  local port=$2
  local log_file=$3
  local attempts=${4:-90}
  local attempt

  for attempt in $(seq 1 "$attempts"); do
    if curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      printf 'service healthy on port %s after attempt %s/%s\n' "$port" "$attempt" "$attempts"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || {
      tail -n 160 "$log_file" >&2 || true
      die "service process $pid exited before becoming healthy"
    }
    printf 'waiting for service: %s/%s\n' "$attempt" "$attempts"
    sleep 10
  done
  tail -n 160 "$log_file" >&2 || true
  die "service did not become healthy on port $port"
}

collect_process_tree() {
  local parent_pid=$1
  local child_pid
  for child_pid in $(pgrep -P "$parent_pid" 2>/dev/null || true); do
    collect_process_tree "$child_pid"
  done
  printf '%s\n' "$parent_pid"
}

stop_process_tree() {
  local root_pid=${1:-}
  test -n "$root_pid" || return 0
  [[ "$root_pid" =~ ^[0-9]+$ ]] || die "invalid process id: $root_pid"
  kill -0 "$root_pid" 2>/dev/null || return 0

  local process_tree
  process_tree=$(collect_process_tree "$root_pid")
  kill -TERM $process_tree 2>/dev/null || true
  local attempt
  for attempt in $(seq 1 20); do
    kill -0 "$root_pid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL $process_tree 2>/dev/null || true
}

start_candidate_service() {
  local run_dir=$1
  local port=$2
  local log_file="$run_dir/service.log"

  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    die "port $port already has a healthy service"
  fi

  local -a command=(
    "$VLLM_BIN" serve "$MODEL_PATH"
    --omni
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --deploy-config "$OFFICIAL_DEPLOY_CONFIG"
    --stage-init-timeout 900
    --host 0.0.0.0
    --port "$port"
    --allowed-local-media-path "$DATA_ROOT"
  )
  record_command "$run_dir/service-command.sh" env VLLM_WORKER_MULTIPROC_METHOD=spawn "${command[@]}"
  (
    cd "$CANDIDATE_SRC"
    nohup env VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "${command[@]}" > "$log_file" 2>&1 &
    printf '%s\n' "$!" > "$run_dir/service.pid"
  )
  cat "$run_dir/service.pid"
}
