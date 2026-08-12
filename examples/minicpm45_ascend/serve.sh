#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#===============================================================================
# serve.sh — MiniCPM-o 4.5 official-spec serve launcher for the Ascend A3 server
#
# 3-stage split pipeline (Thinker / Talker / Code2Wav) via the deploy config
# ``vllm_omni/deploy/minicpmo_4_5.yaml``, which pins the FP16 Code2Wav +
# 4-step CFM settings and Stage-0 32-token sampling:
#   connector_of_shared_memory: token2wav_float16=true, token2wav_n_timesteps=4
#   stage 0 default_sampling_params: max_tokens=32
#
# All model files are local (server has no access to the HuggingFace Hub), so
# HF_HUB_OFFLINE=1 is always set. This script is a no-op if a serve process is
# already listening on $PORT (it never kills or restarts an existing server).
#
# Usage:
#   ./serve.sh                     # defaults: port 8091, local checkpoint
#   PORT=9000 ./serve.sh           # serve on another port
#   MODEL=openbmb/MiniCPM-o-4_5 ./serve.sh   # HF-style id (needs Hub cache)
#   ./serve.sh -- --max-model-len 65536      # append extra vllm serve args
#===============================================================================
set -euo pipefail

export HF_HUB_OFFLINE=1

# --- tunables (override via env) -------------------------------------------------
MODEL="${MODEL:-/root/models/MiniCPM-o-4_5}"
PORT="${PORT:-8091}"
HOST="${HOST:-0.0.0.0}"
REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$REPO_ROOT/vllm_omni/deploy/minicpmo_4_5.yaml}"
STAGE_INIT_TIMEOUT="${STAGE_INIT_TIMEOUT:-600}"
ALLOWED_MEDIA_PATH="${ALLOWED_MEDIA_PATH:-/workspace/vllm-omni-data}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openbmb/MiniCPM-o-4_5}"

# --- don't disturb a running serve -------------------------------------------------
if ss -ltn 2>/dev/null | grep -q "[:.]${PORT} "; then
    echo "[serve.sh] port ${PORT} already in use — refusing to restart (existing serve left running)."
    exit 0
fi

echo "[serve.sh] starting MiniCPM-o 4.5 serve on ${HOST}:${PORT}"
echo "[serve.sh] deploy config: ${DEPLOY_CONFIG} (FP16 + 4-step Code2Wav, Stage-0 32 tokens)"
echo "[serve.sh] model: ${MODEL}"

exec vllm serve "${MODEL}" \
    --omni \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --deploy-config "${DEPLOY_CONFIG}" \
    --stage-init-timeout "${STAGE_INIT_TIMEOUT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --interleave-mm-strings \
    --allowed-local-media-path "${ALLOWED_MEDIA_PATH}" \
    "$@"
