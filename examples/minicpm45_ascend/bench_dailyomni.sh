#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#===============================================================================
# bench_dailyomni.sh — Daily-Omni accuracy benchmark (MiniCPM-o 4.5)
#
# Drives an already-running MiniCPM-o 4.5 server (default port 8091, see
# serve.sh) with ``vllm bench serve --omni``. Runs the full 1197-prompt
# Daily-Omni QA set at concurrency 10 and reports MCQ accuracy
# (``daily_omni_accuracy`` in the saved JSON). The server must be started with
# ``--interleave-mm-strings --allowed-local-media-path /workspace/vllm-omni-data``
# (serve.sh does this), matching the MiniCPM interleaved AV recipe (~78%).
#
# Local data:
#   /workspace/vllm-omni-data/daily-omni/qa.json
#   /workspace/vllm-omni-data/daily-omni/Videos/
#
# HF_HUB_OFFLINE=1 is required: the server cannot reach the Hub, so the QA/video
# fallback download path must never trigger.
#
# Usage:
#   ./bench_dailyomni.sh                     # defaults: port 8091, 1197 prompts, concurrency 10
#   PORT=9000 NUM_PROMPTS=200 ./bench_dailyomni.sh
#   MODEL=openbmb/MiniCPM-o-4_5 ./bench_dailyomni.sh
#   ./bench_dailyomni.sh -- --extra-bench-arg   # append extra bench args
#===============================================================================
set -euo pipefail

export HF_HUB_OFFLINE=1

# --- tunables (override via env) -------------------------------------------------
PORT="${PORT:-8091}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
QA_JSON="${QA_JSON:-/workspace/vllm-omni-data/daily-omni/qa.json}"
VIDEO_DIR="${VIDEO_DIR:-/workspace/vllm-omni-data/daily-omni/Videos}"
NUM_PROMPTS="${NUM_PROMPTS:-1197}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
PACK_MODE="minicpm-interleave"
INPUT_MODE="all"
ENDPOINT="/v1/chat/completions"
BACKEND="openai-chat-omni"
PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,itl,e2el}"
EXTRA_BODY='{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}'

echo "[bench_dailyomni.sh] Daily-Omni bench → port ${PORT}, ${NUM_PROMPTS} prompts, concurrency ${MAX_CONCURRENCY}"

exec vllm bench serve --omni \
    --port "${PORT}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --num-warmups "${NUM_WARMUPS}" \
    --dataset-name daily-omni \
    --daily-omni-qa-json "${QA_JSON}" \
    --daily-omni-video-dir "${VIDEO_DIR}" \
    --daily-omni-pack-mode "${PACK_MODE}" \
    --daily-omni-input-mode "${INPUT_MODE}" \
    --num-prompts "${NUM_PROMPTS}" \
    --no-oversample \
    --model "${MODEL}" \
    --endpoint "${ENDPOINT}" \
    --backend "${BACKEND}" \
    --percentile-metrics "${PERCENTILE_METRICS}" \
    --extra_body "${EXTRA_BODY}" \
    "$@"
