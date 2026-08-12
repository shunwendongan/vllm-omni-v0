#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#===============================================================================
# bench_videomme.sh — Video-MME accuracy benchmark (MiniCPM-o 4.5)
#
# Drives an already-running MiniCPM-o 4.5 server (default port 8091, see
# serve.sh) with ``vllm bench serve --omni``. Runs the 2700-prompt Video-MME
# QA set at concurrency 4 and reports MCQ accuracy (``videomme_accuracy`` in the
# saved JSON). Uses the OmniEvalKit MiniCPM recipe: ``minicpm-frames`` packing,
# max 96 frames, text-only modalities. Official MiniCPM-o 4.5 reports 70.4
# (w/o subs).
#
# Local data:
#   /workspace/vllm-omni-data/videomme/videomme/test-00000-of-00001.parquet
#   /workspace/vllm-omni-data/videomme/video/
#
# The server must be started with
# ``--allowed-local-media-path /workspace/vllm-omni-data`` (serve.sh does this)
# so sampled frames can be referenced as file:// URLs.
#
# HF_HUB_OFFLINE=1 is required: the server cannot reach the Hub.
#
# Usage:
#   ./bench_videomme.sh                     # defaults: port 8091, 2700 prompts, concurrency 4
#   PORT=9000 NUM_PROMPTS=200 ./bench_videomme.sh
#   MODEL=openbmb/MiniCPM-o-4_5 ./bench_videomme.sh
#   ./bench_videomme.sh -- --extra-bench-arg   # append extra bench args
#===============================================================================
set -euo pipefail

export HF_HUB_OFFLINE=1

# --- tunables (override via env) -------------------------------------------------
PORT="${PORT:-8091}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
VIDEOMME_PARQUET="${VIDEOMME_PARQUET:-/workspace/vllm-omni-data/videomme/videomme/test-00000-of-00001.parquet}"
VIDEOMME_VIDEO_DIR="${VIDEOMME_VIDEO_DIR:-/workspace/vllm-omni-data/videomme/video}"
NUM_PROMPTS="${NUM_PROMPTS:-2700}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
PACK_MODE="minicpm-frames"
MAX_FRAMES="${MAX_FRAMES:-96}"
ENDPOINT="/v1/chat/completions"
BACKEND="openai-chat-omni"
PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,itl,e2el}"
EXTRA_BODY='{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}'

echo "[bench_videomme.sh] Video-MME bench → port ${PORT}, ${NUM_PROMPTS} prompts, concurrency ${MAX_CONCURRENCY}"

exec vllm bench serve --omni \
    --port "${PORT}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --num-warmups "${NUM_WARMUPS}" \
    --dataset-name videomme \
    --videomme-parquet "${VIDEOMME_PARQUET}" \
    --videomme-video-dir "${VIDEOMME_VIDEO_DIR}" \
    --videomme-pack-mode "${PACK_MODE}" \
    --videomme-max-frames "${MAX_FRAMES}" \
    --num-prompts "${NUM_PROMPTS}" \
    --no-oversample \
    --model "${MODEL}" \
    --endpoint "${ENDPOINT}" \
    --backend "${BACKEND}" \
    --percentile-metrics "${PERCENTILE_METRICS}" \
    --extra_body "${EXTRA_BODY}" \
    "$@"
