#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#===============================================================================
# bench_seedtts.sh — Seed-TTS performance + WER benchmark (MiniCPM-o 4.5)
#
# Drives an already-running MiniCPM-o 4.5 server (default port 8091, see
# serve.sh) with ``vllm bench serve --omni``. Runs 32 prompts at concurrency 1,
# captures ttft/tpot/itl/e2el plus audio_ttfp/audio_rtf percentiles, and scores
# the synthesized speech with Whisper (WER) via ``--seed-tts-wer-eval``.
#
# The dataset lives under /root/seed-tts-eval/seedtts_testset (en/meta.lst +
# prompt-wavs). HF_HUB_OFFLINE=1 is required: the server cannot reach the Hub,
# so any fallback download would hang or fail.
#
# Usage:
#   ./bench_seedtts.sh                    # defaults: port 8091, 32 prompts, concurrency 1
#   PORT=9000 NUM_PROMPTS=64 ./bench_seedtts.sh
#   MODEL=openbmb/MiniCPM-o-4_5 ./bench_seedtts.sh
#   ./bench_seedtts.sh -- --extra-bench-arg   # append extra bench args
#===============================================================================
set -euo pipefail

export HF_HUB_OFFLINE=1

# --- tunables (override via env) -------------------------------------------------
PORT="${PORT:-8091}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
DATASET_PATH="${DATASET_PATH:-/root/seed-tts-eval/seedtts_testset}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
ENDPOINT="/v1/chat/completions"
BACKEND="openai-chat-omni"
PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,itl,e2el,audio_ttfp,audio_rtf}"
EXTRA_BODY='{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}'

echo "[bench_seedtts.sh] Seed-TTS bench → port ${PORT}, ${NUM_PROMPTS} prompts, concurrency ${MAX_CONCURRENCY}"

exec vllm bench serve --omni \
    --port "${PORT}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --num-warmups "${NUM_WARMUPS}" \
    --dataset-name seed-tts \
    --dataset-path "${DATASET_PATH}" \
    --num-prompts "${NUM_PROMPTS}" \
    --no-oversample \
    --seed-tts-wer-eval \
    --seed-tts-wer-save-items \
    --model "${MODEL}" \
    --endpoint "${ENDPOINT}" \
    --backend "${BACKEND}" \
    --percentile-metrics "${PERCENTILE_METRICS}" \
    --extra_body "${EXTRA_BODY}" \
    "$@"
