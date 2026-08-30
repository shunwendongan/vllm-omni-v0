#!/usr/bin/env bash

set -euo pipefail
export ACCURACY_SUITE=seed_tts
printf '%s\n' 'Use run_accuracy.sh for the gated full run; it executes Seed-TTS zh with WER and WavLM similarity.'
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_accuracy.sh" "$@"
