#!/usr/bin/env bash

set -euo pipefail
export ACCURACY_SUITE=videomme
printf '%s\n' 'Use run_accuracy.sh for the gated full run; it executes Video-MME with the official frozen parameters.'
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_accuracy.sh" "$@"
