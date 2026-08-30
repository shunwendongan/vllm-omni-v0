#!/usr/bin/env bash

set -euo pipefail
export ACCURACY_SUITE=daily_omni
printf '%s\n' 'Use run_accuracy.sh for the gated full run; it executes Daily-Omni with the official frozen parameters.'
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_accuracy.sh" "$@"
