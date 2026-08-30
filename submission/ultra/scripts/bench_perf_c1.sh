#!/usr/bin/env bash

set -euo pipefail
export PERFORMANCE_MATRIX=c1
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_performance.sh" "$@"
