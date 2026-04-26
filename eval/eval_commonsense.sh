#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/eval_general.sh" "$@" \
    --benchmarks "hellaswag,winogrande,xcopa" \
    --batch_size "256"
