#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/eval_general.sh" "$@" \
    --benchmarks "mmlu,arc,gsm8k" \
    --batch_size "128"
