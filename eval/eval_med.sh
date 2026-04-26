#!/bin/bash
# =============================================================================
# PEFTArena Medical Evaluation Script
# =============================================================================
# Evaluates a model checkpoint on medical benchmarks using med_eval_ours.
# Uses vLLM for inference with custom prompt templates.
#
# Usage:
#   bash eval/eval_med.sh --checkpoint_path <path> --output_dir <dir>
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(dirname "$SCRIPT_DIR")"
MED_EVAL_DIR="${ARENA_DIR}/third_party/med_eval"
PREPARE_SCRIPT="${ARENA_DIR}/tools/prepare_eval_checkpoint.py"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CHECKPOINT_PATH=""
OUTPUT_DIR=""
NUM_GPUS="1"
TEMPERATURE="0.0"
GPU_MEMORY_UTILIZATION="0.8"
MAX_TOKENS="8192"
SEED="42"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: eval_med.sh --checkpoint_path <path> [--output_dir <dir>] [--num_gpus N]"
    exit 1
fi

derive_model_name() {
    local checkpoint_path="$1"
    local normalized
    normalized="$(python -c 'import os,sys; print(os.path.normpath(os.path.abspath(sys.argv[1])))' "$checkpoint_path")"
    local rel_path="$normalized"
    case "$rel_path" in
        */release_ckpts/*) rel_path="${rel_path#*/release_ckpts/}" ;;
        */checkpoints/*) rel_path="${rel_path#*/checkpoints/}" ;;
        release_ckpts/*) rel_path="${rel_path#release_ckpts/}" ;;
        checkpoints/*) rel_path="${rel_path#checkpoints/}" ;;
        "${ARENA_DIR}/"*) rel_path="${rel_path#"${ARENA_DIR}/"}" ;;
    esac
    echo "$rel_path" | tr '/' '_' | sed 's/_global_step_/_/g'
}

resolve_output_dir() {
    local requested="$1"
    local domain="$2"
    local model_name="$3"
    if [ -z "$requested" ]; then
        echo "${ARENA_DIR}/results/${model_name}/${domain}"
        return
    fi
    requested="${requested%/}"
    local base_name
    base_name="$(basename "$requested")"
    local parent_name
    parent_name="$(basename "$(dirname "$requested")")"
    if [ "$base_name" = "$domain" ]; then
        if [ "$parent_name" = "$model_name" ]; then
            echo "$requested"
        else
            echo "$(dirname "$requested")/${model_name}/${domain}"
        fi
        return
    fi
    if [ "$base_name" = "$model_name" ]; then
        echo "${requested}/${domain}"
        return
    fi
    if [ "$base_name" = "results" ]; then
        echo "${requested}/${model_name}/${domain}"
        return
    fi
    echo "$requested"
}

MODEL_NAME="$(derive_model_name "$CHECKPOINT_PATH")"

OUTPUT_DIR="$(resolve_output_dir "$OUTPUT_DIR" "med" "$MODEL_NAME")"

mkdir -p "$OUTPUT_DIR"
echo "[PEFTArena] Output path: ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Prepare checkpoint for eval, then run medical evaluation
# ---------------------------------------------------------------------------
MODEL_PATH="$(python "${PREPARE_SCRIPT}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --peft_export_mode adapter | tail -n 1)"

echo "[PEFTArena] Eval-ready checkpoint: ${MODEL_PATH}"

# med_eval main.py handles PEFT adapter merging internally if the prepared
# checkpoint is still an adapter directory.

export PYTHONPATH="${MED_EVAL_DIR}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

MODEL_SAVE_NAME="${MODEL_NAME}"

EVAL_FILE="${MED_EVAL_DIR}/data/m1_eval_data_processed.json"

echo "============================================"
echo "[PEFTArena] Medical Evaluation"
echo "  Model:       ${MODEL_PATH}"
echo "  Eval File:   ${EVAL_FILE}"
echo "  Output Name: ${MODEL_SAVE_NAME}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Max Tokens:  ${MAX_TOKENS}"
echo "============================================"

# Note: main.py writes results to ./results/medical/<model_save_name>/
# We cd into the output dir so results land there.
cd "${OUTPUT_DIR}"

python "${MED_EVAL_DIR}/main.py" \
    --model_name "${MODEL_PATH}" \
    --eval_file "${EVAL_FILE}" \
    --temperature "${TEMPERATURE}" \
    --tensor_parallel_size "${NUM_GPUS}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_tokens "${MAX_TOKENS}" \
    --model_save_name "${MODEL_SAVE_NAME}" \
    --output_file_name "${MODEL_SAVE_NAME}" \
    --output_dir "${OUTPUT_DIR}"

echo "[PEFTArena] Medical evaluation completed. Results saved under ${OUTPUT_DIR}/results/medical/${MODEL_SAVE_NAME}/"
