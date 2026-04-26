#!/bin/bash
# =============================================================================
# PEFTArena Math Evaluation Script
# =============================================================================
# Evaluates a model checkpoint on math benchmarks using vLLM.
# Adapted from limit-of-rl/math/examples/math_eval.
#
# Usage:
#   bash eval/eval_math.sh --checkpoint_path <path> --output_dir <dir>
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(dirname "$SCRIPT_DIR")"
MATH_EVAL_DIR="${ARENA_DIR}/third_party/math_eval"
MERGE_SCRIPT="${ARENA_DIR}/tools/merge_peft.py"
PREPARE_SCRIPT="${ARENA_DIR}/tools/prepare_eval_checkpoint.py"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CHECKPOINT_PATH=""
OUTPUT_DIR=""
NUM_GPUS="1"
#DATA_NAMES="gsm8k,math,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23"
DATA_NAMES="math500,aime24,amc23"
PROMPT_TYPE="cot"
TEMPERATURE="0.6"
NUM_SAMPLES="-1"
SEED="0"
N_SAMPLING="16"
MAX_TOKENS_PER_CALL="8192"
GPU_MEMORY_UTILIZATION="0.7"
MAX_MODEL_LEN="9216"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --data_names) DATA_NAMES="$2"; shift 2 ;;
        --prompt_type) PROMPT_TYPE="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --n_sampling) N_SAMPLING="$2"; shift 2 ;;
        --max_tokens_per_call) MAX_TOKENS_PER_CALL="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: eval_math.sh --checkpoint_path <path> [--output_dir <dir>] [--num_gpus N]"
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

is_valid_math_model_dir() {
    local model_dir="$1"
    [ -d "$model_dir" ] || return 1
    [ -f "${model_dir}/config.json" ] || return 1
    if [ ! -f "${model_dir}/model.safetensors" ] && \
       [ ! -f "${model_dir}/model.safetensors.index.json" ] && \
       [ ! -f "${model_dir}/pytorch_model.bin" ] && \
       [ ! -f "${model_dir}/pytorch_model.bin.index.json" ]; then
        return 1
    fi
    [ -f "${model_dir}/tokenizer_config.json" ] || return 1
    [ -f "${model_dir}/vocab.json" ] || return 1
    [ -f "${model_dir}/merges.txt" ] || return 1
}

MODEL_NAME="$(derive_model_name "$CHECKPOINT_PATH")"

OUTPUT_DIR="$(resolve_output_dir "$OUTPUT_DIR" "math" "$MODEL_NAME")"

mkdir -p "$OUTPUT_DIR"
echo "[PEFTArena] Output path: ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Prepare checkpoint for eval and merge adapter if needed
# ---------------------------------------------------------------------------
MODEL_PATH="$(python "${PREPARE_SCRIPT}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --peft_export_mode adapter | tail -n 1)"
CLEANUP_MERGED=""

echo "[PEFTArena] Eval-ready checkpoint: ${MODEL_PATH}"

if [ -f "${MODEL_PATH}/adapter_model.safetensors" ] || [ -f "${MODEL_PATH}/adapter_model.bin" ]; then
    echo "[PEFTArena] Detected PEFT adapter checkpoint. Merging into base model..."
    MERGED_PATH="${MODEL_PATH}_merged"

    if ! is_valid_math_model_dir "$MERGED_PATH"; then
        rm -rf "$MERGED_PATH"
        python "${MERGE_SCRIPT}" \
            --adapter_path "${MODEL_PATH}" \
            --output_dir "${MERGED_PATH}" \
            --torch_dtype bfloat16
    else
        echo "[PEFTArena] Merged model already exists at ${MERGED_PATH}, skipping merge."
    fi

    MODEL_PATH="$MERGED_PATH"
    CLEANUP_MERGED="$MERGED_PATH"
fi

# ---------------------------------------------------------------------------
# Set GPU visibility
# ---------------------------------------------------------------------------
GPU_LIST=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
    if [ -z "$GPU_LIST" ]; then
        GPU_LIST="$i"
    else
        GPU_LIST="${GPU_LIST},${i}"
    fi
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}"

# ---------------------------------------------------------------------------
# Run math evaluation
# ---------------------------------------------------------------------------
export PYTHONPATH="${MATH_EVAL_DIR}:${PYTHONPATH}"

echo "============================================"
echo "[PEFTArena] Math Evaluation"
echo "  Model:       ${MODEL_PATH}"
echo "  Datasets:    ${DATA_NAMES}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Temperature: ${TEMPERATURE}"
echo "  N_Sampling:  ${N_SAMPLING}"
echo "  Max Tokens:  ${MAX_TOKENS_PER_CALL}"
echo "  vLLM Mem:    ${GPU_MEMORY_UTILIZATION}"
echo "  Model Len:   ${MAX_MODEL_LEN}"
echo "============================================"

python "${MATH_EVAL_DIR}/math_eval.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --data_names "${DATA_NAMES}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_type "${PROMPT_TYPE}" \
    --temperature "${TEMPERATURE}" \
    --num_test_sample "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --n_sampling "${N_SAMPLING}" \
    --max_tokens_per_call "${MAX_TOKENS_PER_CALL}" \
    --dtype bfloat16 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --use_vllm \
    --save_outputs \
    --apply_chat_template

echo "[PEFTArena] Math evaluation completed. Results saved to ${OUTPUT_DIR}"

# Clean up merged model if we created it
if [ -n "$CLEANUP_MERGED" ] && [ -d "$CLEANUP_MERGED" ]; then
    echo "[PEFTArena] Cleaning up merged model at ${CLEANUP_MERGED}"
    rm -rf "$CLEANUP_MERGED"
fi
