#!/bin/bash
# =============================================================================
# PEFTArena General Ability Evaluation Script
# =============================================================================
# Evaluates a model checkpoint on general benchmarks using OpenCompass.
#
# This script automatically:
#   1. Detects PEFT adapters and merges them into the base model
#   2. Auto-detects instruct vs base model for correct OpenCompass class
#   3. Generates the appropriate OpenCompass config file
#   4. Runs OpenCompass evaluation
#
# Model class selection:
#   - BBH / HumanEval / HellaSwag / WinoGrande / MMLU / ARC / GSM8K / XCOPA:
#       base model → VLLM
#       instruct model → VLLMwithChatTemplate
#   - IFEval + NQ: always → VLLMwithChatTemplate
#
# Usage:
#   bash eval/eval_general.sh --checkpoint_path <path>
#   bash eval/eval_general.sh --checkpoint_path <path> --benchmarks bbh
#   bash eval/eval_general.sh --checkpoint_path <path> --benchmarks ifeval_nq
#   bash eval/eval_general.sh --checkpoint_path <path> --benchmarks humaneval
#   bash eval/eval_general.sh --checkpoint_path <path> --benchmarks hellaswag,winogrande,xcopa
#   bash eval/eval_general.sh --checkpoint_path <path> --benchmarks mmlu,arc,gsm8k
#   bash eval/eval_general.sh --checkpoint_path <path> --benchmarks bbh,ifeval_nq,humaneval
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(dirname "$SCRIPT_DIR")"
OPENCOMPASS_DIR="${ARENA_DIR}/third_party/opencompass"
MERGE_SCRIPT="${ARENA_DIR}/tools/merge_peft.py"
CONFIG_GEN_SCRIPT="${ARENA_DIR}/eval/generate_opencompass_config.py"
PREPARE_SCRIPT="${ARENA_DIR}/tools/prepare_eval_checkpoint.py"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CHECKPOINT_PATH=""
OUTPUT_DIR=""
NUM_GPUS="1"
BENCHMARKS="bbh,ifeval_nq"
BATCH_SIZE="256"
ABBR=""
KEEP_MERGED="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --abbr) ABBR="$2"; shift 2 ;;
        --keep_merged) KEEP_MERGED="true"; shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: eval_general.sh --checkpoint_path <path> [options]"
    echo ""
    echo "Options:"
    echo "  --benchmarks    Comma-separated benchmark list."
    echo "                  Supported: bbh, ifeval_nq, humaneval, hellaswag,"
    echo "                             winogrande, mmlu, arc, gsm8k, xcopa"
    echo "                  Default: bbh,ifeval_nq"
    echo "  --output_dir    Results output directory"
    echo "  --num_gpus      Number of GPUs (default: 1)"
    echo "  --batch_size    Batch size (default: 256)"
    echo "  --abbr          Short model name for results"
    echo "  --keep_merged   Don't clean up merged model after eval"
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

is_valid_hf_model_dir() {
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
}

if [ -z "$ABBR" ]; then
    ABBR="$(derive_model_name "$CHECKPOINT_PATH")"
fi

MODEL_NAME="${ABBR}"

OUTPUT_DIR="$(resolve_output_dir "$OUTPUT_DIR" "general" "$MODEL_NAME")"

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

    if ! is_valid_hf_model_dir "$MERGED_PATH"; then
        rm -rf "$MERGED_PATH"
        python "${MERGE_SCRIPT}" \
            --adapter_path "${MODEL_PATH}" \
            --output_dir "${MERGED_PATH}" \
            --torch_dtype bfloat16
    else
        echo "[PEFTArena] Merged model already exists at ${MERGED_PATH}, skipping merge."
    fi

    MODEL_PATH="$MERGED_PATH"

    if [ "$KEEP_MERGED" = "false" ]; then
        CLEANUP_MERGED="$MERGED_PATH"
    fi
fi

# ---------------------------------------------------------------------------
# Generate OpenCompass config automatically
# ---------------------------------------------------------------------------
CONFIG_DIR="${OUTPUT_DIR}/configs"
mkdir -p "$CONFIG_DIR"
CONFIG_BASE="${CONFIG_DIR}/${ABBR}"

echo "[PEFTArena] Generating OpenCompass config..."
CONFIG_FILES="$(python "${CONFIG_GEN_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --benchmarks "${BENCHMARKS}" \
    --output_config "${CONFIG_BASE}.py" \
    --abbr "${ABBR}" \
    --num_gpus "${NUM_GPUS}" \
    --batch_size "${BATCH_SIZE}" \
    2>&1 | tee /dev/stderr | grep "Config →" | awk '{print $NF}' || true)"

# Fallback: if CONFIG_FILES is empty, use generated config files under CONFIG_DIR
if [ -z "$CONFIG_FILES" ]; then
    if [ -f "${CONFIG_BASE}.py" ]; then
        CONFIG_FILES="${CONFIG_BASE}.py"
    else
        mapfile -t CONFIG_FILES_ARRAY < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name "${ABBR}*.py" | sort)
        if [ "${#CONFIG_FILES_ARRAY[@]}" -gt 0 ]; then
            CONFIG_FILES="${CONFIG_FILES_ARRAY[*]}"
        else
            echo "[PEFTArena] ERROR: No config files generated."
            exit 1
        fi
    fi
fi

if [ -z "$CONFIG_FILES" ]; then
    echo "[PEFTArena] ERROR: No config files generated."
    exit 1
fi

if [ -n "$CONFIG_FILES" ]; then
    echo "[PEFTArena] Generated config files:"
    for CONFIG_FILE in $CONFIG_FILES; do
        echo "  ${CONFIG_FILE}"
    done
fi

# ---------------------------------------------------------------------------
# Run OpenCompass evaluation for each config
# ---------------------------------------------------------------------------
export PYTHONPATH="${OPENCOMPASS_DIR}:${PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo ""
echo "============================================"
echo "[PEFTArena] General Ability Evaluation"
echo "  Model:      ${MODEL_PATH}"
echo "  Benchmarks: ${BENCHMARKS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  GPUs:       ${NUM_GPUS}"
echo "============================================"

for CONFIG_FILE in $CONFIG_FILES; do
    if [ ! -f "$CONFIG_FILE" ]; then
        continue
    fi

    CONFIG_NAME=$(basename "$CONFIG_FILE" .py)
    echo ""
    echo "[PEFTArena] Running OpenCompass with config: ${CONFIG_NAME}"

    cd "${OPENCOMPASS_DIR}"
    python -m opencompass.cli.main "${CONFIG_FILE}" \
        -w "${OUTPUT_DIR}/${CONFIG_NAME}" \
        -r latest

    echo "[PEFTArena] Completed: ${CONFIG_NAME}"
done

echo ""
echo "[PEFTArena] General evaluation completed. Results saved to ${OUTPUT_DIR}"

# Clean up merged model if we created it
if [ -n "$CLEANUP_MERGED" ] && [ -d "$CLEANUP_MERGED" ]; then
    echo "[PEFTArena] Cleaning up merged model at ${CLEANUP_MERGED}"
    rm -rf "$CLEANUP_MERGED"
fi
