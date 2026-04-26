#!/bin/bash
# =============================================================================
# PEFTArena SFT Training Script
# =============================================================================
# Trains a model using Supervised Fine-Tuning (SFT) with VeRL's FSDP trainer.
# Supports all PEFT adapters: lora, dora, oft, ia3, vera, adalora, pissa, milora, full.
#
# Usage:
#   bash train/train_sft.sh --model Qwen/Qwen2.5-7B --adapter lora \
#       --lora_rank 8 --lora_alpha 16 --output_dir checkpoints/test
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(dirname "$SCRIPT_DIR")"
ARENA_TRAIN_PY="${ARENA_DIR}/train"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODEL=""
ADAPTER="lora"
OUTPUT_DIR=""
DATA_TRAIN=""
DATA_VAL=""
LR="2e-4"
EPOCHS="4"
NUM_GPUS="8"
BATCH_SIZE="256"
MAX_LENGTH="8192"
MICRO_BATCH_SIZE="2"
LORA_RANK="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
OFT_BLOCK_SIZE="16"
OFT_NORMALIZE_ROTATION="none"
VERA_RANK="256"
SAVE_FREQ=""
TEST_FREQ="50"
EXPERIMENT_NAME=""
WANDB_PROJECT="peft-arena"
PEFT_ADAPTER_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --adapter) ADAPTER="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --data_train) DATA_TRAIN="$2"; shift 2 ;;
        --data_val) DATA_VAL="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --max_length) MAX_LENGTH="$2"; shift 2 ;;
        --micro_batch_size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --lora_rank) LORA_RANK="$2"; shift 2 ;;
        --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --oft_block_size) OFT_BLOCK_SIZE="$2"; shift 2 ;;
        --oft_normalize_rotation) OFT_NORMALIZE_ROTATION="$2"; shift 2 ;;
        --vera_rank) VERA_RANK="$2"; shift 2 ;;
        --save_freq) SAVE_FREQ="$2"; shift 2 ;;
        --test_freq) TEST_FREQ="$2"; shift 2 ;;
        --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --peft_adapter_path) PEFT_ADAPTER_PATH="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: train_sft.sh --model <model> --output_dir <dir> [options]"
    exit 1
fi

if [ -z "$DATA_TRAIN" ] || [ -z "$DATA_VAL" ]; then
    echo "Error: --data_train and --data_val are required."
    exit 1
fi

# Auto-generate experiment name if not provided
if [ -z "$EXPERIMENT_NAME" ]; then
    MODEL_SHORT=$(basename "$MODEL")
    EXPERIMENT_NAME="sft-${ADAPTER}-${MODEL_SHORT}"
fi

# Auto-compute save_freq if not provided (save 5 times per epoch)
if [ -z "$SAVE_FREQ" ]; then
    SAVE_FREQ="91"  # reasonable default
fi

# ---------------------------------------------------------------------------
# Build torchrun command
# ---------------------------------------------------------------------------
CMD="torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    -m peft_arena_verl.trainer.fsdp_sft_trainer \
    data.train_files=${DATA_TRAIN} \
    data.val_files=${DATA_VAL} \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.filter_overlong_examples=True \
    optim.lr=${LR} \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    model.partial_pretrain=${MODEL} \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S) \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.total_epochs=${EPOCHS} \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true"

# ---------------------------------------------------------------------------
# Add adapter-specific config
# ---------------------------------------------------------------------------
case "$ADAPTER" in
    lora|dora|loraplus|rslora|qalora|keeplora)
        CMD="${CMD} \
            model.lora_rank=${LORA_RANK} \
            model.lora_alpha=${LORA_ALPHA} \
            model.lora_dropout=${LORA_DROPOUT}"
        if [ "$ADAPTER" = "dora" ]; then
            CMD="${CMD} model.use_dora=true"
        elif [ "$ADAPTER" = "loraplus" ]; then
            CMD="${CMD} model.lora_variant=loraplus"
        elif [ "$ADAPTER" = "rslora" ]; then
            CMD="${CMD} model.use_rslora=true"
        elif [ "$ADAPTER" = "qalora" ]; then
            CMD="${CMD} model.use_qalora=true"
        elif [ "$ADAPTER" = "keeplora" ]; then
            CMD="${CMD} model.lora_variant=keeplora"
        fi
        ;;
    adalora)
        CMD="${CMD} \
            model.lora_rank=${LORA_RANK} \
            model.lora_alpha=${LORA_ALPHA} \
            model.lora_dropout=${LORA_DROPOUT} \
            model.lora_variant=adalora"
        ;;
    miss)
        CMD="${CMD} \
            model.lora_rank=${LORA_RANK} \
            model.lora_alpha=${LORA_ALPHA} \
            model.lora_dropout=${LORA_DROPOUT} \
            model.lora_variant=miss"
        ;;
    pissa|milora)
        CMD="${CMD} \
            model.lora_rank=${LORA_RANK} \
            model.lora_alpha=${LORA_ALPHA} \
            model.lora_dropout=${LORA_DROPOUT}"
        if [ -n "$PEFT_ADAPTER_PATH" ]; then
            CMD="${CMD} model.peft_adapter_path=${PEFT_ADAPTER_PATH}"
        fi
        ;;
    oft)
        CMD="${CMD} \
            model.oft_block_size=${OFT_BLOCK_SIZE} \
            +model.oft_normalize_rotation=${OFT_NORMALIZE_ROTATION}"
        ;;
    ia3)
        CMD="${CMD} \
            model.lora_rank=${LORA_RANK} \
            model.lora_variant=ia3"
        ;;
    vera)
        CMD="${CMD} \
            model.lora_rank=${VERA_RANK} \
            model.lora_dropout=${LORA_DROPOUT} \
            model.lora_variant=vera"
        ;;
    full)
        # No adapter-specific config needed
        ;;
    *)
        echo "Unknown adapter type: ${ADAPTER}"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Set PYTHONPATH and run
# ---------------------------------------------------------------------------
export PYTHONPATH="${ARENA_TRAIN_PY}:${PYTHONPATH}"

echo "============================================"
echo "[PEFTArena] SFT Training"
echo "  Model:   ${MODEL}"
echo "  Adapter: ${ADAPTER}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  GPUs:    ${NUM_GPUS}"
echo "============================================"

eval $CMD
