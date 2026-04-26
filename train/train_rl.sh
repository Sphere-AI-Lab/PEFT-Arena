#!/bin/bash
# =============================================================================
# PEFTArena RL (GRPO) Training Script
# =============================================================================
# Trains a model using Reinforcement Learning (GRPO) with VeRL framework.
#
# Usage:
#   bash train/train_rl.sh --model Qwen/Qwen2.5-7B-Instruct --adapter lora \
#       --lora_rank 16 --lora_alpha 32 --output_dir checkpoints/grpo-test
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(dirname "$SCRIPT_DIR")"
VERL_DIR="${ARENA_DIR}/third_party/verl"
ARENA_TRAIN_PY="${ARENA_DIR}/train"
ARENA_PYTHON="${ARENA_DIR}/conda_env/bin/python3.10"

export RAY_USE_MULTIPROCESSING_CPU_COUNT="${RAY_USE_MULTIPROCESSING_CPU_COUNT:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-DEBUG}"

if [ ! -x "${ARENA_PYTHON}" ]; then
    ARENA_PYTHON="python3"
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODEL=""
ADAPTER="lora"
OUTPUT_DIR=""
DATA_TRAIN="${ARENA_DIR}/data/openr1-50k/train.parquet"
DATA_VAL="${ARENA_DIR}/data/openr1-50k/test.parquet"
LR=""
EPOCHS=""
NUM_GPUS="8"
BATCH_SIZE="256"
MAX_LENGTH="8192"
PROMPT_MAX_LENGTH="1024"
MICRO_BATCH_SIZE="2"
PPO_MICRO_BATCH_SIZE_PER_GPU=""
PPO_MINI_BATCH_SIZE=""
ROLLOUT_N="8"
GPU_MEMORY_UTILIZATION="0.7"
MAX_NUM_BATCHED_TOKENS="9216"
ROLLOUT_UPDATE_BUCKET_MB="4096"
LORA_RANK="16"
LORA_ALPHA="32"
LORA_DROPOUT="0.05"
OFT_BLOCK_SIZE="16"
OFT_NORMALIZE_ROTATION="none"
VERA_RANK="256"
EXPERIMENT_NAME=""
WANDB_PROJECT="peft-arena-rl"
PEFT_ADAPTER_PATH=""
SAVE_FREQ="50"
TEST_FREQ="1000"
TARGET_MODULES=""
QWEN_RL_TARGET_MODULES="['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']"

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
        --prompt_max_length) PROMPT_MAX_LENGTH="$2"; shift 2 ;;
        --micro_batch_size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --ppo_micro_batch_size_per_gpu) PPO_MICRO_BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
        --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
        --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max_num_batched_tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
        --rollout_update_bucket_mb) ROLLOUT_UPDATE_BUCKET_MB="$2"; shift 2 ;;
        --lora_rank) LORA_RANK="$2"; shift 2 ;;
        --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --oft_block_size) OFT_BLOCK_SIZE="$2"; shift 2 ;;
        --oft_normalize_rotation) OFT_NORMALIZE_ROTATION="$2"; shift 2 ;;
        --vera_rank) VERA_RANK="$2"; shift 2 ;;
        --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --peft_adapter_path) PEFT_ADAPTER_PATH="$2"; shift 2 ;;
        --save_freq) SAVE_FREQ="$2"; shift 2 ;;
        --test_freq) TEST_FREQ="$2"; shift 2 ;;
        --target_modules) TARGET_MODULES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: train_rl.sh --model <model> --output_dir <dir> [options]"
    exit 1
fi

if [ ! -f "$DATA_TRAIN" ]; then
    echo "Error: training data not found: ${DATA_TRAIN}"
    exit 1
fi

if [ ! -f "$DATA_VAL" ]; then
    echo "Error: validation data not found: ${DATA_VAL}"
    exit 1
fi

if [ -z "$LR" ]; then
    if [ "$ADAPTER" = "full" ]; then
        LR="1e-6"
    else
        LR="1e-5"
    fi
fi

if [ -z "$EPOCHS" ]; then
    EPOCHS="10"
fi

if [ -z "$PPO_MICRO_BATCH_SIZE_PER_GPU" ]; then
    if [ "$ADAPTER" = "full" ]; then
        PPO_MICRO_BATCH_SIZE_PER_GPU="8"
    else
        PPO_MICRO_BATCH_SIZE_PER_GPU="64"
    fi
fi

if [ -z "$PPO_MINI_BATCH_SIZE" ]; then
    PPO_MINI_BATCH_SIZE="64"
fi

MODEL_LOWER=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')
if [ -z "$TARGET_MODULES" ] && [[ "$MODEL_LOWER" == *"qwen"* ]]; then
    TARGET_MODULES="$QWEN_RL_TARGET_MODULES"
fi

if [ -z "$EXPERIMENT_NAME" ]; then
    MODEL_SHORT=$(basename "$MODEL")
    EXPERIMENT_NAME="grpo-${ADAPTER}-${MODEL_SHORT}"
fi

# ---------------------------------------------------------------------------
# Build the GRPO training command
# ---------------------------------------------------------------------------
# NOTE: This script provides the general structure. The actual GRPO training
# config depends on the VeRL version and reward model setup. Users should
# customize the config file or extend this script based on their VeRL setup.

GRPO_SCRIPT="${VERL_DIR}/grpo_train/hpt_train.sh"

if [ ! -f "$GRPO_SCRIPT" ]; then
    echo "Warning: Default GRPO script not found at ${GRPO_SCRIPT}"
    echo "Falling back to inline GRPO config..."

    CMD="${ARENA_PYTHON} -m peft_arena_verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        critic.enable=false \
        data.train_files=['${DATA_TRAIN}'] \
        data.val_files=['${DATA_VAL}'] \
        data.train_batch_size=${BATCH_SIZE} \
        data.max_prompt_length=${PROMPT_MAX_LENGTH} \
        data.max_response_length=${MAX_LENGTH} \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        reward_model.reward_manager=prime \
        actor_rollout_ref.model.path=${MODEL} \
        actor_rollout_ref.actor.optim.lr=${LR} \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
        actor_rollout_ref.rollout.n=${ROLLOUT_N} \
        actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${ROLLOUT_UPDATE_BUCKET_MB} \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.experiment_name=${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S) \
        trainer.logger=['console','wandb'] \
        trainer.project_name=${WANDB_PROJECT} \
        trainer.default_local_dir=${OUTPUT_DIR} \
        trainer.total_epochs=${EPOCHS} \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        trainer.use_legacy_worker_impl=enable \
        trainer.val_before_train=False \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.test_freq=${TEST_FREQ}"

    # Add adapter config
    case "$ADAPTER" in
        lora|dora|loraplus|rslora|qalora)
            CMD="${CMD} \
                actor_rollout_ref.model.lora_rank=${LORA_RANK} \
                actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
                actor_rollout_ref.model.lora_dropout=${LORA_DROPOUT}"
            if [ -n "$TARGET_MODULES" ]; then
                CMD="${CMD} actor_rollout_ref.model.target_modules=${TARGET_MODULES}"
            fi
            if [ "$ADAPTER" = "dora" ]; then
                CMD="${CMD} actor_rollout_ref.model.use_dora=true"
            elif [ "$ADAPTER" = "loraplus" ]; then
                CMD="${CMD} actor_rollout_ref.model.lora_variant=loraplus"
            elif [ "$ADAPTER" = "rslora" ]; then
                CMD="${CMD} actor_rollout_ref.model.use_rslora=true"
            elif [ "$ADAPTER" = "qalora" ]; then
                CMD="${CMD} actor_rollout_ref.model.use_qalora=true"
            fi
            ;;
        adalora)
            CMD="${CMD} \
                actor_rollout_ref.model.lora_rank=${LORA_RANK} \
                actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
                actor_rollout_ref.model.lora_dropout=${LORA_DROPOUT} \
                actor_rollout_ref.model.lora_variant=adalora"
            if [ -n "$TARGET_MODULES" ]; then
                CMD="${CMD} actor_rollout_ref.model.target_modules=${TARGET_MODULES}"
            fi
            ;;
        pissa|milora)
            CMD="${CMD} \
                actor_rollout_ref.model.lora_rank=${LORA_RANK} \
                actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
                actor_rollout_ref.model.lora_dropout=${LORA_DROPOUT}"
            if [ -n "$TARGET_MODULES" ]; then
                CMD="${CMD} actor_rollout_ref.model.target_modules=${TARGET_MODULES}"
            fi
            if [ -n "$PEFT_ADAPTER_PATH" ]; then
                CMD="${CMD} actor_rollout_ref.model.peft_adapter_path=${PEFT_ADAPTER_PATH}"
            fi
            ;;
        oft)
            CMD="${CMD} \
                actor_rollout_ref.model.oft_block_size=${OFT_BLOCK_SIZE} \
                actor_rollout_ref.actor.fsdp_config.use_orig_params=True"
            if [ -n "$TARGET_MODULES" ]; then
                CMD="${CMD} actor_rollout_ref.model.target_modules=${TARGET_MODULES}"
            fi
            ;;
        ia3)
            CMD="${CMD} \
                actor_rollout_ref.model.lora_rank=${LORA_RANK} \
                actor_rollout_ref.model.lora_variant=ia3"
            ;;
        vera)
            CMD="${CMD} \
                actor_rollout_ref.model.lora_rank=${VERA_RANK} \
                actor_rollout_ref.model.lora_dropout=${LORA_DROPOUT} \
                actor_rollout_ref.model.lora_variant=vera"
            ;;
        full)
            ;;
        *)
            echo "Note: Adapter '${ADAPTER}' may need manual config for RL training."
            ;;
    esac

    export PYTHONPATH="${ARENA_TRAIN_PY}:${PYTHONPATH}"

    echo "============================================"
    echo "[PEFTArena] RL (GRPO) Training"
    echo "  Model:   ${MODEL}"
    echo "  Adapter: ${ADAPTER}"
    echo "  Train:   ${DATA_TRAIN}"
    echo "  Val:     ${DATA_VAL}"
    echo "  Output:  ${OUTPUT_DIR}"
    echo "  GPUs:    ${NUM_GPUS}"
    echo "  Prompt:  ${PROMPT_MAX_LENGTH}"
    echo "  Resp:    ${MAX_LENGTH}"
    echo "  LR:      ${LR}"
    if [ -n "$TARGET_MODULES" ]; then
        echo "  Targets: ${TARGET_MODULES}"
    fi
    echo "============================================"

    eval $CMD
else
    echo "============================================"
    echo "[PEFTArena] RL (GRPO) Training via VeRL script"
    echo "  Model:   ${MODEL}"
    echo "  Adapter: ${ADAPTER}"
    echo "  Output:  ${OUTPUT_DIR}"
    echo "  GPUs:    ${NUM_GPUS}"
    echo "============================================"

    export PYTHONPATH="${ARENA_TRAIN_PY}:${PYTHONPATH}"
    bash "${GRPO_SCRIPT}"
fi
