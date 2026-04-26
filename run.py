#!/usr/bin/env python3
"""PEFTArena: Unified CLI for PEFT benchmark training and evaluation.

Usage:
    python run.py train sft --model <model> --adapter <adapter> ...
    python run.py train rl  --model <model> --adapter <adapter> ...
    python run.py eval      --checkpoint_path <path> --domain <domain>
    python run.py merge     --adapter_path <path> --output_path <path>
"""

import argparse
import os
import sys
import subprocess
import json
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_FULL_ADAPTERS = {"full"}


def resolve_train_lr(train_type: str, adapter: str, lr: Optional[float]) -> float:
    """Resolve training LR with RL-specific adapter defaults."""
    if lr is not None:
        return lr
    if train_type == "rl":
        return 1e-6 if adapter in RL_FULL_ADAPTERS else 1e-5
    return 2e-4


def resolve_train_epochs(train_type: str, epochs: Optional[int]) -> int:
    """Resolve training epoch count with RL-specific defaults."""
    if epochs is not None:
        return epochs
    if train_type == "rl":
        return 10
    return 4


def is_adapter_checkpoint(path: str) -> bool:
    """Check if a path contains a PEFT adapter checkpoint."""
    if not os.path.isdir(path):
        return False
    return (
        os.path.exists(os.path.join(path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(path, "adapter_model.bin"))
    )


def get_adapter_type(path: str) -> str:
    """Detect the PEFT adapter type from checkpoint config."""
    config_path = os.path.join(path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        return config.get("peft_type", "unknown").lower()
    return "unknown"


def derive_eval_model_name(checkpoint_path: str) -> str:
    """Derive a stable per-checkpoint result directory name."""
    normalized = os.path.normpath(os.path.abspath(checkpoint_path))
    rel_path = normalized
    markers = [
        os.sep + "release_ckpts" + os.sep,
        os.sep + "checkpoints" + os.sep,
        "release_ckpts" + os.sep,
        "checkpoints" + os.sep,
    ]
    for marker in markers:
        if marker in rel_path:
            rel_path = rel_path.split(marker, 1)[1]
            break
    else:
        if rel_path.startswith(SCRIPT_DIR + os.sep):
            rel_path = rel_path[len(SCRIPT_DIR) + 1 :]
    return rel_path.replace(os.sep, "_").replace("_global_step_", "_")


def resolve_eval_output_dir(checkpoint_path: str, output_dir: Optional[str], domain: str) -> str:
    """Resolve eval output dir to results/<checkpoint_name>/<domain> by default."""
    model_name = derive_eval_model_name(checkpoint_path)
    if output_dir is None:
        return os.path.join(SCRIPT_DIR, "results", model_name, domain)

    normalized = os.path.normpath(output_dir)
    base_name = os.path.basename(normalized)
    parent_name = os.path.basename(os.path.dirname(normalized))

    if base_name == domain:
        if parent_name == model_name:
            return normalized
        return os.path.join(os.path.dirname(normalized), model_name, domain)
    if base_name == model_name:
        return os.path.join(normalized, domain)
    if base_name == "results":
        return os.path.join(normalized, model_name, domain)
    return normalized


# ---------------------------------------------------------------------------
# Train subcommand
# ---------------------------------------------------------------------------
def cmd_train(args):
    """Dispatch training (SFT or RL)."""
    if args.train_type == "sft":
        script = os.path.join(SCRIPT_DIR, "train", "train_sft.sh")
    elif args.train_type == "rl":
        script = os.path.join(SCRIPT_DIR, "train", "train_rl.sh")
    else:
        print(f"Unknown training type: {args.train_type}")
        sys.exit(1)

    cmd = ["bash", script]
    lr = resolve_train_lr(args.train_type, args.adapter, args.lr)
    epochs = resolve_train_epochs(args.train_type, args.epochs)

    # Common args
    cmd += ["--model", args.model]
    cmd += ["--adapter", args.adapter]
    cmd += ["--output_dir", args.output_dir]
    cmd += ["--lr", str(lr)]
    cmd += ["--epochs", str(epochs)]
    cmd += ["--num_gpus", str(args.num_gpus)]
    cmd += ["--batch_size", str(args.batch_size)]
    cmd += ["--max_length", str(args.max_length)]
    cmd += ["--micro_batch_size", str(args.micro_batch_size)]
    if args.train_type == "rl":
        cmd += ["--prompt_max_length", str(args.prompt_max_length)]

    if args.data_train:
        cmd += ["--data_train", args.data_train]
    if args.data_val:
        cmd += ["--data_val", args.data_val]

    # Adapter-specific args
    if args.adapter in ("lora", "dora", "adalora", "pissa", "milora", "keeplora", "miss"):
        cmd += ["--lora_rank", str(args.lora_rank)]
        cmd += ["--lora_alpha", str(args.lora_alpha)]
        cmd += ["--lora_dropout", str(args.lora_dropout)]
    elif args.adapter == "oft":
        cmd += ["--oft_block_size", str(args.oft_block_size)]
        cmd += ["--oft_normalize_rotation", args.oft_normalize_rotation]
    elif args.adapter == "vera":
        cmd += ["--vera_rank", str(args.vera_rank)]

    if args.save_freq:
        cmd += ["--save_freq", str(args.save_freq)]
    if args.test_freq:
        cmd += ["--test_freq", str(args.test_freq)]
    if args.experiment_name:
        cmd += ["--experiment_name", args.experiment_name]
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project]
    if args.peft_adapter_path:
        cmd += ["--peft_adapter_path", args.peft_adapter_path]

    print(f"[PEFTArena] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Eval subcommand
# ---------------------------------------------------------------------------
def cmd_eval(args):
    """Dispatch evaluation for a given domain."""
    domains = [args.domain] if args.domain != "all" else ["math", "med", "general"]

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"[PEFTArena] Evaluating domain: {domain}")
        print(f"{'='*60}")

        if domain == "math":
            script = os.path.join(SCRIPT_DIR, "eval", "eval_math.sh")
            cmd = [
                "bash", script,
                "--checkpoint_path", args.checkpoint_path,
                "--output_dir", resolve_eval_output_dir(args.checkpoint_path, args.output_dir, "math"),
                "--num_gpus", str(args.num_gpus),
            ]
            if args.data_names:
                cmd += ["--data_names", args.data_names]
            if args.temperature is not None:
                cmd += ["--temperature", str(args.temperature)]
            if args.n_sampling is not None:
                cmd += ["--n_sampling", str(args.n_sampling)]
            if args.max_tokens_per_call is not None:
                cmd += ["--max_tokens_per_call", str(args.max_tokens_per_call)]
        elif domain == "med":
            script = os.path.join(SCRIPT_DIR, "eval", "eval_med.sh")
            cmd = [
                "bash", script,
                "--checkpoint_path", args.checkpoint_path,
                "--output_dir", resolve_eval_output_dir(args.checkpoint_path, args.output_dir, "med"),
                "--num_gpus", str(args.num_gpus),
            ]
            if args.tasks:
                cmd += ["--tasks", args.tasks]
        elif domain == "general":
            script = os.path.join(SCRIPT_DIR, "eval", "eval_general.sh")
            cmd = [
                "bash", script,
                "--checkpoint_path", args.checkpoint_path,
                "--output_dir", resolve_eval_output_dir(args.checkpoint_path, args.output_dir, "general"),
                "--num_gpus", str(args.num_gpus),
            ]
            if args.benchmarks:
                cmd += ["--benchmarks", args.benchmarks]
            else:
                cmd += ["--benchmarks", "bbh,ifeval_nq"]
        else:
            print(f"Unknown domain: {domain}")
            continue

        print(f"[PEFTArena] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[PEFTArena] WARNING: {domain} evaluation returned exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Merge subcommand
# ---------------------------------------------------------------------------
def cmd_merge(args):
    """Merge a PEFT adapter into the base model."""
    from tools.merge_peft import merge_adapter
    merge_adapter(
        adapter_path=args.adapter_path,
        output_dir=args.output_path,
        torch_dtype=args.torch_dtype,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PEFTArena: Unified benchmark for PEFT methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_subparsers = train_parser.add_subparsers(dest="train_type", help="Training type")

    for train_type in ["sft", "rl"]:
        tp = train_subparsers.add_parser(train_type, help=f"{train_type.upper()} training")
        tp.add_argument("--model", required=True, help="Base model name or path")
        tp.add_argument("--adapter", default="lora",
                        choices=["lora", "oft", "full", "dora", "ia3", "vera", "adalora", "pissa", "milora", "loraplus", "rslora", "qalora", "keeplora", "miss"],
                        help="PEFT adapter type")
        tp.add_argument("--output_dir", required=True, help="Checkpoint output directory")
        tp.add_argument("--data_train", default=None, help="Training data parquet file")
        tp.add_argument("--data_val", default=None, help="Validation data parquet file")
        tp.add_argument("--lr", type=float, default=None,
                        help="Learning rate (RL defaults: full=1e-6, PEFT=1e-5; SFT default: 2e-4)")
        tp.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (RL default: 10; SFT default: 4)")
        tp.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs")
        tp.add_argument("--batch_size", type=int, default=256, help="Global batch size")
        tp.add_argument("--max_length", type=int, default=8192, help="Max response length")
        tp.add_argument("--prompt_max_length", type=int, default=1024, help="Max prompt length for RL")
        tp.add_argument("--micro_batch_size", type=int, default=2, help="Micro batch size per GPU")

        # LoRA-family args
        tp.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
        tp.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
        tp.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

        # OFT args
        tp.add_argument("--oft_block_size", type=int, default=16, help="OFT block size")
        tp.add_argument("--oft_normalize_rotation", default="none", choices=["none", "learnable", "mean_norm", "upper_clamp"], help="OFT rotation normalization mode")

        # VeRA args
        tp.add_argument("--vera_rank", type=int, default=256, help="VeRA rank")

        # Training control
        tp.add_argument("--save_freq", type=int, default=None, help="Save frequency (steps)")
        tp.add_argument("--test_freq", type=int, default=None, help="Test frequency (steps)")
        tp.add_argument("--experiment_name", default=None, help="Experiment name for logging")
        tp.add_argument("--wandb_project", default=None, help="W&B project name")
        tp.add_argument("--peft_adapter_path", default=None, help="Pre-initialized PEFT adapter path (for MiLoRA, PiSSA, etc.)")

        tp.set_defaults(func=cmd_train)

    # ---- eval ----
    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    eval_parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint (full or PEFT adapter)")
    eval_parser.add_argument("--domain", default="all", choices=["math", "med", "general", "all"], help="Evaluation domain")
    eval_parser.add_argument("--output_dir", default=None, help="Results output directory")
    eval_parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    # Math-specific
    eval_parser.add_argument("--data_names", default=None, help="Comma-separated math dataset names")
    eval_parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (math default: 0.6)")
    eval_parser.add_argument("--n_sampling", type=int, default=None, help="Number of samples per problem (math default: 16)")
    eval_parser.add_argument("--max_tokens_per_call", type=int, default=None, help="Max tokens per generation call (math default: 8192)")
    # Med-specific
    eval_parser.add_argument("--tasks", default=None, help="Comma-separated medical tasks")
    # General-specific
    eval_parser.add_argument("--benchmarks", default=None, help="Comma-separated general benchmarks")
    eval_parser.set_defaults(func=cmd_eval)

    # ---- merge ----
    merge_parser = subparsers.add_parser("merge", help="Merge PEFT adapter into base model")
    merge_parser.add_argument("--adapter_path", required=True, help="Path to PEFT adapter checkpoint")
    merge_parser.add_argument("--output_path", default=None, help="Output merged model path")
    merge_parser.add_argument("--torch_dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    merge_parser.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
