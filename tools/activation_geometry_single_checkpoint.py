#!/usr/bin/env python3
"""Activation-space geometry diagnostics for one checkpoint.

The script compares full-forward module outputs from a base model and a single
finetuned checkpoint or PEFT adapter.  It reports pointwise movement and
non-isometric geometry distortion metrics for user-provided prompt sets.
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
from typing import List

import torch

from diagnostic_utils import (
    collect_module_tensor,
    gram_cosine_dist,
    linear_cka,
    load_effective_model,
    load_prompts,
    load_tokenizer,
    parse_dtype,
    procrustes_metrics,
    resolve_device,
    seed_everything,
    subspace_distance,
    write_csv,
    write_json,
)


PAPER8_MODULES = [
    "model.layers.9.self_attn.q_proj",
    "model.layers.9.self_attn.k_proj",
    "model.layers.9.self_attn.v_proj",
    "model.layers.9.mlp.down_proj",
    "model.layers.18.self_attn.q_proj",
    "model.layers.18.self_attn.k_proj",
    "model.layers.18.self_attn.v_proj",
    "model.layers.18.mlp.down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True, help="Path or HF id of the pretrained base model.")
    parser.add_argument("--checkpoint", required=True, help="Full finetuned model or PEFT adapter checkpoint.")
    parser.add_argument(
        "--modules",
        default="model.layers.18.mlp.down_proj",
        help="Comma-separated module paths, or 'paper8' for layer 9/18 q,k,v,down modules.",
    )
    parser.add_argument("--general_prompts", help="txt/json/jsonl prompts for the general capability distribution.")
    parser.add_argument("--target_prompts", help="txt/json/jsonl prompts for the target-domain distribution.")
    parser.add_argument("--text_key", help="Optional JSON/JSONL key containing prompt text.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_examples", type=int, default=300)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_pair_tokens", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--train_setting", default="")
    parser.add_argument("--target_domain", default="")
    parser.add_argument("--method", default="")
    return parser.parse_args()


def parse_modules(value: str) -> List[str]:
    if value.strip().lower() == "paper8":
        return PAPER8_MODULES
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--modules resolved to an empty list")
    return modules


def load_datasets(args: argparse.Namespace):
    datasets = {}
    if args.general_prompts:
        datasets["general"] = load_prompts(
            args.general_prompts,
            text_key=args.text_key,
            max_examples=args.max_examples,
            seed=args.seed,
        )
    if args.target_prompts:
        datasets["target"] = load_prompts(
            args.target_prompts,
            text_key=args.text_key,
            max_examples=args.max_examples,
            seed=args.seed,
        )
    if not datasets:
        raise ValueError("Provide at least one of --general_prompts or --target_prompts.")
    return datasets


def collect_outputs_for_model(model, tokenizer, datasets, modules, args):
    outputs = {}
    for dataset_type, prompts in datasets.items():
        outputs[dataset_type] = {}
        for module in modules:
            outputs[dataset_type][module] = collect_module_tensor(
                model,
                tokenizer,
                prompts,
                module,
                capture="output",
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_tokens=args.max_tokens,
            )
    return outputs


def compute_metrics(h0: torch.Tensor, h1: torch.Tensor, *, max_pair_tokens: int, seed: int) -> dict:
    n = min(h0.shape[0], h1.shape[0])
    h0 = h0[:n].float()
    h1 = h1[:n].float()

    z0 = torch.nn.functional.normalize(h0, dim=1)
    z1 = torch.nn.functional.normalize(h1, dim=1)
    cosine = (z0 * z1).sum(dim=1).clamp(-1.0, 1.0)
    angular = torch.arccos(cosine)

    norm0 = h0.norm(dim=1)
    norm1 = h1.norm(dim=1)
    norm_ratio = norm1 / (norm0 + 1e-12)

    raw_residual, proc_residual, proc_improvement = procrustes_metrics(h0, h1)
    return {
        "n_tokens": int(n),
        "mean_cosine": float(cosine.mean().item()),
        "mean_cosine_drift": float((1.0 - cosine).mean().item()),
        "mean_angular_drift": float(angular.mean().item()),
        "median_angular_drift": float(angular.median().item()),
        "mean_abs_log_norm_ratio": float(torch.log(norm_ratio + 1e-12).abs().mean().item()),
        "mean_norm_ratio": float(norm_ratio.mean().item()),
        "std_norm_ratio": float(norm_ratio.std(unbiased=False).item()),
        "gram_cosine_dist": gram_cosine_dist(h0, h1, max_pair_tokens=max_pair_tokens, seed=seed),
        "raw_residual": raw_residual,
        "proc_residual": proc_residual,
        "proc_improvement": proc_improvement,
        "linear_cka": linear_cka(h0, h1),
        "subspace_dist_k32": subspace_distance(h0, h1, 32),
        "subspace_dist_k64": subspace_distance(h0, h1, 64),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = parse_modules(args.modules)
    dtype = parse_dtype(args.dtype)
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.base_model)
    datasets = load_datasets(args)

    base_model = load_effective_model(
        base_model=args.base_model,
        checkpoint=args.base_model,
        dtype=dtype,
        device=device,
    )
    base_outputs = collect_outputs_for_model(base_model, tokenizer, datasets, modules, args)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    ft_model = load_effective_model(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        dtype=dtype,
        device=device,
    )

    rows = []
    for dataset_type, prompts in datasets.items():
        for module in modules:
            ft_output = collect_module_tensor(
                ft_model,
                tokenizer,
                prompts,
                module,
                capture="output",
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_tokens=args.max_tokens,
            )
            metrics = compute_metrics(
                base_outputs[dataset_type][module],
                ft_output,
                max_pair_tokens=args.max_pair_tokens,
                seed=args.seed,
            )
            rows.append(
                {
                    "model": args.model_name,
                    "train_setting": args.train_setting,
                    "target_domain": args.target_domain,
                    "method": args.method,
                    "base_model": args.base_model,
                    "checkpoint": args.checkpoint,
                    "dataset_type": dataset_type,
                    "n_examples": len(prompts),
                    "module": module,
                    "activation_mode": "full_forward",
                    **metrics,
                }
            )
            del ft_output

    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fieldnames = [
        "model",
        "train_setting",
        "target_domain",
        "method",
        "base_model",
        "checkpoint",
        "dataset_type",
        "n_examples",
        "n_tokens",
        "module",
        "activation_mode",
        "mean_cosine",
        "mean_cosine_drift",
        "mean_angular_drift",
        "median_angular_drift",
        "mean_abs_log_norm_ratio",
        "mean_norm_ratio",
        "std_norm_ratio",
        "gram_cosine_dist",
        "raw_residual",
        "proc_residual",
        "proc_improvement",
        "linear_cka",
        "subspace_dist_k32",
        "subspace_dist_k64",
    ]
    write_csv(output_dir / "activation_geometry_summary.csv", rows, fieldnames)
    write_json(output_dir / "activation_geometry_summary.json", {"rows": rows})
    print(f"[activation-geometry] wrote {output_dir / 'activation_geometry_summary.csv'}")


if __name__ == "__main__":
    main()
