#!/usr/bin/env python3
"""Capability-conditioned drift for one checkpoint and one module.

This is a lightweight release script.  It does not depend on PEFT-Arena's
internal result tables or checkpoint manifests.  Provide the base model,
one finetuned checkpoint or PEFT adapter, and prompt files for the capability
distributions you want to probe.
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
from typing import Dict, List

import torch

from diagnostic_utils import (
    collect_module_tensor,
    get_linear_weight,
    load_effective_model,
    load_prompts,
    load_tokenizer,
    parse_dtype,
    resolve_device,
    seed_everything,
    write_csv,
    write_json,
)


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True, help="Path or HF id of the pretrained base model.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a full finetuned model or PEFT adapter checkpoint.",
    )
    parser.add_argument(
        "--module",
        default="model.layers.18.mlp.down_proj",
        help="Linear module path whose input activations and effective weight update are analyzed.",
    )
    parser.add_argument("--general_prompts", help="txt/json/jsonl prompts for the general capability distribution.")
    parser.add_argument("--target_prompts", help="txt/json/jsonl prompts for the target-domain distribution.")
    parser.add_argument("--text_key", help="Optional JSON/JSONL key containing prompt text.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_examples", type=int, default=300)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--token_chunk_size", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--compute_device", default="auto", help="Device for matrix products; defaults to --device behavior.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--train_setting", default="")
    parser.add_argument("--target_domain", default="")
    parser.add_argument("--method", default="")
    return parser.parse_args()


def collect_base_activations(args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    tokenizer = load_tokenizer(args.base_model)
    base_model = load_effective_model(
        base_model=args.base_model,
        checkpoint=args.base_model,
        dtype=dtype,
        device=device,
    )
    w0 = get_linear_weight(base_model, args.module)

    datasets: Dict[str, List[str]] = {}
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

    activations = {}
    for dataset_name, prompts in datasets.items():
        activations[dataset_name] = {
            "prompts": prompts,
            "h": collect_module_tensor(
                base_model,
                tokenizer,
                prompts,
                args.module,
                capture="input",
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_tokens=args.max_tokens,
            ),
        }

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return w0, activations


def compute_csd_for_h(
    h_cpu: torch.Tensor,
    w0_cpu: torch.Tensor,
    delta_cpu: torch.Tensor,
    *,
    compute_device: torch.device,
    chunk_size: int,
) -> dict:
    w0 = w0_cpu.to(compute_device, dtype=torch.float32)
    delta = delta_cpu.to(compute_device, dtype=torch.float32)
    delta_frob2 = float((delta.square().sum()).item())

    dy_sum = 0.0
    y0_sum = 0.0
    h_sum = 0.0
    n = int(h_cpu.shape[0])
    for start in range(0, n, chunk_size):
        h = h_cpu[start : start + chunk_size].to(compute_device, dtype=torch.float32)
        y0 = h @ w0.T
        dy = h @ delta.T
        dy_sum += float(dy.square().sum(dim=1).sum().item())
        y0_sum += float(y0.square().sum(dim=1).sum().item())
        h_sum += float(h.square().sum(dim=1).sum().item())
        del h, y0, dy

    h_norm2_mean = h_sum / max(n, 1)
    w0h_norm2_mean = y0_sum / max(n, 1)
    dy_norm2_mean = dy_sum / max(n, 1)
    return {
        "n_tokens": n,
        "delta_frob2": delta_frob2,
        "h_norm2_mean": h_norm2_mean,
        "w0h_norm2_mean": w0h_norm2_mean,
        "CSD_abs": dy_norm2_mean,
        "CSD_rel": dy_norm2_mean / (w0h_norm2_mean + EPS),
        "CSD_update_norm": dy_norm2_mean / ((delta_frob2 + EPS) * (h_norm2_mean + EPS)),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = parse_dtype(args.dtype)
    device = resolve_device(args.device)
    compute_device = resolve_device(args.compute_device)

    w0, activations = collect_base_activations(args, dtype=dtype, device=device)

    ft_model = load_effective_model(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        dtype=dtype,
        device=device,
    )
    wft = get_linear_weight(ft_model, args.module)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if tuple(w0.shape) != tuple(wft.shape):
        raise ValueError(f"Weight shape mismatch: base={tuple(w0.shape)} checkpoint={tuple(wft.shape)}")
    delta = wft - w0

    dataset_rows = []
    metrics_by_dataset = {}
    for dataset_name, payload in activations.items():
        metrics = compute_csd_for_h(
            payload["h"],
            w0,
            delta,
            compute_device=compute_device,
            chunk_size=args.token_chunk_size,
        )
        metrics_by_dataset[dataset_name] = metrics
        dataset_rows.append(
            {
                "dataset_type": dataset_name,
                "n_examples": len(payload["prompts"]),
                "module": args.module,
                **metrics,
            }
        )

    general = metrics_by_dataset.get("general")
    target = metrics_by_dataset.get("target")
    summary = {
        "model": args.model_name,
        "train_setting": args.train_setting,
        "target_domain": args.target_domain,
        "method": args.method,
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "module": args.module,
        "weight_shape": "x".join(str(dim) for dim in w0.shape),
        "n_general_examples": len(activations["general"]["prompts"]) if "general" in activations else "",
        "n_target_examples": len(activations["target"]["prompts"]) if "target" in activations else "",
        "CSD_G_abs": general["CSD_abs"] if general else math.nan,
        "CSD_G_rel": general["CSD_rel"] if general else math.nan,
        "CSD_G_update_norm": general["CSD_update_norm"] if general else math.nan,
        "CSD_T_abs": target["CSD_abs"] if target else math.nan,
        "CSD_T_rel": target["CSD_rel"] if target else math.nan,
        "CSD_T_update_norm": target["CSD_update_norm"] if target else math.nan,
        "ratio_rel": (general["CSD_rel"] / (target["CSD_rel"] + EPS)) if general and target else math.nan,
        "ratio_update_norm": (
            general["CSD_update_norm"] / (target["CSD_update_norm"] + EPS)
        )
        if general and target
        else math.nan,
    }

    write_csv(
        output_dir / "csd_dataset_metrics.csv",
        dataset_rows,
        [
            "dataset_type",
            "n_examples",
            "n_tokens",
            "module",
            "delta_frob2",
            "h_norm2_mean",
            "w0h_norm2_mean",
            "CSD_abs",
            "CSD_rel",
            "CSD_update_norm",
        ],
    )
    write_csv(output_dir / "csd_summary.csv", [summary], list(summary.keys()))
    write_json(output_dir / "csd_summary.json", {"summary": summary, "datasets": dataset_rows})
    print(f"[csd] wrote {output_dir / 'csd_summary.csv'}")


if __name__ == "__main__":
    main()
