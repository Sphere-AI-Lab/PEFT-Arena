#!/usr/bin/env python3
"""
Compare parameter norm differences between a base model and an SFT model.

This script reports global and per-parameter L1/L2 distances:
  - delta_l1 = ||W_sft - W_base||_1
  - delta_l2 = ||W_sft - W_base||_2

It also reports relative distances against the base parameter norm:
  - relative_l1 = ||ΔW||_1 / (||W_base||_1 + eps)
  - relative_l2 = ||ΔW||_2 / (||W_base||_2 + eps)

Both full-model checkpoints and PEFT adapter checkpoints are supported. If the
SFT path points to an adapter, the adapter is merged into the base model before
comparison.

Example:
  python tools/compare_model_norms.py \
    --base-model Qwen/Qwen2.5-7B \
    --sft-model checkpoints/sft/math/qwen2.5-7b/lora-r8/global_step_780 \
    --output analysis/model_norms/qwen2.5-7b_lora-r8
"""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


EPS = 1e-12


def is_adapter_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    return (
        "adapter_config.json" in files
        or "adapter_model.safetensors" in files
        or "adapter_model.bin" in files
    )


def load_models(base_model_path: str, sft_model_path: str):
    from transformers import AutoModelForCausalLM

    model_base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model_base.eval()

    peft_type = None
    if is_adapter_checkpoint(sft_model_path):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(sft_model_path)
        peft_type = str(peft_config.peft_type)

        model_sft_base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(model_sft_base, sft_model_path)
        model_sft = peft_model.merge_and_unload()
        del peft_model
    else:
        model_sft = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    model_sft.eval()
    return model_base, model_sft, peft_type


def named_float_parameters(model) -> Dict[str, torch.nn.Parameter]:
    return {
        name: param
        for name, param in model.named_parameters()
        if torch.is_tensor(param) and param.is_floating_point()
    }


def keep_parameter(
    name: str,
    parameter: torch.nn.Parameter,
    include_regex: Optional[str],
    exclude_regex: Optional[str],
    min_ndim: int,
) -> bool:
    if parameter.ndim < min_ndim:
        return False
    if include_regex and re.search(include_regex, name) is None:
        return False
    if exclude_regex and re.search(exclude_regex, name):
        return False
    return True


def tensor_l1(tensor: torch.Tensor) -> float:
    return float(tensor.abs().sum().item())


def tensor_l2(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.reshape(-1), ord=2).item())


def compare_parameters(
    base_parameters: Dict[str, torch.nn.Parameter],
    sft_parameters: Dict[str, torch.nn.Parameter],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
    min_ndim: int,
) -> Dict[str, object]:
    shared_names = sorted(set(base_parameters) & set(sft_parameters))
    base_only = sorted(set(base_parameters) - set(sft_parameters))
    sft_only = sorted(set(sft_parameters) - set(base_parameters))

    rows: List[Dict[str, object]] = []
    total_numel = 0
    total_base_l1 = 0.0
    total_sft_l1 = 0.0
    total_delta_l1 = 0.0
    total_base_l2_sq = 0.0
    total_sft_l2_sq = 0.0
    total_delta_l2_sq = 0.0

    skipped_by_filter = 0
    skipped_by_shape = 0

    for name in shared_names:
        base_param = base_parameters[name]
        sft_param = sft_parameters[name]

        if base_param.shape != sft_param.shape:
            skipped_by_shape += 1
            continue

        if not keep_parameter(
            name=name,
            parameter=base_param,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
            min_ndim=min_ndim,
        ):
            skipped_by_filter += 1
            continue

        base_tensor = base_param.detach().float().cpu()
        sft_tensor = sft_param.detach().float().cpu()
        delta_tensor = sft_tensor - base_tensor

        base_l1 = tensor_l1(base_tensor)
        sft_l1 = tensor_l1(sft_tensor)
        delta_l1 = tensor_l1(delta_tensor)
        base_l2 = tensor_l2(base_tensor)
        sft_l2 = tensor_l2(sft_tensor)
        delta_l2 = tensor_l2(delta_tensor)
        numel = int(base_tensor.numel())

        total_numel += numel
        total_base_l1 += base_l1
        total_sft_l1 += sft_l1
        total_delta_l1 += delta_l1
        total_base_l2_sq += base_l2 * base_l2
        total_sft_l2_sq += sft_l2 * sft_l2
        total_delta_l2_sq += delta_l2 * delta_l2

        rows.append(
            {
                "name": name,
                "shape": list(base_tensor.shape),
                "numel": numel,
                "base_l1": base_l1,
                "base_l2": base_l2,
                "sft_l1": sft_l1,
                "sft_l2": sft_l2,
                "delta_l1": delta_l1,
                "delta_l2": delta_l2,
                "relative_l1": delta_l1 / (base_l1 + EPS),
                "relative_l2": delta_l2 / (base_l2 + EPS),
                "mean_abs_delta": delta_l1 / max(numel, 1),
                "rmse_delta": delta_l2 / math.sqrt(max(numel, 1)),
            }
        )

    rows.sort(key=lambda item: float(item["delta_l2"]), reverse=True)

    summary = {
        "matched_parameter_count": len(shared_names),
        "compared_parameter_count": len(rows),
        "base_only_parameter_count": len(base_only),
        "sft_only_parameter_count": len(sft_only),
        "skipped_by_filter": skipped_by_filter,
        "skipped_by_shape": skipped_by_shape,
        "total_numel": total_numel,
        "global_base_l1": total_base_l1,
        "global_sft_l1": total_sft_l1,
        "global_delta_l1": total_delta_l1,
        "global_base_l2": math.sqrt(total_base_l2_sq),
        "global_sft_l2": math.sqrt(total_sft_l2_sq),
        "global_delta_l2": math.sqrt(total_delta_l2_sq),
        "global_relative_l1": total_delta_l1 / (total_base_l1 + EPS),
        "global_relative_l2": math.sqrt(total_delta_l2_sq) / (math.sqrt(total_base_l2_sq) + EPS),
        "global_mean_abs_delta": total_delta_l1 / max(total_numel, 1),
        "global_rmse_delta": math.sqrt(total_delta_l2_sq) / math.sqrt(max(total_numel, 1)),
        "base_only_parameter_examples": base_only[:20],
        "sft_only_parameter_examples": sft_only[:20],
    }
    return {"summary": summary, "rows": rows}


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "name",
                    "shape",
                    "numel",
                    "base_l1",
                    "base_l2",
                    "sft_l1",
                    "sft_l2",
                    "delta_l1",
                    "delta_l2",
                    "relative_l1",
                    "relative_l2",
                    "mean_abs_delta",
                    "rmse_delta",
                ]
            )
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure L1/L2 norm differences between a base model and an SFT model."
    )
    parser.add_argument("--base-model", required=True, help="base model directory")
    parser.add_argument("--sft-model", required=True, help="SFT full-model directory or PEFT adapter directory")
    parser.add_argument(
        "--output",
        default=None,
        help="output prefix or directory; writes <prefix>.json and <prefix>.csv",
    )
    parser.add_argument(
        "--include-regex",
        default=None,
        help="only compare parameter names matching this regex",
    )
    parser.add_argument(
        "--exclude-regex",
        default=None,
        help="skip parameter names matching this regex",
    )
    parser.add_argument(
        "--min-ndim",
        type=int,
        default=1,
        help="minimum tensor ndim to include",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="number of largest-delta parameters to print",
    )
    return parser.parse_args()


def resolve_output_paths(output_arg: Optional[str]) -> Optional[tuple[Path, Path]]:
    if output_arg is None:
        return None
    output_path = Path(output_arg)
    if output_path.suffix:
        output_prefix = output_path.with_suffix("")
    elif output_path.exists() and output_path.is_dir():
        output_prefix = output_path / "model_norm_comparison"
    elif output_path.name == "":
        output_prefix = output_path / "model_norm_comparison"
    else:
        output_prefix = output_path
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    return output_prefix.with_suffix(".json"), output_prefix.with_suffix(".csv")


def main() -> None:
    args = parse_args()

    print(f"[norms] Loading base model: {args.base_model}")
    print(f"[norms] Loading SFT model:  {args.sft_model}")
    model_base, model_sft, peft_type = load_models(args.base_model, args.sft_model)
    if peft_type:
        print(f"[norms] Detected PEFT adapter: {peft_type}")

    base_parameters = named_float_parameters(model_base)
    sft_parameters = named_float_parameters(model_sft)
    result = compare_parameters(
        base_parameters=base_parameters,
        sft_parameters=sft_parameters,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        min_ndim=args.min_ndim,
    )

    summary = result["summary"]
    rows = result["rows"]

    print("")
    print("[norms] Global summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("")
    print(f"[norms] Top {min(args.top_k, len(rows))} parameters by delta_l2")
    for row in rows[: args.top_k]:
        print(
            f"  - {row['name']}: "
            f"delta_l2={row['delta_l2']:.6f}, "
            f"delta_l1={row['delta_l1']:.6f}, "
            f"relative_l2={row['relative_l2']:.6f}, "
            f"shape={tuple(row['shape'])}"
        )

    output_paths = resolve_output_paths(args.output)
    if output_paths is not None:
        json_path, csv_path = output_paths
        payload = {
            "base_model": args.base_model,
            "sft_model": args.sft_model,
            "peft_type": peft_type,
            "include_regex": args.include_regex,
            "exclude_regex": args.exclude_regex,
            "min_ndim": args.min_ndim,
            "summary": summary,
            "per_parameter": rows,
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(csv_path, rows)
        print("")
        print(f"[norms] Wrote JSON: {json_path}")
        print(f"[norms] Wrote CSV:  {csv_path}")


if __name__ == "__main__":
    main()
