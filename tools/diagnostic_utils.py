#!/usr/bin/env python3
"""Shared utilities for single-checkpoint geometry diagnostics."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


TEXT_KEYS = (
    "prompt",
    "input",
    "question",
    "query",
    "problem",
    "instruction",
    "text",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def parse_dtype(dtype: str) -> torch.dtype:
    normalized = dtype.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def is_adapter_checkpoint(path: str | Path) -> bool:
    path = Path(path)
    return (
        path.is_dir()
        and ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists())
    )


def load_prompts(
    path: str | Path,
    *,
    text_key: str | None = None,
    max_examples: int | None = None,
    seed: int = 0,
) -> List[str]:
    """Load prompts from txt/json/jsonl.

    JSON/JSONL records can use an explicit ``text_key`` or one of TEXT_KEYS.
    Plain string JSON lists are also supported.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".txt":
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        prompts = [line for line in prompts if line]
    elif path.suffix.lower() == ".jsonl":
        prompts = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                prompts.append(_extract_prompt(json.loads(line), text_key=text_key, source=path))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if text_key and text_key in data and isinstance(data[text_key], list):
                records = data[text_key]
            else:
                records = list(data.values())
        elif isinstance(data, list):
            records = data
        else:
            raise TypeError(f"Unsupported JSON root in {path}: {type(data)}")
        prompts = [_extract_prompt(record, text_key=text_key, source=path) for record in records]
    else:
        raise ValueError(f"Unsupported prompt file extension: {path.suffix}")

    prompts = [prompt for prompt in prompts if prompt and prompt.strip()]
    if max_examples is not None and len(prompts) > max_examples:
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(prompts)), max_examples))
        prompts = [prompts[i] for i in idxs]
    return prompts


def _extract_prompt(record, *, text_key: str | None, source: Path) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        raise TypeError(f"Prompt record in {source} is not str/dict: {type(record)}")
    if text_key is not None:
        if text_key not in record:
            raise KeyError(f"{source} record does not contain requested text_key={text_key}")
        return str(record[text_key])
    for key in TEXT_KEYS:
        if key in record:
            return str(record[key])
    raise KeyError(f"Could not infer prompt text key in {source}; pass --text_key")


def load_tokenizer(base_model: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_path: str, *, dtype: torch.dtype, device: torch.device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model


def load_effective_model(
    *,
    base_model: str,
    checkpoint: str,
    dtype: torch.dtype,
    device: torch.device,
):
    """Load a full checkpoint or merge a PEFT adapter into a fresh base model."""

    if is_adapter_checkpoint(checkpoint):
        from peft import PeftModel

        base = load_causal_lm(base_model, dtype=dtype, device=device)
        model = PeftModel.from_pretrained(base, checkpoint, torch_dtype=dtype)
        model = model.merge_and_unload()
        model.to(device)
        model.eval()
        return model
    return load_causal_lm(checkpoint, dtype=dtype, device=device)


def get_submodule(model: torch.nn.Module, module_path: str) -> torch.nn.Module:
    candidates = [
        module_path,
        f"model.{module_path}",
        f"base_model.model.{module_path}",
        f"base_model.model.model.{module_path}",
    ]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return model.get_submodule(candidate)
        except AttributeError as exc:
            last_error = exc
    raise AttributeError(f"Could not resolve module {module_path}; tried {candidates}") from last_error


def get_linear_weight(model: torch.nn.Module, module_path: str) -> torch.Tensor:
    module = get_submodule(model, module_path)
    if not hasattr(module, "weight"):
        raise TypeError(f"Module {module_path} has no .weight parameter")
    weight = module.weight.detach()
    if weight.ndim != 2:
        raise ValueError(f"Module {module_path}.weight must be 2D, got {tuple(weight.shape)}")
    return weight.float().cpu()


def tokenize_batch(tokenizer, prompts: Sequence[str], *, max_length: int, device: torch.device):
    batch = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in batch.items()}


def batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def collect_module_tensor(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    module_path: str,
    *,
    capture: str,
    batch_size: int,
    max_length: int,
    max_tokens: int,
) -> torch.Tensor:
    """Collect token-level module inputs or outputs from one model.

    Returns a CPU float16 tensor of shape [n_tokens, hidden_dim].  Padding
    tokens are removed using the tokenizer attention mask.
    """

    if capture not in {"input", "output"}:
        raise ValueError("capture must be 'input' or 'output'")
    device = next(model.parameters()).device
    module = get_submodule(model, module_path)
    captured: Dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        captured["tensor"] = inputs[0].detach()

    def fwd_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured["tensor"] = output.detach()

    handle = module.register_forward_pre_hook(pre_hook) if capture == "input" else module.register_forward_hook(fwd_hook)
    chunks: List[torch.Tensor] = []
    total = 0
    try:
        with torch.no_grad():
            for prompt_batch in batched(prompts, batch_size):
                if total >= max_tokens:
                    break
                encoded = tokenize_batch(tokenizer, prompt_batch, max_length=max_length, device=device)
                mask = encoded["attention_mask"].bool()
                captured.clear()
                _ = model(**encoded)
                if "tensor" not in captured:
                    raise RuntimeError(f"Hook for {module_path} did not capture a tensor")
                tensor = captured["tensor"].float()
                if tensor.ndim != 3:
                    raise ValueError(f"Expected [batch, seq, dim] activation, got {tuple(tensor.shape)}")
                tensor = tensor[mask]
                remaining = max_tokens - total
                if tensor.shape[0] > remaining:
                    tensor = tensor[:remaining]
                chunks.append(tensor.cpu().to(torch.float16))
                total += int(tensor.shape[0])
                del encoded, tensor
    finally:
        handle.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not chunks:
        raise RuntimeError(f"No activations collected for {module_path}")
    return torch.cat(chunks, dim=0).contiguous()


def write_csv(path: str | Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    num = torch.linalg.norm(xty, ord="fro") ** 2
    den = torch.linalg.norm(xtx, ord="fro") * torch.linalg.norm(yty, ord="fro")
    return float((num / (den + eps)).item())


def procrustes_metrics(x0: torch.Tensor, x1: torch.Tensor, eps: float = 1e-12) -> Tuple[float, float, float]:
    x0 = x0.float()
    x1 = x1.float()
    x0 = x0 - x0.mean(dim=0, keepdim=True)
    x1 = x1 - x1.mean(dim=0, keepdim=True)
    denom = torch.linalg.norm(x0, ord="fro") + eps
    raw = torch.linalg.norm(x1 - x0, ord="fro") / denom
    u, _s, vh = torch.linalg.svd(x1.T @ x0, full_matrices=False)
    r = u @ vh
    proc = torch.linalg.norm(x1 @ r - x0, ord="fro") / denom
    return float(raw.item()), float(proc.item()), float((raw - proc).item())


def gram_cosine_dist(x0: torch.Tensor, x1: torch.Tensor, *, max_pair_tokens: int, seed: int, eps: float = 1e-12) -> float:
    n = min(x0.shape[0], x1.shape[0])
    if n > max_pair_tokens:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_pair_tokens, replace=False))
        x0 = x0[idx]
        x1 = x1[idx]
    z0 = torch.nn.functional.normalize(x0.float(), dim=1)
    z1 = torch.nn.functional.normalize(x1.float(), dim=1)
    k0 = z0 @ z0.T
    k1 = z1 @ z1.T
    return float((torch.linalg.norm(k1 - k0, ord="fro") / (torch.linalg.norm(k0, ord="fro") + eps)).item())


def subspace_distance(x0: torch.Tensor, x1: torch.Tensor, k: int, eps: float = 1e-12) -> float:
    n, d = x0.shape
    k_eff = min(k, n - 1, d)
    if k_eff < 1:
        return math.nan
    x0 = x0.float() - x0.float().mean(dim=0, keepdim=True)
    x1 = x1.float() - x1.float().mean(dim=0, keepdim=True)
    try:
        _, _, v0 = torch.pca_lowrank(x0, q=k_eff, center=False)
        _, _, v1 = torch.pca_lowrank(x1, q=k_eff, center=False)
    except RuntimeError:
        _, _, vh0 = torch.linalg.svd(x0, full_matrices=False)
        _, _, vh1 = torch.linalg.svd(x1, full_matrices=False)
        v0 = vh0[:k_eff].T
        v1 = vh1[:k_eff].T
    p0 = v0[:, :k_eff] @ v0[:, :k_eff].T
    p1 = v1[:, :k_eff] @ v1[:, :k_eff].T
    return float((torch.linalg.norm(p0 - p1, ord="fro") / math.sqrt(2 * k_eff + eps)).item())
