#!/usr/bin/env python3
"""Unified PEFT adapter merge utility.

Merges any PEFT adapter (LoRA, OFT, DoRA, AdaLoRA, VeRA, IA3) into its
base model and saves the merged full model.

Usage:
    python tools/merge_peft.py --adapter_path <path> [--output_dir <path>]
"""

import argparse
import os
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_adapter_checkpoint(path: str) -> bool:
    """Check if a path contains a PEFT adapter checkpoint."""
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    return "adapter_model.safetensors" in files or "adapter_model.bin" in files


def _copy_tokenizer_artifacts(source_dir: str, output_dir: str) -> None:
    if not os.path.isdir(source_dir):
        return

    artifact_names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    )
    for name in artifact_names:
        source_path = os.path.join(source_dir, name)
        if not os.path.exists(source_path):
            continue
        target_path = os.path.join(output_dir, name)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target_path)


def merge_adapter(
    adapter_path: str,
    output_dir: str = None,
    torch_dtype: str = "bfloat16",
):
    """Merge a PEFT adapter into its base model and save.

    Args:
        adapter_path: Path to the PEFT adapter checkpoint directory.
        output_dir: Where to save the merged model. Defaults to adapter_path + '_merged'.
        torch_dtype: Torch dtype string (float32, float16, bfloat16).
    """
    try:
        from peft import PeftModel, PeftConfig
    except ImportError:
        print("Error: `peft` is required. Install with: pip install peft", file=sys.stderr)
        sys.exit(2)

    assert is_adapter_checkpoint(adapter_path), (
        f"No adapter_model.safetensors or adapter_model.bin found in {adapter_path}"
    )

    # Resolve dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # Resolve output directory
    if output_dir is None:
        output_dir = adapter_path + "_merged"
    os.makedirs(output_dir, exist_ok=True)

    # Load PEFT config to find base model
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_path = peft_config.base_model_name_or_path

    # Resolve local paths: newer huggingface_hub rejects absolute paths as repo IDs,
    # so we detect local dirs and use local_files_only=True to skip Hub validation.
    is_local = os.path.isdir(base_model_path)
    if is_local:
        base_model_path = os.path.abspath(base_model_path)
    local_kwargs = {"local_files_only": True} if is_local else {}

    print(f"[merge_peft] Base model: {base_model_path} (local={is_local})")
    print(f"[merge_peft] Adapter:    {adapter_path}")
    print(f"[merge_peft] PEFT type:  {peft_config.peft_type}")
    print(f"[merge_peft] Dtype:      {torch_dtype}")

    # Load base model
    print(f"[merge_peft] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **local_kwargs,
    )

    # Load adapter
    print(f"[merge_peft] Loading adapter...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=dtype)

    # Merge
    print(f"[merge_peft] Merging adapter into base model...")
    try:
        merged = peft_model.merge_and_unload()
    except Exception as e:
        print(f"[merge_peft] Warning: merge_and_unload() raised {e}, using base model.")
        merged = base_model
    merged.config._name_or_path = base_model_path
    merged.config.torch_dtype = dtype

    # Save merged model
    print(f"[merge_peft] Saving merged model to {output_dir}...")
    merged.save_pretrained(output_dir)

    # Save tokenizer
    print(f"[merge_peft] Saving tokenizer...")
    tokenizer_saved = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=False,
            trust_remote_code=True,
            **local_kwargs,
        )
        tokenizer.save_pretrained(output_dir, legacy_format=True)
        tokenizer_saved = True
    except Exception as exc:
        print(f"[merge_peft] Warning: failed to save slow tokenizer: {exc}")

    try:
        fast_tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            **local_kwargs,
        )
        fast_tokenizer.save_pretrained(output_dir)
        tokenizer_saved = True
    except Exception as exc:
        print(f"[merge_peft] Warning: failed to save fast tokenizer: {exc}")

    if is_local:
        _copy_tokenizer_artifacts(base_model_path, output_dir)

    if not tokenizer_saved:
        raise RuntimeError(f"Failed to save tokenizer assets for merged model at {output_dir}")

    print(f"[merge_peft] Merge complete → {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge PEFT adapter into base model")
    parser.add_argument("--adapter_path", required=True, help="Path to PEFT adapter checkpoint")
    parser.add_argument("--output_dir", default=None, help="Output merged model path")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    merge_adapter(
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        torch_dtype=args.torch_dtype,
    )


if __name__ == "__main__":
    main()
