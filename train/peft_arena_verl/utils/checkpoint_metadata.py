"""Helpers for PEFTArena checkpoint metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_BASE_MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-7B"
CHECKPOINT_METADATA_FILENAME = "peft_arena_checkpoint_meta.json"


def normalize_model_reference(path_or_id: str | None) -> str | None:
    if not path_or_id:
        return None
    value = os.path.expanduser(str(path_or_id))
    if os.path.isdir(value):
        return os.path.abspath(value)
    return str(path_or_id)


def build_checkpoint_metadata(
    *,
    base_model_name_or_path: str | None,
    tokenizer_name_or_path: str | None = None,
) -> dict:
    metadata = {
        "format_version": 1,
        "base_model_name_or_path": normalize_model_reference(base_model_name_or_path) or DEFAULT_BASE_MODEL_NAME_OR_PATH,
    }
    normalized_tokenizer = normalize_model_reference(tokenizer_name_or_path)
    if normalized_tokenizer:
        metadata["tokenizer_name_or_path"] = normalized_tokenizer
    return metadata


def save_checkpoint_metadata(
    checkpoint_dir: str,
    *,
    base_model_name_or_path: str | None,
    tokenizer_name_or_path: str | None = None,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    metadata_path = os.path.join(checkpoint_dir, CHECKPOINT_METADATA_FILENAME)
    metadata = build_checkpoint_metadata(
        base_model_name_or_path=base_model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata_path


def load_checkpoint_metadata(checkpoint_dir: str) -> dict:
    metadata_path = Path(checkpoint_dir) / CHECKPOINT_METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)
