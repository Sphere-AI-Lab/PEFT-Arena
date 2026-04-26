import json
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
VERL_DIR = REPO_ROOT / "third_party" / "verl"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(VERL_DIR) not in sys.path:
    sys.path.insert(0, str(VERL_DIR))


def _write_legacy_list_metadata_parquet(path: Path):
    pyarrow = pytest.importorskip("pyarrow")
    pa = pyarrow
    pq = pytest.importorskip("pyarrow.parquet")

    rows = [
        {
            "system": "You are a helpful assistant.",
            "data_source": "openr1",
            "prompt": [{"content": "hello", "role": "user"}],
            "ability": "math",
            "reward_model": {"ground_truth": "42", "style": "rule"},
            "extra_info": {
                "answer": "42",
                "index": 0,
                "question": "What is 6 * 7?",
                "split": "train",
            },
        }
    ]
    features = {
        "system": {"dtype": "string", "_type": "Value"},
        "data_source": {"dtype": "string", "_type": "Value"},
        "prompt": {
            "feature": {
                "content": {"dtype": "string", "_type": "Value"},
                "role": {"dtype": "string", "_type": "Value"},
            },
            "_type": "List",
        },
        "ability": {"dtype": "string", "_type": "Value"},
        "reward_model": {
            "ground_truth": {"dtype": "string", "_type": "Value"},
            "style": {"dtype": "string", "_type": "Value"},
        },
        "extra_info": {
            "answer": {"dtype": "string", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "question": {"dtype": "string", "_type": "Value"},
            "split": {"dtype": "string", "_type": "Value"},
        },
    }

    table = pa.Table.from_pylist(rows)
    table = table.replace_schema_metadata(
        {b"huggingface": json.dumps({"info": {"features": features}}).encode("utf-8")}
    )
    pq.write_table(table, path)


class _DummyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert add_generation_prompt is True
        assert tokenize is False
        rendered = []
        for message in messages:
            rendered.append(f"{message['role']}: {message['content']}")
        return "\n".join(rendered)

    def __call__(self, raw_prompt, return_tensors="pt", add_special_tokens=False):
        del return_tensors, add_special_tokens
        values = [ord(ch) % 31 + 1 for ch in raw_prompt]
        if not values:
            values = [1]
        input_ids = torch.tensor([values], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, raw_prompt, add_special_tokens=False):
        del add_special_tokens
        values = [ord(ch) % 31 + 1 for ch in raw_prompt]
        return values or [1]


def test_sanitize_huggingface_parquet_metadata_allows_load_dataset(tmp_path):
    datasets = pytest.importorskip("datasets")
    datasets_cache = tmp_path / "hf_datasets"

    parquet_path = tmp_path / "legacy_list.parquet"
    _write_legacy_list_metadata_parquet(parquet_path)

    with pytest.raises(ValueError, match="Feature type 'List' not found"):
        datasets.load_dataset(
            "parquet",
            data_files=str(parquet_path),
            cache_dir=str(datasets_cache),
        )["train"]

    from peft_arena_verl.utils.dataset.rl_dataset import sanitize_huggingface_parquet_metadata

    compat_path = sanitize_huggingface_parquet_metadata(str(parquet_path), str(tmp_path / "cache"))

    dataset = datasets.load_dataset(
        "parquet",
        data_files=compat_path,
        cache_dir=str(datasets_cache),
    )["train"]
    sample = dataset[0]

    assert compat_path != str(parquet_path)
    assert sample["prompt"][0]["role"] == "user"
    assert sample["extra_info"]["question"] == "What is 6 * 7?"


def test_peft_arena_rl_dataset_handles_legacy_list_metadata(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("pyarrow")

    from peft_arena_verl.utils.dataset.rl_dataset import RLHFDataset

    parquet_path = tmp_path / "legacy_list.parquet"
    _write_legacy_list_metadata_parquet(parquet_path)

    dataset = RLHFDataset(
        data_files=str(parquet_path),
        tokenizer=_DummyTokenizer(),
        processor=None,
        config={
            "cache_dir": str(tmp_path / "cache"),
            "prompt_key": "prompt",
            "filter_overlong_prompts": False,
            "return_raw_chat": True,
        },
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["raw_prompt"][0]["content"] == "hello"
    assert sample["raw_prompt"][0]["role"] == "user"
