import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
VERL_DIR = REPO_ROOT / "third_party" / "verl"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(VERL_DIR) not in sys.path:
    sys.path.insert(0, str(VERL_DIR))


class DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        text = " ".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += " assistant"
        return text.strip()

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        n_tokens = max(1, len(text.split()))
        ids = torch.arange(1, n_tokens + 1, dtype=torch.long).unsqueeze(0)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }


def test_sft_dataset_filters_overlong_examples(tmp_path):
    from peft_arena_verl.utils.dataset.sft_dataset import SFTDataset

    parquet_path = tmp_path / "sft.parquet"
    pd.DataFrame(
        [
            {"extra_info": {"question": "short question", "answer": "short answer"}},
            {
                "extra_info": {
                    "question": " ".join(["long"] * 20),
                    "answer": " ".join(["answer"] * 10),
                }
            },
        ]
    ).to_parquet(parquet_path)

    cfg = {
        "prompt_key": "extra_info",
        "response_key": "extra_info",
        "prompt_dict_keys": ["question"],
        "response_dict_keys": ["answer"],
        "max_length": 10,
        "truncation": "error",
        "filter_overlong_examples": True,
        "use_shm": False,
        "apply_chat_template_kwargs": {},
    }

    dataset = SFTDataset(str(parquet_path), DummyTokenizer(), cfg)
    sample = dataset[0]

    assert len(dataset) == 1
    assert tuple(sample["input_ids"].shape) == (10,)
    assert int(sample["loss_mask"].sum().item()) > 0


def test_sft_dataset_without_filtering_keeps_overlong_examples(tmp_path):
    from peft_arena_verl.utils.dataset.sft_dataset import SFTDataset

    parquet_path = tmp_path / "sft.parquet"
    pd.DataFrame(
        [
            {"extra_info": {"question": "short question", "answer": "short answer"}},
            {
                "extra_info": {
                    "question": " ".join(["long"] * 20),
                    "answer": " ".join(["answer"] * 10),
                }
            },
        ]
    ).to_parquet(parquet_path)

    cfg = {
        "prompt_key": "extra_info",
        "response_key": "extra_info",
        "prompt_dict_keys": ["question"],
        "response_dict_keys": ["answer"],
        "max_length": 10,
        "truncation": "error",
        "filter_overlong_examples": False,
        "use_shm": False,
        "apply_chat_template_kwargs": {},
    }

    dataset = SFTDataset(str(parquet_path), DummyTokenizer(), cfg)

    assert len(dataset) == 2
    with pytest.raises(NotImplementedError, match="larger than"):
        dataset[1]
