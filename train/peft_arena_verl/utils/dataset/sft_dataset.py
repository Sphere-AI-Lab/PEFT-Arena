import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

try:
    from omegaconf.listconfig import ListConfig
except Exception:  # pragma: no cover - environment dependent
    ListConfig = tuple

try:
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local
    from verl.utils.model import compute_position_id_with_mask
except Exception:  # pragma: no cover - environment dependent
    def hf_tokenizer(tokenizer_name):
        raise RuntimeError(f"verl is unavailable, cannot load tokenizer '{tokenizer_name}'")

    def copy_to_local(path, verbose=True, use_shm=False):
        del verbose, use_shm
        return path

    def compute_position_id_with_mask(attention_mask):
        if attention_mask.numel() == 0:
            return torch.zeros(0, dtype=torch.long)
        return torch.cumsum(attention_mask.to(dtype=torch.long), dim=-1) - 1


class SFTDataset(Dataset):
    """PEFTArena-owned SFT dataset with max-length filtering."""

    def __init__(self, parquet_files, tokenizer, config, max_samples: int = -1):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.filter_overlong_examples = config.get("filter_overlong_examples", True)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.max_samples = max_samples
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                if isinstance(ls, pandas.core.series.Series):
                    ls = ls.iloc[0]
                else:
                    ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()

        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

        if self.filter_overlong_examples:
            self._filter_overlong_examples()

    def _get_sequence_length(self, prompt, response):
        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_chat_str = self.tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        response_chat_str = response + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        response_ids = self.tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        return int(prompt_ids.shape[0] + response_ids.shape[0])

    def _filter_overlong_examples(self):
        kept_prompts = []
        kept_responses = []

        for prompt, response in zip(self.prompts, self.responses):
            sequence_length = self._get_sequence_length(prompt, response)
            if sequence_length <= self.max_length:
                kept_prompts.append(prompt)
                kept_responses.append(response)

        filtered = len(self.prompts) - len(kept_prompts)
        if filtered > 0:
            print(f"filtered {filtered} overlong SFT examples with max_length={self.max_length}")

        self.prompts = kept_prompts
        self.responses = kept_responses

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        prompt_chat = [{"role": "user", "content": prompt}]

        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        response_chat_str = response + tokenizer.eos_token

        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
