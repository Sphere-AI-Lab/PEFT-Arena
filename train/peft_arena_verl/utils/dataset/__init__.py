from .sft_dataset import SFTDataset

__all__ = ["RLHFDataset", "SFTDataset", "collate_fn", "get_dataset_class", "sanitize_huggingface_parquet_metadata"]


def __getattr__(name):
    if name in {"RLHFDataset", "collate_fn", "get_dataset_class", "sanitize_huggingface_parquet_metadata"}:
        from . import rl_dataset

        return getattr(rl_dataset, name)
    raise AttributeError(name)
