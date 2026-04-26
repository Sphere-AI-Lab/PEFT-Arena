#!/usr/bin/env python3
"""Prepare a checkpoint directory for evaluation.

This utility normalizes three checkpoint layouts into an eval-ready path:

1. Exported Hugging Face checkpoint -> returned as-is
2. PEFT adapter checkpoint -> returned as-is
3. FSDP SFT training-resume checkpoint -> exported offline either as:
   - a PEFT adapter checkpoint, or
   - a merged Hugging Face checkpoint
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
VERL_DIR = REPO_ROOT / "third_party" / "verl"

for extra_path in (str(TRAIN_DIR), str(VERL_DIR)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from peft_arena_verl.utils.checkpoint_metadata import (
    DEFAULT_BASE_MODEL_NAME_OR_PATH,
    load_checkpoint_metadata,
    normalize_model_reference,
)

LOG_PREFIX = "[prepare_eval_checkpoint]"
LORA_FAMILY = {
    "lora",
    "keeplora",
    "dora",
    "adalora",
    "pissa",
    "milora",
    "loraplus",
    "rslora",
    "qalora",
    "miss",
}
PEFT_EXPORT_MODES = {"auto", "adapter", "merged_hf"}
WRAPPED_PEFT_WEIGHT_MARKERS = ("base_model.", ".base_layer.", ".lora_", ".miss_block.", ".oft_")


@dataclass(frozen=True)
class PeftSpec:
    kind: str
    rank: int | None
    alpha: int | None
    oft_block_size: int | None
    oft_normalize_rotation: str | None
    target_modules: tuple[str, ...]


def _log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", file=sys.stderr)


def is_adapter_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return (
        os.path.exists(os.path.join(path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(path, "adapter_model.bin"))
    )


def is_exported_hf_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    model_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not any(os.path.exists(os.path.join(path, name)) for name in model_files):
        return False

    # Guard against broken "HF exports" that still contain PEFT wrapper names
    # like base_model.* or *.base_layer.*. Those are not loadable by vLLM.
    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            if any(any(marker in key for marker in WRAPPED_PEFT_WEIGHT_MARKERS) for key in weight_map):
                return False
        except Exception:
            pass

    return True


def is_fsdp_training_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "fsdp_config.json")):
        return False
    if not os.path.isdir(os.path.join(path, "huggingface")):
        return False
    shard_files = list(Path(path).glob("model_world_size_*_rank_*.pt"))
    return bool(shard_files)


def _resolve_verl_actor_path(path: str) -> str | None:
    """If *path* is a verl RL checkpoint with shards inside an ``actor/``
    sub-directory, return the actor path.  Otherwise return ``None``."""
    actor_dir = os.path.join(path, "actor")
    if os.path.isdir(actor_dir) and is_fsdp_training_checkpoint(actor_dir):
        return actor_dir
    return None


def detect_checkpoint_layout(path: str) -> str:
    if is_adapter_checkpoint(path):
        return "adapter"
    if is_exported_hf_checkpoint(path):
        return "hf"
    if is_fsdp_training_checkpoint(path):
        return "fsdp_training"
    if _resolve_verl_actor_path(path) is not None:
        return "fsdp_training"
    return "unknown"


def _collect_target_modules(state_dict: dict[str, torch.Tensor], marker: str) -> tuple[str, ...]:
    modules = set()
    pattern = re.compile(rf"\.([^.]+)\.{re.escape(marker)}(?:\.default)?\.")
    for key in state_dict:
        match = pattern.search(key)
        if match:
            modules.add(match.group(1))
    if not modules:
        raise ValueError(f"Failed to infer target_modules from state dict using marker '{marker}'")
    return tuple(sorted(modules))


def _infer_rank_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if ".lora_A." in key or ".lora_B." in key:
            if value.ndim >= 2:
                return int(min(value.shape[0], value.shape[1]))
    raise ValueError("Failed to infer LoRA rank from state dict")


def _infer_training_method_name(checkpoint_path: str) -> str:
    step_dir = Path(checkpoint_path)
    if not step_dir.name.startswith("global_step_"):
        raise ValueError(f"Expected checkpoint step dir, got {checkpoint_path}")
    return step_dir.parent.name.lower()


def _checkpoint_path_suggests_peft(checkpoint_path: str) -> bool:
    try:
        method_name = _infer_training_method_name(checkpoint_path)
    except ValueError:
        return False

    family = method_name.split("-", 1)[0]
    return family in LORA_FAMILY or family == "oft"


def _tensors_are_identical(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    return reference.shape == candidate.shape and reference.dtype == candidate.dtype and torch.equal(reference, candidate)


def _merge_unplaced_shards(
    key: str,
    shards: list[torch.Tensor],
    *,
    dtensor_checkpoint: bool,
) -> torch.Tensor:
    if not shards:
        raise ValueError(f"No shards available while merging '{key}'")

    sample = shards[0]
    if dtensor_checkpoint or sample.ndim == 0:
        reference = sample
        for shard in shards[1:]:
            if not _tensors_are_identical(reference, shard):
                raise ValueError(f"Replicated tensor '{key}' differs across FSDP shards")
        return reference.clone()

    return torch.cat(shards, dim=0)


def infer_peft_spec(checkpoint_path: str, state_dict: dict[str, torch.Tensor]) -> PeftSpec | None:
    has_lora = any(".lora_" in key for key in state_dict)
    has_miss = any(".miss_block." in key.lower() for key in state_dict)
    has_oft = any(".oft_r." in key.lower() or ".oft_" in key.lower() for key in state_dict)

    if any(".oftv3" in key.lower() for key in state_dict):
        raise ValueError("OFTv3 checkpoints are not supported in the release repo")

    if not has_lora and not has_miss and not has_oft:
        return None

    method_name = _infer_training_method_name(checkpoint_path)

    if has_miss:
        match = re.search(r"-r(\d+)", method_name)
        rank = int(match.group(1)) if match else _infer_rank_from_state_dict(state_dict)
        return PeftSpec(
            kind="miss",
            rank=rank,
            alpha=None,
            oft_block_size=None,
            oft_normalize_rotation=None,
            target_modules=_collect_target_modules(state_dict, "miss_block"),
        )

    if has_lora:
        match = re.search(r"-r(\d+)", method_name)
        rank = int(match.group(1)) if match else _infer_rank_from_state_dict(state_dict)
        alpha_match = re.search(r"-a(\d+)", method_name)
        alpha = int(alpha_match.group(1)) if alpha_match else rank * 2
        return PeftSpec(
            kind=method_name.split("-", 1)[0],
            rank=rank,
            alpha=alpha,
            oft_block_size=None,
            oft_normalize_rotation=None,
            target_modules=_collect_target_modules(state_dict, "lora_A"),
        )

    if has_oft:
        match = re.search(r"-b(\d+)", method_name)
        if not match:
            raise ValueError(f"Failed to infer OFT block size from '{method_name}'")
        kind = "oft"
        marker = "oft_R"
        
        normalize_rot = "none"
        if "learnable" in method_name:
            normalize_rot = "learnable"
        elif "mean_norm" in method_name:
            normalize_rot = "mean_norm"
        elif "upper_clamp" in method_name:
            normalize_rot = "upper_clamp"

        return PeftSpec(
            kind=kind,
            rank=None,
            alpha=None,
            oft_block_size=int(match.group(1)),
            oft_normalize_rotation=normalize_rot,
            target_modules=_collect_target_modules(state_dict, marker),
        )

    return None


def _build_peft_config(spec: PeftSpec):
    from peft import AdaLoraConfig, LoraConfig, MissConfig, OFTConfig, TaskType

    if spec.kind == "adalora":
        return AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=spec.rank,
            lora_alpha=spec.alpha,
            lora_dropout=0.05,
            target_modules=list(spec.target_modules),
            bias="none",
        )
    if spec.kind == "miss":
        return MissConfig(
            task_type=TaskType.CAUSAL_LM,
            r=spec.rank,
            miss_dropout=0.05,
            target_modules=list(spec.target_modules),
            bias="none",
        )
    if spec.kind in LORA_FAMILY:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=spec.rank,
            lora_alpha=spec.alpha,
            lora_dropout=0.05,
            target_modules=list(spec.target_modules),
            use_dora=spec.kind == "dora",
            use_rslora=spec.kind == "rslora",
            use_qalora=spec.kind == "qalora",
            bias="none",
        )
    if spec.kind == "oft":
        kwargs = {
            "task_type": TaskType.CAUSAL_LM,
            "oft_block_size": spec.oft_block_size,
            "target_modules": list(spec.target_modules),
            "block_partition_method": "index",
            "block_ratio": 1.0,
            "doft": False,
        }
        supported_kwargs = inspect.signature(OFTConfig).parameters
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_kwargs}
        return OFTConfig(**filtered_kwargs)
    raise NotImplementedError(f"Unsupported PEFT checkpoint type '{spec.kind}'")


def _copy_processing_artifacts(hf_source_dir: str, output_dir: str, trust_remote_code: bool) -> None:
    from transformers import GenerationConfig
    from verl.utils import hf_processor, hf_tokenizer

    processor = hf_processor(hf_source_dir, trust_remote_code=trust_remote_code)
    tokenizer = hf_tokenizer(hf_source_dir, trust_remote_code=trust_remote_code)
    if processor is not None:
        processor.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    try:
        generation_config = GenerationConfig.from_pretrained(hf_source_dir)
        generation_config.save_pretrained(output_dir)
    except OSError:
        pass


def _resolve_base_model_name_or_path(checkpoint_path: str) -> str:
    metadata = load_checkpoint_metadata(checkpoint_path)
    from_metadata = normalize_model_reference(metadata.get("base_model_name_or_path"))
    if from_metadata:
        return from_metadata

    hf_config_path = Path(checkpoint_path) / "huggingface" / "config.json"
    if hf_config_path.exists():
        try:
            import json

            with open(hf_config_path, encoding="utf-8") as f:
                hf_config = json.load(f)
            for key in ("_name_or_path", "name_or_path"):
                value = normalize_model_reference(hf_config.get(key))
                if value:
                    return value
        except Exception:
            pass

    return DEFAULT_BASE_MODEL_NAME_OR_PATH


def _merge_fsdp_training_state_dict(checkpoint_path: str, target_dir: str):
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor

    class SafeFSDPModelMerger(FSDPModelMerger):
        @staticmethod
        def _prepare_tensor_for_merge(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.is_floating_point():
                return tensor.bfloat16()
            return tensor

        def _load_and_merge_state_dicts(
            self,
            world_size: int,
            total_shards: int,
            mesh_shape: tuple[int, ...],
            mesh_dim_names: tuple[str, ...],
        ) -> dict[str, torch.Tensor]:
            model_state_dict_lst = [None] * total_shards

            def process_one_shard(rank: int, model_state_dict_list: list):
                model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                model_state_dict_list[rank] = state_dict
                return state_dict

            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
                futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
                for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                    future.result()

            state_dict: dict[str, list[torch.Tensor] | torch.Tensor] = {}
            param_placements: dict[str, tuple] = {}
            has_dtensor_entries = False

            for key in set(model_state_dict_lst[0].keys()):
                state_dict[key] = []
                for model_state_shard in model_state_dict_lst:
                    tensor = model_state_shard.pop(key)
                    if isinstance(tensor, DTensor):
                        has_dtensor_entries = True
                        state_dict[key].append(self._prepare_tensor_for_merge(tensor._local_tensor))

                        placements = tuple(tensor.placements)
                        if mesh_dim_names[0] in ("dp", "ddp"):
                            placements = placements[1:]

                        if key not in param_placements:
                            param_placements[key] = placements
                        else:
                            assert param_placements[key] == placements
                    else:
                        state_dict[key].append(self._prepare_tensor_for_merge(tensor))

            del model_state_dict_lst

            for key in sorted(state_dict):
                if not isinstance(state_dict[key], list):
                    continue
                if key in param_placements:
                    placements = param_placements[key]
                    if len(mesh_shape) == 1:
                        assert len(placements) == 1
                        state_dict[key] = self._merge_by_placement(state_dict[key], placements[0])
                    else:
                        raise NotImplementedError("FSDP + TP is not supported yet")
                else:
                    state_dict[key] = _merge_unplaced_shards(
                        key,
                        state_dict[key],
                        dtensor_checkpoint=has_dtensor_entries,
                    )

            return state_dict

    merger = SafeFSDPModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            target_dir=target_dir,
            local_dir=checkpoint_path,
            hf_model_config_path=os.path.join(checkpoint_path, "huggingface"),
        )
    )
    world_size = merger._get_world_size()
    rank_zero_state_dict = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(rank_zero_state_dict, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_dim_names)
    _log(f"Merging {total_shards} FSDP shards from {checkpoint_path}")
    merged_state_dict = merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)
    return merger, merged_state_dict


def _export_full_model(merger, checkpoint_path: str, state_dict: dict[str, torch.Tensor]) -> None:
    os.makedirs(merger.config.target_dir, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        merger.save_hf_model_and_tokenizer(state_dict)
    config_path = Path(merger.config.target_dir) / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        config["_name_or_path"] = _resolve_base_model_name_or_path(checkpoint_path)
        for tensor in state_dict.values():
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                config["torch_dtype"] = str(tensor.dtype).replace("torch.", "")
                break
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
            f.write("\n")


def _load_peft_model_from_state_dict(
    merger,
    checkpoint_path: str,
    state_dict: dict[str, torch.Tensor],
) -> tuple[PeftSpec, object]:
    from accelerate import init_empty_weights
    from peft import get_peft_model

    spec = infer_peft_spec(checkpoint_path, state_dict)
    if spec is None:
        raise ValueError("PEFT checkpoint expected, but no PEFT spec could be inferred")
    base_model_name_or_path = _resolve_base_model_name_or_path(checkpoint_path)

    auto_model_class = merger.get_transformers_auto_model_class()
    with init_empty_weights():
        base_model = auto_model_class.from_config(
            merger.model_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=merger.config.trust_remote_code,
        )
    base_model.to_empty(device="cpu")
    base_model.name_or_path = base_model_name_or_path
    base_model.config._name_or_path = base_model_name_or_path
    peft_model = get_peft_model(base_model, _build_peft_config(spec))
    for peft_config in peft_model.peft_config.values():
        peft_config.base_model_name_or_path = base_model_name_or_path

    missing, unexpected = peft_model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys while loading merged PEFT state dict: {sorted(unexpected)[:10]}")

    if missing:
        adapter_missing = [key for key in missing if "lora_" in key or "oft" in key.lower()]
        if adapter_missing:
            raise ValueError(f"Missing adapter keys while loading merged PEFT state dict: {adapter_missing[:10]}")

    return spec, peft_model


def _export_peft_adapter_model(
    merger,
    checkpoint_path: str,
    state_dict: dict[str, torch.Tensor],
    output_dir: str,
) -> None:
    spec, peft_model = _load_peft_model_from_state_dict(merger, checkpoint_path, state_dict)
    peft_model.save_pretrained(output_dir)
    _copy_processing_artifacts(merger.hf_model_config_path, output_dir, merger.config.trust_remote_code)
    _log(f"Saved {spec.kind} adapter export to {output_dir}")


def _export_peft_merged_model(
    merger,
    checkpoint_path: str,
    state_dict: dict[str, torch.Tensor],
    output_dir: str,
) -> None:
    from transformers import GenerationConfig

    spec, peft_model = _load_peft_model_from_state_dict(merger, checkpoint_path, state_dict)
    base_model_name_or_path = _resolve_base_model_name_or_path(checkpoint_path)

    merged_model = peft_model.merge_and_unload()
    merged_model.config._name_or_path = base_model_name_or_path
    try:
        merged_model.generation_config = GenerationConfig.from_pretrained(merger.hf_model_config_path)
    except OSError:
        pass
    merged_model.save_pretrained(output_dir)
    _copy_processing_artifacts(merger.hf_model_config_path, output_dir, merger.config.trust_remote_code)
    _log(f"Saved merged {spec.kind} export to {output_dir}")


def _resolve_output_path(checkpoint_path: str, output_path: str | None, peft_export_mode: str) -> str:
    if output_path is not None:
        return output_path
    if peft_export_mode == "merged_hf":
        return checkpoint_path + "_exported"
    if peft_export_mode == "adapter":
        if _checkpoint_path_suggests_peft(checkpoint_path):
            return checkpoint_path + "_adapter_exported"
        return checkpoint_path + "_exported"
    if _checkpoint_path_suggests_peft(checkpoint_path):
        return checkpoint_path + "_adapter_exported"
    return checkpoint_path + "_exported"


def _expected_reuse_layout(checkpoint_path: str, peft_export_mode: str) -> str:
    if peft_export_mode == "merged_hf":
        return "hf"
    if peft_export_mode == "adapter":
        return "adapter" if _checkpoint_path_suggests_peft(checkpoint_path) else "hf"
    return "adapter" if _checkpoint_path_suggests_peft(checkpoint_path) else "hf"


def export_fsdp_training_checkpoint(
    checkpoint_path: str,
    output_path: str,
    peft_export_mode: str = "auto",
    *,
    fsdp_shard_path: str | None = None,
) -> str:
    if peft_export_mode not in PEFT_EXPORT_MODES:
        raise ValueError(f"Unsupported peft_export_mode '{peft_export_mode}'")

    # If FSDP shards live in a sub-directory (e.g. verl ``actor/``), use that
    # for merging while keeping *checkpoint_path* for method-name inference.
    shard_path = fsdp_shard_path or checkpoint_path

    tmp_output = output_path + ".tmp"
    shutil.rmtree(tmp_output, ignore_errors=True)
    os.makedirs(tmp_output, exist_ok=True)

    merger, merged_state_dict = _merge_fsdp_training_state_dict(shard_path, tmp_output)

    try:
        spec = infer_peft_spec(checkpoint_path, merged_state_dict)
        if spec is None:
            _log("Detected full-finetuning checkpoint; exporting as Hugging Face checkpoint")
            _export_full_model(merger, checkpoint_path, merged_state_dict)
        else:
            effective_export_mode = "adapter" if peft_export_mode == "auto" else peft_export_mode
            if effective_export_mode == "adapter":
                _log(f"Detected {spec.kind} checkpoint; exporting adapter checkpoint")
                _export_peft_adapter_model(merger, checkpoint_path, merged_state_dict, tmp_output)
            else:
                _log(f"Detected {spec.kind} checkpoint; exporting merged Hugging Face checkpoint")
                _export_peft_merged_model(merger, checkpoint_path, merged_state_dict, tmp_output)

        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        shutil.move(tmp_output, output_path)
        return output_path
    except Exception:
        shutil.rmtree(tmp_output, ignore_errors=True)
        raise
    finally:
        merger.cleanup()


def prepare_eval_checkpoint(
    checkpoint_path: str,
    output_path: str | None = None,
    peft_export_mode: str = "auto",
) -> str:
    if peft_export_mode not in PEFT_EXPORT_MODES:
        raise ValueError(f"Unsupported peft_export_mode '{peft_export_mode}'")

    layout = detect_checkpoint_layout(checkpoint_path)

    if layout in {"adapter", "hf"}:
        _log(f"Checkpoint already eval-ready ({layout}): {checkpoint_path}")
        return checkpoint_path

    if layout != "fsdp_training":
        raise ValueError(
            f"Unsupported checkpoint layout at {checkpoint_path}. Expected adapter, exported HF model, or FSDP training checkpoint."
        )

    output_path = _resolve_output_path(checkpoint_path, output_path, peft_export_mode)

    expected_layout = _expected_reuse_layout(checkpoint_path, peft_export_mode)
    if expected_layout == "adapter" and is_adapter_checkpoint(output_path):
        _log(f"Reusing existing exported adapter checkpoint: {output_path}")
        return output_path
    if expected_layout == "hf" and is_exported_hf_checkpoint(output_path):
        _log(f"Reusing existing exported checkpoint: {output_path}")
        return output_path
    # Resolve verl actor/ sub-directory if present
    fsdp_shard_path = _resolve_verl_actor_path(checkpoint_path)
    if fsdp_shard_path:
        _log(f"Detected verl RL checkpoint; using actor sub-directory: {fsdp_shard_path}")

    return export_fsdp_training_checkpoint(
        checkpoint_path, output_path, peft_export_mode=peft_export_mode,
        fsdp_shard_path=fsdp_shard_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a checkpoint path for evaluation")
    parser.add_argument("--checkpoint_path", required=True, help="Input checkpoint path")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output path for exported training checkpoints. Defaults depend on peft_export_mode",
    )
    parser.add_argument(
        "--peft_export_mode",
        choices=sorted(PEFT_EXPORT_MODES),
        default="auto",
        help="How to export PEFT training checkpoints: auto, adapter, or merged_hf",
    )
    args = parser.parse_args()

    prepared_path = prepare_eval_checkpoint(
        checkpoint_path=os.path.abspath(args.checkpoint_path),
        output_path=os.path.abspath(args.output_path) if args.output_path else None,
        peft_export_mode=args.peft_export_mode,
    )
    print(prepared_path)


if __name__ == "__main__":
    main()
