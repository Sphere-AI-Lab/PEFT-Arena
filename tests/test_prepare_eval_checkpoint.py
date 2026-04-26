from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_eval_checkpoint import (
    _build_peft_config,
    _merge_unplaced_shards,
    _resolve_base_model_name_or_path,
    _resolve_output_path,
    detect_checkpoint_layout,
    infer_peft_spec,
    prepare_eval_checkpoint,
)


def test_detect_checkpoint_layouts(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"stub")
    assert detect_checkpoint_layout(str(adapter_dir)) == "adapter"

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")
    (hf_dir / "model.safetensors").write_bytes(b"stub")
    assert detect_checkpoint_layout(str(hf_dir)) == "hf"

    fsdp_dir = tmp_path / "lora-r8" / "global_step_91"
    (fsdp_dir / "huggingface").mkdir(parents=True)
    (fsdp_dir / "fsdp_config.json").write_text('{"world_size": 8}')
    (fsdp_dir / "model_world_size_8_rank_0.pt").write_bytes(b"stub")
    assert detect_checkpoint_layout(str(fsdp_dir)) == "fsdp_training"


def test_infer_lora_spec_from_state_dict():
    state_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.zeros(8, 64),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.zeros(64, 8),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight": torch.zeros(8, 64),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight": torch.zeros(64, 8),
    }
    checkpoint_path = "/tmp/math/qwen2.5-7b/lora-r8/global_step_91"
    spec = infer_peft_spec(checkpoint_path, state_dict)
    assert spec is not None
    assert spec.kind == "lora"
    assert spec.rank == 8
    assert spec.alpha == 16
    assert spec.target_modules == ("q_proj", "v_proj")


def test_infer_oftv3_state_dict_is_rejected():
    state_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.oftv3_R.default.weight": torch.zeros(4, 16, 16),
        "base_model.model.model.layers.0.self_attn.q_proj.oftv3_R.default.permutation": torch.zeros(16),
        "base_model.model.model.layers.0.self_attn.k_proj.oftv3_R.default.weight": torch.zeros(4, 16, 16),
    }
    checkpoint_path = "/tmp/med/qwen2.5-7b/oft-b16/global_step_91"
    with pytest.raises(ValueError, match="OFTv3"):
        infer_peft_spec(checkpoint_path, state_dict)


def test_infer_oft_spec_from_standard_oft_state_dict():
    state_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.oft_R.default.weight": torch.zeros(4, 16, 16),
        "base_model.model.model.layers.0.self_attn.q_proj.oft_R.default.permutation": torch.zeros(16, dtype=torch.long),
        "base_model.model.model.layers.0.self_attn.k_proj.oft_R.default.weight": torch.zeros(4, 16, 16),
    }
    checkpoint_path = "/tmp/med/qwen2.5-7b/oft-b16/global_step_91"
    spec = infer_peft_spec(checkpoint_path, state_dict)
    assert spec is not None
    assert spec.kind == "oft"
    assert spec.oft_block_size == 16
    assert spec.target_modules == ("k_proj", "q_proj")


def test_merge_unplaced_shards_uses_first_value_for_dtensor_checkpoint_buffers():
    merged = _merge_unplaced_shards(
        "buffer",
        [
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([0, 1, 2], dtype=torch.long),
        ],
        dtensor_checkpoint=True,
    )
    assert torch.equal(merged, torch.tensor([0, 1, 2], dtype=torch.long))


def test_merge_unplaced_shards_concats_legacy_non_dtensor_shards():
    merged = _merge_unplaced_shards(
        "legacy_weight",
        [
            torch.ones(2, 4),
            torch.zeros(2, 4),
        ],
        dtensor_checkpoint=False,
    )
    assert merged.shape == (4, 4)


def test_prepare_eval_checkpoint_passthrough(tmp_path: Path):
    hf_dir = tmp_path / "exported"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")
    (hf_dir / "model.safetensors").write_bytes(b"stub")
    assert prepare_eval_checkpoint(str(hf_dir)) == str(hf_dir)


def test_resolve_output_path_prefers_adapter_export_for_peft_auto_mode():
    checkpoint_path = "/tmp/math/qwen2.5-7b/lora-r8/global_step_91"
    assert _resolve_output_path(checkpoint_path, None, "auto") == checkpoint_path + "_adapter_exported"


def test_resolve_output_path_keeps_hf_export_for_full_auto_mode():
    checkpoint_path = "/tmp/math/qwen2.5-7b/full/global_step_91"
    assert _resolve_output_path(checkpoint_path, None, "auto") == checkpoint_path + "_exported"


def test_resolve_output_path_keeps_hf_export_for_full_adapter_mode():
    checkpoint_path = "/tmp/math/qwen2.5-7b/full/global_step_91"
    assert _resolve_output_path(checkpoint_path, None, "adapter") == checkpoint_path + "_exported"


def test_build_oft_config_filters_version_specific_kwargs():
    try:
        import peft  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"peft is unavailable in this environment: {exc}")

    spec = infer_peft_spec(
        "/tmp/med/qwen2.5-7b/oft-b16/global_step_91",
        {
            "base_model.model.model.layers.0.self_attn.q_proj.oft_R.default.weight": torch.zeros(4, 16, 16),
            "base_model.model.model.layers.0.self_attn.q_proj.oft_R.default.permutation": torch.zeros(16, dtype=torch.long),
        },
    )
    assert spec is not None
    config = _build_peft_config(spec)
    assert config.peft_type.value.lower() == "oft"
    assert config.oft_block_size == 16


def test_prepare_eval_checkpoint_reuses_existing_adapter_export(tmp_path: Path):
    checkpoint_dir = tmp_path / "math" / "qwen2.5-7b" / "lora-r8" / "global_step_91"
    (checkpoint_dir / "huggingface").mkdir(parents=True)
    (checkpoint_dir / "fsdp_config.json").write_text('{"world_size": 8}')
    (checkpoint_dir / "model_world_size_8_rank_0.pt").write_bytes(b"stub")

    adapter_export = checkpoint_dir.with_name(checkpoint_dir.name + "_adapter_exported")
    adapter_export.mkdir()
    (adapter_export / "adapter_model.safetensors").write_bytes(b"stub")

    assert prepare_eval_checkpoint(str(checkpoint_dir), peft_export_mode="adapter") == str(adapter_export)


def test_prepare_eval_checkpoint_prefers_canonical_full_export_name(tmp_path: Path):
    checkpoint_dir = tmp_path / "math" / "qwen2.5-7b" / "full" / "global_step_91"
    (checkpoint_dir / "huggingface").mkdir(parents=True)
    (checkpoint_dir / "fsdp_config.json").write_text('{"world_size": 8}')
    (checkpoint_dir / "model_world_size_8_rank_0.pt").write_bytes(b"stub")

    legacy_export = checkpoint_dir.with_name(checkpoint_dir.name + "_adapter_exported")
    legacy_export.mkdir()
    (legacy_export / "config.json").write_text("{}")
    (legacy_export / "model.safetensors").write_bytes(b"stub")

    canonical_export = checkpoint_dir.with_name(checkpoint_dir.name + "_exported")
    canonical_export.mkdir()
    (canonical_export / "config.json").write_text("{}")
    (canonical_export / "model.safetensors").write_bytes(b"stub")

    assert prepare_eval_checkpoint(str(checkpoint_dir), peft_export_mode="adapter") == str(canonical_export)


def test_resolve_base_model_name_or_path_prefers_saved_metadata(tmp_path: Path):
    checkpoint_dir = tmp_path / "global_step_91"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "peft_arena_checkpoint_meta.json").write_text(
        '{"format_version": 1, "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}'
    )
    assert _resolve_base_model_name_or_path(str(checkpoint_dir)) == "Qwen/Qwen2.5-7B-Instruct"


def test_resolve_base_model_name_or_path_defaults_for_legacy_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "global_step_91"
    (checkpoint_dir / "huggingface").mkdir(parents=True)
    (checkpoint_dir / "huggingface" / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    assert _resolve_base_model_name_or_path(str(checkpoint_dir)) == "Qwen/Qwen2.5-7B"
