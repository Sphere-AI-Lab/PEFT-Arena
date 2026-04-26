import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
VERL_DIR = REPO_ROOT / "third_party" / "verl"
VERL_CONFIG_DIR = REPO_ROOT / "train" / "peft_arena_verl" / "trainer" / "config"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(VERL_DIR) not in sys.path:
    sys.path.insert(0, str(VERL_DIR))


def _compose_config(config_name: str, overrides: list[str]):
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"hydra is unavailable in this environment: {exc}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(VERL_CONFIG_DIR), version_base=None
    ):
        cfg = compose(config_name=config_name, overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


def test_sft_trainer_accepts_oft_override():
    cfg = _compose_config('sft_trainer', ['model.oft_block_size=16'])

    assert cfg.model.oft_block_size == 16


def test_sft_trainer_accepts_keeplora_override():
    cfg = _compose_config('sft_trainer', ['model.lora_variant=keeplora', 'model.keeplora_principal_rank=32'])

    assert cfg.model.lora_variant == 'keeplora'
    assert cfg.model.keeplora_principal_rank == 32


def test_ppo_trainer_accepts_actor_and_critic_oft_overrides():
    cfg = _compose_config(
        'ppo_trainer',
        [
            'actor_rollout_ref.model.oft_block_size=16',
            'critic.model.oft_block_size=16',
        ],
    )

    assert cfg.actor_rollout_ref.model.oft_block_size == 16
    assert cfg.critic.model.oft_block_size == 16


def test_ppo_trainer_defaults_match_rl_runtime_expectations():
    cfg = _compose_config('ppo_trainer', [])

    assert cfg.data.max_prompt_length == 1024
    assert cfg.data.max_response_length == 8192
    assert cfg.actor_rollout_ref.actor.fsdp_config.model_dtype == 'bf16'
    assert cfg.actor_rollout_ref.ref.fsdp_config.model_dtype == 'bf16'
    assert cfg.critic.model.fsdp_config.model_dtype == 'bf16'
