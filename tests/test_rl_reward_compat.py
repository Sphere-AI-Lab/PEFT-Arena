from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))


def _import_rl_modules():
    try:
        from peft_arena_verl.reward_score.shared_reward import compute_score
        from peft_arena_verl.reward_score import math_rlvr
        from peft_arena_verl.trainer.main_ppo import _configure_local_prime_reward_impl
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"RL modules are unavailable in this environment: {exc}")
    return compute_score, math_rlvr, _configure_local_prime_reward_impl


def test_shared_reward_supports_openr1_and_medthink():
    compute_score, _, _ = _import_rl_modules()

    openr1_score = compute_score(
        data_source="openr1",
        solution_str=r"We simplify the expression and get \boxed{756}.",
        ground_truth="756",
    )
    med_score = compute_score(
        data_source="med_23k_think",
        solution_str=r"After reviewing the paper, the best option is \boxed{C}.",
        ground_truth="C",
    )

    assert openr1_score["score"] == 1.0
    assert openr1_score["pred"] == "756"
    assert med_score["score"] == 1.0
    assert med_score["pred"] == "C"


def test_prime_reward_manager_is_redirected_to_local_importlib_impl():
    _, _, _configure_local_prime_reward_impl = _import_rl_modules()

    try:
        from omegaconf import OmegaConf
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"omegaconf is unavailable in this environment: {exc}")

    cfg = OmegaConf.create(
        {
            "reward_model": {
                "reward_manager": "prime",
                "reward_loop_source": None,
                "reward_loop_module_path": None,
                "reward_loop_class_name": None,
            },
            "custom_reward_function": {"path": None, "name": None},
            "reward": {
                "reward_manager": {"name": "custom_reward_manager", "source": "register", "module": {"path": None, "name": None}},
                "custom_reward_function": {"path": None, "name": None},
            },
        }
    )

    cfg = _configure_local_prime_reward_impl(cfg)

    assert cfg.reward_model.reward_loop_source == "importlib"
    assert cfg.reward_model.reward_loop_module_path.endswith("prime_reward_manager.py")
    assert cfg.custom_reward_function.path.endswith("shared_reward.py")
    assert cfg.custom_reward_function.name == "compute_score"


def test_call_with_timeout_ignores_process_lookup_error(monkeypatch):
    _, math_rlvr, _ = _import_rl_modules()

    class FakeQueue:
        def empty(self):
            return True

    class FakeProcess:
        pid = 12345

        def __init__(self, *args, **kwargs):
            self.daemon = False

        def start(self):
            return None

        def is_alive(self):
            return True

        def terminate(self):
            return None

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(math_rlvr.multiprocessing, "Queue", lambda: FakeQueue())
    monkeypatch.setattr(math_rlvr.multiprocessing, "Process", FakeProcess)

    def raise_process_lookup_error(pid, sig):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(math_rlvr.os, "kill", raise_process_lookup_error)

    assert math_rlvr.call_with_timeout(lambda output_queue: None, timeout=0) is False
