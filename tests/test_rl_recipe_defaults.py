import sys
from pathlib import Path


ARENA_DIR = Path(__file__).resolve().parents[1]
if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))

import run


def test_run_py_uses_rl_specific_default_learning_rates():
    assert run.resolve_train_lr("rl", "full", None) == 1e-6
    assert run.resolve_train_lr("rl", "lora", None) == 1e-5
    assert run.resolve_train_lr("rl", "oft", None) == 1e-5
    assert run.resolve_train_lr("sft", "lora", None) == 2e-4
    assert run.resolve_train_lr("rl", "full", 3e-6) == 3e-6


def test_run_py_uses_rl_specific_default_epochs():
    assert run.resolve_train_epochs("rl", None) == 10
    assert run.resolve_train_epochs("sft", None) == 4
    assert run.resolve_train_epochs("rl", 6) == 6
