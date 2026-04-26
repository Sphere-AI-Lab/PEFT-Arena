import sys
from pathlib import Path

import torch


TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))


class _MockLoraLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(4, 3, bias=False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 3, bias=False)})


class _MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _MockLoraLayer()


def test_project_gradient_to_residual_subspace_removes_principal_direction():
    from peft_arena_verl.utils.keeplora import project_gradient_to_residual_subspace

    torch.manual_seed(0)
    weight = torch.diag(torch.tensor([100.0, 1.0, 0.5]))
    grad = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    projected = project_gradient_to_residual_subspace(weight, grad, principal_rank=1)

    assert float(projected.abs().max()) < 1e-6


def test_initialize_keeplora_adapters_from_grad_map():
    from peft_arena_verl.utils.keeplora import initialize_keeplora_adapters

    model = _MockModel()
    grad_map = {"layer": torch.randn_like(model.layer.base_layer.weight)}

    initialized = initialize_keeplora_adapters(model, principal_rank=1, grad_map=grad_map)

    assert initialized == 1
    assert model.layer.lora_A["default"].weight.shape == (2, 4)
    assert model.layer.lora_B["default"].weight.shape == (3, 2)
    assert torch.isfinite(model.layer.lora_A["default"].weight).all()
    assert torch.isfinite(model.layer.lora_B["default"].weight).all()


def test_freeze_keeplora_down_projection_only():
    from peft_arena_verl.utils.keeplora import freeze_keeplora_down_projection

    model = _MockModel()

    frozen = freeze_keeplora_down_projection(model)

    assert frozen == 1
    assert model.layer.lora_A["default"].weight.requires_grad is False
    assert model.layer.lora_B["default"].weight.requires_grad is True
