"""KeepLoRA helpers for SFT-only LoRA initialization.

This implements a minimal adaptation of the KeepLoRA idea for the current
causal-LM SFT pipeline:

1. Use a single first-step gradient estimate from the current task.
2. Remove the component aligned with the principal subspace of the pretrained
   weight matrix.
3. Initialize LoRA with a low-rank factorization of the residual gradient.
4. Freeze the down-projection matrix and only optimize the up-projection.

The current implementation is intentionally isolated from existing LoRA/OFT
paths and only activates when ``lora_variant=keeplora``.
"""

from __future__ import annotations

import torch


def resolve_keeplora_principal_rank(config_rank: int, lora_rank: int) -> int:
    if config_rank > 0:
        return config_rank
    return max(lora_rank * 4, lora_rank)


def _safe_lowrank_q(tensor: torch.Tensor, rank: int) -> int:
    if tensor.ndim != 2:
        raise ValueError(f"KeepLoRA expects 2D tensors, got shape={tuple(tensor.shape)}")
    return max(1, min(rank, min(tensor.shape)))


def _compute_principal_basis(weight: torch.Tensor, principal_rank: int) -> torch.Tensor | None:
    if principal_rank <= 0:
        return None

    q = _safe_lowrank_q(weight, principal_rank)
    _, _, right = torch.svd_lowrank(weight.float(), q=q)
    return right


def project_gradient_to_residual_subspace(
    weight: torch.Tensor,
    grad: torch.Tensor,
    principal_rank: int,
) -> torch.Tensor:
    basis = _compute_principal_basis(weight, principal_rank)
    grad_f32 = grad.float()
    if basis is None:
        return grad_f32
    return grad_f32 - (grad_f32 @ basis) @ basis.transpose(0, 1)


def build_keeplora_factors(
    weight: torch.Tensor,
    grad: torch.Tensor,
    lora_rank: int,
    principal_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_grad = project_gradient_to_residual_subspace(weight, grad, principal_rank)
    q = _safe_lowrank_q(residual_grad, lora_rank)
    left, singular_values, right = torch.svd_lowrank(residual_grad, q=q)

    lora_b = left @ torch.diag(singular_values)
    lora_a = right.transpose(0, 1)

    if q < lora_rank:
        out_features, in_features = residual_grad.shape
        padded_a = residual_grad.new_zeros((lora_rank, in_features))
        padded_b = residual_grad.new_zeros((out_features, lora_rank))
        padded_a[:q, :] = lora_a
        padded_b[:, :q] = lora_b
        return padded_a, padded_b

    return lora_a, lora_b


def freeze_keeplora_down_projection(model, adapter_name: str = "default") -> int:
    frozen = 0
    for module in model.modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is None or adapter_name not in lora_a:
            continue
        lora_a[adapter_name].weight.requires_grad = False
        frozen += 1
    return frozen


@torch.no_grad()
def initialize_keeplora_adapters(
    model,
    principal_rank: int,
    grad_map: dict[str, torch.Tensor] | None = None,
    adapter_name: str = "default",
) -> int:
    initialized = 0

    for module_name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        base_layer = getattr(module, "base_layer", None)
        if lora_a is None or lora_b is None or base_layer is None:
            continue
        if adapter_name not in lora_a or adapter_name not in lora_b:
            continue

        base_weight = getattr(base_layer, "weight", None)
        base_grad = grad_map.get(module_name) if grad_map is not None else getattr(base_weight, "grad", None)
        if base_weight is None or base_grad is None or base_weight.ndim != 2:
            continue

        lora_a_weight = lora_a[adapter_name].weight
        lora_b_weight = lora_b[adapter_name].weight

        adapter_rank = lora_a_weight.shape[0]
        init_a, init_b = build_keeplora_factors(
            weight=base_weight.detach(),
            grad=base_grad.detach(),
            lora_rank=adapter_rank,
            principal_rank=principal_rank,
        )

        lora_a_weight.copy_(init_a.to(device=lora_a_weight.device, dtype=lora_a_weight.dtype))
        lora_b_weight.copy_(init_b.to(device=lora_b_weight.device, dtype=lora_b_weight.dtype))
        initialized += 1

    return initialized
