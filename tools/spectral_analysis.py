#!/usr/bin/env python3
"""
Spectral Analysis of PEFT Checkpoint Updates
=============================================
Implements the Dual-View Spectral Analysis from the paper:
  - View I:  Diagonal projection on pre-trained basis (Retention)
  - View II: Projected update energy spectrum (Adaptation)
  - Effective rank of ΔW
  - Spectral smoothness (fluctuation score)
  - Singular Vector Alignment (SVA) for OFT adapters

All SVD computations use torch.linalg.svd in float32 on CUDA.

Usage:
    python tools/spectral_analysis.py \
        --base_model /path/to/pretrained \
        --finetuned_model /path/to/finetuned_or_adapter \
        --output_dir /path/to/output \
        [--layers 0,1,5,10,15,20,25,31] \
        [--modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] \
        [--smoothness_window 5]
"""

import argparse
import json
import os
import sys
import gc
from collections import OrderedDict

import torch
import numpy as np


# ============================================================================
# SVD helper — always float32, CUDA
# ============================================================================

def safe_svd(W: torch.Tensor, full_matrices: bool = False):
    """Compute SVD using torch.linalg.svd in float32 on CUDA."""
    W_f32 = W.float()
    U, S, Vh = torch.linalg.svd(W_f32, full_matrices=full_matrices)
    return U.cpu(), S.cpu(), Vh.cpu()


# ============================================================================
# Metric computation functions
# ============================================================================

def compute_diagonal_projection(W_pre: torch.Tensor, W_ft: torch.Tensor,
                                U_pre: torch.Tensor, S_pre: torch.Tensor,
                                Vh_pre: torch.Tensor):
    """
    View I: Diagonal projection on the pre-trained basis (Retention).

    P_diag(i) = u_i^T @ W_ft @ v_i
    ΔP_diag(i) = |P_diag_ft(i) - σ_pre(i)|

    Returns:
        delta_P_diag: tensor of shape (min(m,n),) — |ΔP_diag(i)|
        P_diag_ft:    tensor of shape (min(m,n),) — P_diag(i) for W_ft
    """
    # V_pre = Vh_pre.T, columns are right singular vectors
    # P_diag_ft(i) = U_pre[:, i]^T @ W_ft @ V_pre[:, i]
    # Efficient: diag(U_pre^T @ W_ft @ V_pre^T^T) = diag(U_pre^T @ W_ft @ Vh_pre^T)
    # But Vh_pre rows are v_i^T, so W_ft @ Vh_pre^T has columns W_ft @ v_i
    # Then U_pre^T @ (W_ft @ Vh_pre^T) is a matrix, diagonal is what we want

    W_ft_f32 = W_ft.float()
    # shape: (m, k) where k = min(m, n)
    proj = U_pre.T @ W_ft_f32 @ Vh_pre.T  # (k, k)
    P_diag_ft = torch.diag(proj)  # (k,)
    delta_P_diag = torch.abs(P_diag_ft - S_pre)  # (k,)

    return delta_P_diag.cpu(), P_diag_ft.cpu()


def compute_update_energy(W_pre: torch.Tensor, W_ft: torch.Tensor,
                          Vh_pre: torch.Tensor):
    """
    View II: Update energy spectrum (Adaptation).

    E_Δ(i) = ||ΔW @ v_i||_2

    Returns:
        E_delta: tensor of shape (min(m,n),) — E_Δ(i)
    """
    W_pre_f32 = W_pre.float()
    W_ft_f32 = W_ft.float()
    delta_W = W_ft_f32 - W_pre_f32  # (m, n)

    # ΔW @ v_i = ΔW @ Vh_pre^T columns
    # delta_W @ Vh_pre.T has shape (m, k), each column is ΔW @ v_i
    projected = delta_W @ Vh_pre.T  # (m, k)
    E_delta = torch.norm(projected, dim=0)  # (k,)

    return E_delta.cpu()


def compute_effective_rank(W_pre: torch.Tensor, W_ft: torch.Tensor):
    """
    Effective rank of ΔW via energy ratio (consistent with
    peft/analysis_delta_weight_norm.py):

        erank = (Σ σ_i²) / σ_1²  =  ‖ΔW‖_F² / ‖ΔW‖_spectral²

    Equals 1 for a rank-1 matrix; equals min(m,n) when all singular
    values are equal.

    Returns:
        erank: float — effective rank
        singular_values: tensor — singular values of ΔW
    """
    delta_W = (W_ft.float().cuda()) - (W_pre.float().cuda())
    _, S, _ = torch.linalg.svd(delta_W, full_matrices=False)

    if S[0] > 0:
        total_energy = (S ** 2).sum()
        erank = (total_energy / (S[0] ** 2)).item()
    else:
        erank = 0.0

    return erank, S.cpu()


def compute_spectral_smoothness(values: torch.Tensor, window: int = 5):
    """
    Spectral smoothness: mean absolute deviation from moving average.

    A higher score means more "spiky" (incoherent); lower means smoother.

    Args:
        values: 1D tensor of spectral values (|ΔP_diag| or E_Δ)
        window: moving average window size

    Returns:
        fluctuation_score: float
        deviations: tensor — pointwise deviations from moving average
    """
    if len(values) < window:
        return 0.0, torch.zeros_like(values)

    values_f = values.float()
    # Compute moving average using 1D convolution
    kernel = torch.ones(window, dtype=torch.float32) / window
    # Pad symmetrically
    pad = window // 2
    padded = torch.nn.functional.pad(values_f.unsqueeze(0).unsqueeze(0),
                                     (pad, pad), mode='reflect')
    ma = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1)).squeeze()

    deviations = torch.abs(values_f - ma)
    fluctuation_score = deviations.mean().item()

    return fluctuation_score, deviations


def compute_sva(Vh_pre: torch.Tensor, Vh_ft: torch.Tensor):
    """
    Singular Vector Alignment (SVA) for OFT adapters.

    Cosine similarity between pre-trained and fine-tuned right singular vectors
    at each index: cos(v_pre_i, v_ft_i)

    Returns:
        cosine_sims: tensor of shape (k,) — cosine similarities
    """
    k = min(Vh_pre.shape[0], Vh_ft.shape[0])
    # Vh rows are v_i^T, so dot product of rows
    Vh_pre_f32 = Vh_pre[:k].float().cuda()
    Vh_ft_f32 = Vh_ft[:k].float().cuda()

    # Normalize rows
    Vh_pre_norm = Vh_pre_f32 / (Vh_pre_f32.norm(dim=1, keepdim=True) + 1e-10)
    Vh_ft_norm = Vh_ft_f32 / (Vh_ft_f32.norm(dim=1, keepdim=True) + 1e-10)

    cosine_sims = (Vh_pre_norm * Vh_ft_norm).sum(dim=1)  # (k,)
    return cosine_sims.abs().cpu()


# ============================================================================
# Model loading utilities
# ============================================================================

def is_adapter_checkpoint(path: str) -> bool:
    """Check if a path contains a PEFT adapter checkpoint."""
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    return "adapter_model.safetensors" in files or "adapter_model.bin" in files


def load_models(base_model_path: str, finetuned_path: str):
    """
    Load pre-trained and fine-tuned models. Returns (model_pre, model_ft, peft_type).

    If finetuned_path is an adapter, loads via PEFT and merges.
    If finetuned_path is a full model, loads directly.
    """
    from transformers import AutoModelForCausalLM

    print(f"[spectral] Loading base model from: {base_model_path}")
    model_pre = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    peft_type = None

    if is_adapter_checkpoint(finetuned_path):
        from peft import PeftModel, PeftConfig
        peft_config = PeftConfig.from_pretrained(finetuned_path)
        peft_type = str(peft_config.peft_type)
        print(f"[spectral] Detected adapter type: {peft_type}")
        print(f"[spectral] Loading adapter from: {finetuned_path}")

        # Load a separate base model for the adapter
        model_ft_base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(
            model_ft_base, finetuned_path
        )
        model_ft = peft_model.merge_and_unload()
        del peft_model
    else:
        print(f"[spectral] Loading full fine-tuned model from: {finetuned_path}")
        model_ft = AutoModelForCausalLM.from_pretrained(
            finetuned_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    return model_pre, model_ft, peft_type


def get_linear_layer_pairs(model_pre, model_ft, layers=None, modules=None):
    """
    Extract matching (W_pre, W_ft, layer_name) tuples for linear layers.

    Args:
        layers: list of layer indices to analyze (None = all)
        modules: list of module name suffixes to include (None = all linear)

    Yields:
        (layer_name, W_pre, W_ft) tuples
    """
    pre_state = model_pre.state_dict()
    ft_state = model_ft.state_dict()

    for name in sorted(pre_state.keys()):
        if name not in ft_state:
            continue
        W_pre = pre_state[name]
        W_ft = ft_state[name]

        # Skip non-2D (biases, layernorms, embeddings)
        if W_pre.ndim != 2:
            continue

        # Filter by layer index if specified
        if layers is not None:
            # Extract layer number from name like "model.layers.5.self_attn.q_proj.weight"
            import re
            match = re.search(r'layers\.(\d+)\.', name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in layers:
                    continue
            else:
                continue  # skip non-layer weights when layer filter is active

        # Filter by module name
        if modules is not None:
            matched = any(m in name for m in modules)
            if not matched:
                continue

        yield name, W_pre, W_ft


# ============================================================================
# Main analysis pipeline
# ============================================================================

def analyze_checkpoint(base_model_path: str, finetuned_path: str,
                       output_dir: str, layers=None, modules=None,
                       smoothness_window: int = 5):
    """Run the full spectral analysis pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    # Load models
    model_pre, model_ft, peft_type = load_models(base_model_path, finetuned_path)
    is_oft = peft_type is not None and "OFT" in peft_type.upper()

    results = OrderedDict()
    all_eranks = {}
    all_smoothness_retention = {}
    all_smoothness_adaptation = {}

    layer_count = 0
    for name, W_pre, W_ft in get_linear_layer_pairs(model_pre, model_ft, layers, modules):
        # if name == 'model.layers.18.mlp.down_proj.weight':
        #     import pdb; pdb.set_trace()
        layer_count += 1
        print(f"\n[spectral] Analyzing: {name}")

        # 6. SVA (OFT only)
        sva_cosines = None
        if is_oft:
            U_ft, S_ft, Vh_ft = safe_svd(W_ft, full_matrices=False)
            U_pre, S_pre, Vh_pre = safe_svd(W_pre, full_matrices=False)
            sva_cosines = compute_sva(Vh_pre, Vh_ft)
        else:
            U_pre, S_pre, Vh_pre = safe_svd(W_pre.cuda(), full_matrices=False)

        # 2. Diagonal projection (Retention)
        delta_P_diag, P_diag_ft = compute_diagonal_projection(
            W_pre, W_ft, U_pre, S_pre, Vh_pre
        )

        # 3. Update energy (Adaptation)
        E_delta = compute_update_energy(W_pre, W_ft, Vh_pre)

        # 4. Effective rank
        erank, delta_sv = compute_effective_rank(W_pre, W_ft)

        # 5. Spectral smoothness
        fluct_retention, dev_retention = compute_spectral_smoothness(
            delta_P_diag, smoothness_window
        )
        fluct_adaptation, dev_adaptation = compute_spectral_smoothness(
            E_delta, smoothness_window
        )


        # Store results
        layer_results = {
            "effective_rank": erank,
            "smoothness_retention": fluct_retention,
            "smoothness_adaptation": fluct_adaptation,
            "delta_P_diag_mean": delta_P_diag.mean().item(),
            "delta_P_diag_max": delta_P_diag.max().item(),
            "E_delta_mean": E_delta.mean().item(),
            "E_delta_max": E_delta.max().item(),
            "delta_sv_top5": delta_sv[:5].tolist(),
        }
        if sva_cosines is not None:
            layer_results["sva_mean"] = sva_cosines.mean().item()
            layer_results["sva_min"] = sva_cosines.min().item()

        results[name] = layer_results
        all_eranks[name] = erank
        all_smoothness_retention[name] = fluct_retention
        all_smoothness_adaptation[name] = fluct_adaptation

        # Save per-layer tensor data
        layer_data = {
            "delta_P_diag": delta_P_diag,
            "P_diag_ft": P_diag_ft,
            "E_delta": E_delta,
            "delta_sv": delta_sv,
            "S_pre": S_pre.cpu(),
            "dev_retention": dev_retention,
            "dev_adaptation": dev_adaptation,
        }
        if sva_cosines is not None:
            layer_data["sva_cosines"] = sva_cosines

        safe_name = name.replace(".", "_").replace("/", "_")
        torch.save(layer_data, os.path.join(output_dir, f"{safe_name}.pt"))

        # Cleanup GPU memory
        del U_pre, S_pre, Vh_pre
        torch.cuda.empty_cache()

        print(f"  erank={erank:.2f}  fluct_ret={fluct_retention:.4e}  "
              f"fluct_adap={fluct_adaptation:.4e}")

    # Save summary JSON
    summary = {
        "base_model": base_model_path,
        "finetuned_model": finetuned_path,
        "peft_type": peft_type,
        "smoothness_window": smoothness_window,
        "num_layers_analyzed": layer_count,
        "per_layer": results,
    }

    summary_path = os.path.join(output_dir, "spectral_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[spectral] Summary saved to: {summary_path}")
    print(f"[spectral] Per-layer tensors saved to: {output_dir}")

    # Cleanup
    del model_pre, model_ft
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Spectral analysis of PEFT checkpoint updates"
    )
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to the pre-trained base model")
    parser.add_argument("--finetuned_model", type=str, required=True,
                        help="Path to fine-tuned model or PEFT adapter checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save analysis results")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to analyze (default: all)")
    parser.add_argument("--modules", type=str, default=None,
                        help="Comma-separated module name suffixes (default: all linear)")
    parser.add_argument("--smoothness_window", type=int, default=5,
                        help="Window size for spectral smoothness moving average")
    return parser.parse_args()


def main():
    args = parse_args()

    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    modules = None
    if args.modules:
        modules = [x.strip() for x in args.modules.split(",")]

    analyze_checkpoint(
        base_model_path=args.base_model,
        finetuned_path=args.finetuned_model,
        output_dir=args.output_dir,
        layers=layers,
        modules=modules,
        smoothness_window=args.smoothness_window,
    )


if __name__ == "__main__":
    main()
