#!/usr/bin/env python3
"""
Visualization for Spectral Analysis of PEFT Checkpoints
========================================================
Reads output from spectral_analysis.py and generates publication-quality plots.

Plots:
  1. Effective rank bar chart across layers
  2. Projected spectrum change distributions (|ΔP_diag| and E_Δ)
  3. Per-layer spectral curves with moving-average overlay
  4. Spectral smoothness (fluctuation) summary
  5. SVA curves for OFT checkpoints

Usage:
    python tools/plot_spectral_analysis.py \
        --input_dirs /path/to/results_method1 /path/to/results_method2 \
        --labels "LoRA" "OFT" \
        --output_dir /path/to/figures \
        [--plot_type all]
        [--layer_names "model.layers.0.self_attn.q_proj.weight"]
        [--dpi 300]
"""

import argparse
import json
import os
import sys
import glob
from collections import OrderedDict

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================================
# Style configuration  (aligned with peft/visualize_analysis_results.py)
# ============================================================================

# Colorblind-safe palette — Wong (2011, Nature Methods)
# Distinguishable under protanopia, deuteranopia, and tritanopia
COLORS = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#009E73",  # Bluish Green
    "#CC79A7",  # Reddish Purple
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#F0E442",  # Yellow
    "#000000",  # Black
]

# Extra visual distinctors for multi-method comparisons
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']


def _label_style(label: str):
    """Return (linestyle, hatch) based on whether the label contains SFT or RL.

    SFT  → solid line ('-'),  no hatch ('')
    RL   → dashed line ('--'), diagonal hatch ('///')
    other → solid, no hatch
    """
    upper = label.upper()
    if "RL" in upper:
        return "--", "///"
    return "-", ""


def setup_style():
    """Set up matplotlib style (aligned with peft/visualize_analysis_results.py)."""
    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        # Bold black borders (spines)
        "axes.linewidth": 1.5,
        "axes.edgecolor": "black",
    })


# ============================================================================
# Data loading
# ============================================================================

def load_analysis_results(input_dir: str):
    """Load spectral analysis results from a directory.

    Returns:
        summary: dict from spectral_summary.json
        layer_data: dict of {layer_name: tensor_dict} from .pt files
    """
    summary_path = os.path.join(input_dir, "spectral_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No spectral_summary.json found in {input_dir}")

    with open(summary_path) as f:
        summary = json.load(f)

    # Load per-layer tensor data
    layer_data = {}
    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    for pt_file in pt_files:
        # Reconstruct layer name from filename
        basename = os.path.splitext(os.path.basename(pt_file))[0]
        data = torch.load(pt_file, map_location="cpu", weights_only=True)
        layer_data[basename] = data

    return summary, layer_data


def shorten_layer_name(name: str) -> str:
    """Shorten a layer name for display."""
    name = name.replace("model_layers_", "L").replace("_weight", "")
    name = name.replace("self_attn_", "").replace("mlp_", "")
    return name


# ============================================================================
# Plot 1: Effective rank bar chart
# ============================================================================

def plot_effective_rank(all_summaries, labels, output_dir, dpi=300):
    """Bar chart comparing effective rank across layers and methods."""
    fig, ax = plt.subplots(figsize=(14, 5))

    n_methods = len(all_summaries)

    # Collect all layer names (union)
    all_layers = []
    for summary in all_summaries:
        for layer_name in summary["per_layer"]:
            if layer_name not in all_layers:
                all_layers.append(layer_name)

    short_names = [shorten_layer_name(n.replace(".", "_")) for n in all_layers]

    x = np.arange(len(all_layers))
    width = 0.8 / n_methods

    for i, (summary, label) in enumerate(zip(all_summaries, labels)):
        eranks = []
        for layer_name in all_layers:
            if layer_name in summary["per_layer"]:
                eranks.append(summary["per_layer"][layer_name]["effective_rank"])
            else:
                eranks.append(0)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, eranks, width, label=label, color=COLORS[i % len(COLORS)],
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank of Weight Updates ($\Delta W$)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="upper right")
    ax.set_xlim(-0.5, len(all_layers) - 0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "effective_rank.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {path}")


def plot_effective_rank_curves(all_summaries, labels, output_dir, dpi=300,
                               color_indices=None):
    """Line plot of effective rank vs layer index, one line per method.

    Groups layers by module type (e.g. q_proj, k_proj, ...) and produces
    one plot per module type, analogous to the overall comparison in
    peft/visualize_analysis_results.py.
    """
    import re

    n_methods = len(labels)
    if color_indices is None:
        color_indices = list(range(n_methods))

    # Collect per-method, per-module-type data: {module_type: {layer_idx: erank}}
    per_method_data = []  # list of {module_type: {layer_idx: erank}}
    all_module_types = set()
    for summary in all_summaries:
        method_data = {}  # module_type -> {layer_idx: erank}
        for layer_name, layer_info in summary["per_layer"].items():
            # e.g. "model.layers.5.self_attn.q_proj.weight"
            match = re.search(r'layers\.(\d+)\.(.+?)(?:\.weight)?$', layer_name)
            if match:
                layer_idx = int(match.group(1))
                module_type = match.group(2)
            else:
                # Fallback: use full name, index = insertion order
                layer_idx = len(method_data)
                module_type = "all"
            all_module_types.add(module_type)
            if module_type not in method_data:
                method_data[module_type] = {}
            method_data[module_type][layer_idx] = layer_info["effective_rank"]
        per_method_data.append(method_data)

    for module_type in sorted(all_module_types):
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.set_dpi(dpi)

        for i, (method_data, label) in enumerate(zip(per_method_data, labels)):
            if module_type not in method_data:
                continue
            layer_dict = method_data[module_type]
            indices = sorted(layer_dict.keys())
            eranks = [layer_dict[idx] for idx in indices]

            cidx = color_indices[i] % len(COLORS)
            midx = i % len(MARKERS)
            ls, _ = _label_style(label)
            ax.plot(indices, eranks,
                    color=COLORS[cidx],
                    linestyle=ls, alpha=0.7, linewidth=1.5,
                    marker=MARKERS[midx], markersize=5,
                    markevery=max(1, len(indices) // 10),
                    label=label)

        ax.set_xlabel(r"Layer Index", fontsize=12)
        ax.set_ylabel(r"Effective Rank", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, framealpha=0.3)

        plt.tight_layout()
        safe_module = module_type.replace(".", "_").replace("/", "_")
        path = os.path.join(output_dir, f"effective_rank_curve_{safe_module}.pdf")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved: {path}")


# ============================================================================
# Plot 2: Spectrum change distributions (violin/box plots)
# ============================================================================

def plot_spectrum_distributions(all_layer_data, labels, output_dir, dpi=300, log_scale=False):
    """Violin plots of |ΔP_diag| and E_Δ distributions across methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric_key, title) in enumerate([
        ("delta_P_diag", "$|\Delta P_diag|$ (Retention)"),
        ("E_delta", "$E_\Delta$ (Adaptation)"),
    ]):
        ax = axes[ax_idx]
        data_for_violin = []
        positions = []

        for i, (layer_data, label) in enumerate(zip(all_layer_data, labels)):
            # Aggregate across all layers
            all_vals = []
            for lname, ldata in layer_data.items():
                if metric_key in ldata:
                    vals = ldata[metric_key].numpy()
                    all_vals.extend(vals.tolist())
            if all_vals:
                data_for_violin.append(all_vals)
                positions.append(i + 1)

        if data_for_violin:
            parts = ax.violinplot(data_for_violin, positions=positions,
                                  showmeans=True, showmedians=True)
            for j, pc in enumerate(parts.get("bodies", [])):
                pc.set_facecolor(COLORS[j % len(COLORS)])
                pc.set_alpha(0.6)

        ax.set_xticks(list(range(1, len(labels) + 1)))
        ax.set_xticklabels(labels)
        ax.set_title(title)
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel("Value")

    plt.tight_layout()
    path = os.path.join(output_dir, "spectrum_distributions.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {path}")


# ============================================================================
# Plot 3: Per-layer spectral curves  (retention & adaptation profiling)
# ============================================================================

def plot_spectral_curves(all_layer_data, labels, output_dir,
                         layer_names=None, dpi=300, log_scale=False,
                         color_indices=None):
    """Plot |ΔP_diag(i)| and E_Δ(i) vs singular index i.

    Style:
      - Pretrained SV: black dashed line (k--)
      - Method curves: colored semi-transparent solid lines, NO markers
      - color_indices controls which COLORS entry each method uses
        (default 0,1,2,… ; e.g. [3,2,1,0] reverses the palette)
      - log scale on y-axis when --log_scale is set
    """
    # Find common layers
    common_layers = None
    for layer_data in all_layer_data:
        keys = set(layer_data.keys())
        if common_layers is None:
            common_layers = keys
        else:
            common_layers = common_layers.intersection(keys)

    if layer_names is not None:
        # Filter to requested layers
        requested = set(n.replace(".", "_").replace("/", "_") for n in layer_names)
        common_layers = common_layers.intersection(requested)

    common_layers = sorted(common_layers)
    if not common_layers:
        print("[plot] Warning: no common layers for spectral curves")
        return

    # Resolve color indices — default 0,1,2,...
    n_methods = len(labels)
    if color_indices is None:
        color_indices = list(range(n_methods))
    assert len(color_indices) == n_methods, \
        f"color_indices length ({len(color_indices)}) must match number of methods ({n_methods})"

    for layer_key in common_layers:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.set_dpi(dpi)
        short_name = shorten_layer_name(layer_key)

        for ax_idx, (metric_key, ylabel_raw, title_suffix) in enumerate([
            ("delta_P_diag",
             r"Normalized $|\Delta\mathcal{P}_{\mathrm{diag}}(i)|$",
             "Retention Profiling"),
            ("E_delta",
             r"Normalized $\mathcal{E}_{\Delta}(i)$",
             "Adaptation Profiling"),
        ]):
            ax = axes[ax_idx]
            pretrained_plotted = False

            for i, (layer_data, label) in enumerate(zip(all_layer_data, labels)):
                if layer_key not in layer_data:
                    continue
                vals = layer_data[layer_key].get(metric_key)
                if vals is None:
                    continue

                # --- Normalization by pretrained spectrum sum (S_pre.sum()) ---
                S_pre = layer_data[layer_key].get("S_pre")
                if S_pre is not None and len(S_pre) > 0:
                    if metric_key == 'delta_P_diag':
                        norm_factor = float(S_pre[0]) if float(S_pre[0]) > 0 else 1.0
                    else:
                        norm_factor = float(S_pre.sum()) if float(S_pre.sum()) > 0 else 1.0
                else:
                    norm_factor = 1.0

                # Plot pretrained SV reference curve once per subplot
                if not pretrained_plotted and S_pre is not None:
                    S_pre_np = S_pre.numpy().copy()
                    S_pre_norm = S_pre_np / norm_factor
                    if log_scale:
                        S_pre_norm = np.where(S_pre_norm > 0, S_pre_norm, 1e-10)
                    ax.plot(np.arange(len(S_pre_norm)), S_pre_norm,
                            'k--', linewidth=2, alpha=0.8,
                            label='Pretrained SV')
                    pretrained_plotted = True

                if metric_key == "E_delta":
                    vals = vals / vals.sum()
                vals_np = vals.numpy().copy()
                indices = np.arange(len(vals_np))

                # Floor very small / zero values for log-scale safety
                if log_scale:
                    vals_np = np.where(vals_np > 0, vals_np, 1e-10)

                cidx = color_indices[i] % len(COLORS)
                ax.plot(indices, vals_np,
                        color=COLORS[cidx],
                        linestyle='-',
                        alpha=0.6, linewidth=1.5, label=label)

            ylabel = ylabel_raw
            if log_scale:
                ax.set_yscale("log")
                ylabel = ylabel_raw + r" (log scale)"
            ax.set_xlabel(r"Singular Value Index $i$", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12, ncol=max(1, (n_methods + 1) // 3),
                      loc="lower right" if ax_idx == 1 else "upper right",
                      framealpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f"spectral_curve_{layer_key}.pdf")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved: {path}")


# ============================================================================
# Plot 4: Spectral smoothness summary
# ============================================================================

def plot_smoothness_summary(all_summaries, labels, output_dir, dpi=300, log_scale=False):
    """Grouped bar chart of fluctuation scores across layers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric_key, title) in enumerate([
        ("smoothness_retention", "Fluctuation Score — Retention"),
        ("smoothness_adaptation", "Fluctuation Score — Adaptation"),
    ]):
        ax = axes[ax_idx]
        n_methods = len(all_summaries)

        all_layers = []
        for summary in all_summaries:
            for layer_name in summary["per_layer"]:
                if layer_name not in all_layers:
                    all_layers.append(layer_name)

        short_names = [shorten_layer_name(n.replace(".", "_")) for n in all_layers]

        x = np.arange(len(all_layers))
        width = 0.8 / n_methods

        for i, (summary, label) in enumerate(zip(all_summaries, labels)):
            scores = []
            for layer_name in all_layers:
                if layer_name in summary["per_layer"]:
                    scores.append(summary["per_layer"][layer_name][metric_key])
                else:
                    scores.append(0)
            offset = (i - n_methods / 2 + 0.5) * width
            ax.bar(x + offset, scores, width, label=label,
                   color=COLORS[i % len(COLORS)], alpha=0.85,
                   edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Fluctuation Score")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(output_dir, "smoothness_summary.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {path}")


def plot_smoothness_per_layer(all_summaries, labels, output_dir, dpi=300,
                              log_scale=False, layer_names=None,
                              color_indices=None):
    """Per-layer bar chart: for each requested layer, show a grouped bar
    chart with one bar per method for retention and adaptation smoothness.

    If layer_names is None, plots all layers found in the summaries.
    """
    n_methods = len(labels)
    if color_indices is None:
        color_indices = list(range(n_methods))

    # Collect all layer names (union across methods)
    all_layers = []
    for summary in all_summaries:
        for layer_name in summary["per_layer"]:
            if layer_name not in all_layers:
                all_layers.append(layer_name)

    # Filter to requested layers
    if layer_names is not None:
        requested = set(n.replace(".", "_").replace("/", "_") for n in layer_names)
        all_layers = [l for l in all_layers
                      if l.replace(".", "_").replace("/", "_") in requested
                      or l in layer_names]

    if not all_layers:
        print("[plot] Warning: no matching layers for per-layer smoothness")
        return

    smoothness_dir = os.path.join(output_dir, "smoothness_per_layer")
    os.makedirs(smoothness_dir, exist_ok=True)

    for layer_name in all_layers:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.set_dpi(dpi)

        # Gather scores for both metrics
        ret_scores = []
        adap_scores = []
        bar_colors = []
        for i, (summary, label) in enumerate(zip(all_summaries, labels)):
            if layer_name in summary["per_layer"]:
                ret_scores.append(summary["per_layer"][layer_name]["smoothness_retention"])
                adap_scores.append(summary["per_layer"][layer_name]["smoothness_adaptation"])
            else:
                ret_scores.append(0)
                adap_scores.append(0)
            cidx = color_indices[i] % len(COLORS)
            bar_colors.append(COLORS[cidx])

        # Two groups: Retention (x=0) and Adaptation (x=1)
        n = n_methods
        width = 0.8 / n
        group_positions = np.array([0.0, 1.2])  # slight gap between groups

        for i in range(n):
            offset = (i - n / 2 + 0.5) * width
            _, hatch = _label_style(labels[i])
            ax.bar(group_positions[0] + offset, ret_scores[i], width,
                   color=bar_colors[i], alpha=0.85,
                   edgecolor="black" if hatch else "white",
                   linewidth=0.5, hatch=hatch,
                   label=labels[i])
            ax.bar(group_positions[1] + offset, adap_scores[i], width,
                   color=bar_colors[i], alpha=0.85,
                   edgecolor="black" if hatch else "white",
                   linewidth=0.5, hatch=hatch)

        ax.set_xticks(group_positions)
        ax.set_xticklabels(
            [r"Retention $|\Delta\mathcal{P}_{\mathrm{diag}}|$",
             r"Adaptation $\mathcal{E}_{\Delta}$"],
            fontsize=11)
        ax.set_ylabel("Fluctuation Score", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        if log_scale:
            ax.set_yscale("log")

        # De-duplicate legend handles (each method plotted twice)
        handles, leg_labels = ax.get_legend_handles_labels()
        ax.legend(handles[:n], leg_labels[:n], fontsize=10, framealpha=0.3)

        plt.tight_layout()
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        path = os.path.join(smoothness_dir, f"smoothness_{safe_name}.pdf")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved: {path}")


# ============================================================================
# Plot 5: SVA curves (OFT only)
# ============================================================================

def plot_sva_curves(all_layer_data, all_summaries, labels, output_dir,
                    layer_names=None, dpi=300):
    """Plot Singular Vector Alignment cosine similarity vs index for OFT."""
    # Filter to methods that have SVA data
    sva_indices = []
    for i, summary in enumerate(all_summaries):
        has_sva = any(
            "sva_mean" in v for v in summary["per_layer"].values()
        )
        if has_sva:
            sva_indices.append(i)

    if not sva_indices:
        print("[plot] No SVA data found (SVA is OFT-specific). Skipping SVA plot.")
        return

    # Find layers with SVA data
    sva_layers = set()
    for idx in sva_indices:
        for lname, ldata in all_layer_data[idx].items():
            if "sva_cosines" in ldata:
                sva_layers.add(lname)

    if layer_names is not None:
        requested = set(n.replace(".", "_").replace("/", "_") for n in layer_names)
        sva_layers = sva_layers.intersection(requested)

    sva_layers = sorted(sva_layers)
    if not sva_layers:
        print("[plot] No matching SVA layers found. Skipping SVA plot.")
        return

    for layer_key in sva_layers:
        fig, ax = plt.subplots(figsize=(10, 5))
        short_name = shorten_layer_name(layer_key)

        for idx in sva_indices:
            layer_data = all_layer_data[idx]
            label = labels[idx]
            color = COLORS[idx % len(COLORS)]

            if layer_key not in layer_data or "sva_cosines" not in layer_data[layer_key]:
                continue

            cosines = layer_data[layer_key]["sva_cosines"].numpy()
            indices = np.arange(len(cosines))

            ax.plot(indices, cosines, color=color, alpha=0.6,
                    linewidth=1.5, label=label)

        ax.set_xlabel("Singular Value Index i")
        ax.set_ylabel("$cos(v_pre_i, v_ft_i)$")
        ax.set_title(f"Singular Vector Alignment — {short_name}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)

        plt.tight_layout()
        path = os.path.join(output_dir, f"sva_{layer_key}.pdf")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize spectral analysis of PEFT checkpoints"
    )
    parser.add_argument("--input_dirs", type=str, nargs="+", required=True,
                        help="Directories containing spectral analysis results")
    parser.add_argument("--labels", type=str, nargs="+", required=True,
                        help="Labels for each method (must match input_dirs)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save plots")
    parser.add_argument("--plot_type", type=str, default="all",
                        choices=["all", "effective_rank", "erank_curves",
                                 "spectrum", "curves", "smoothness", "sva"],
                        help="Which plots to generate")
    parser.add_argument("--layer_names", type=str, nargs="*", default=None,
                        help="Specific layer names for per-layer plots")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved figures")
    parser.add_argument("--log_scale", action="store_true", default=False,
                        help="Use log scale for y-axis on retention/adaptation profiling plots")
    parser.add_argument("--color_indices", type=int, nargs="*", default=None,
                        help="Color index for each method curve (e.g. --color_indices 3 2 1 0). "
                             "Indices map into the COLORS palette. Default: 0 1 2 ...")
    args = parser.parse_args()

    assert len(args.input_dirs) == len(args.labels), \
        "Number of input_dirs must match number of labels"

    os.makedirs(args.output_dir, exist_ok=True)
    setup_style()

    # Load all results
    all_summaries = []
    all_layer_data = []
    for input_dir in args.input_dirs:
        summary, layer_data = load_analysis_results(input_dir)
        all_summaries.append(summary)
        all_layer_data.append(layer_data)
        print(f"[plot] Loaded results from: {input_dir} "
              f"({summary['num_layers_analyzed']} layers)")

    plot_type = args.plot_type

    if plot_type in ("all", "effective_rank"):
        plot_effective_rank(all_summaries, args.labels, args.output_dir, args.dpi)

    if plot_type in ("all", "erank_curves"):
        plot_effective_rank_curves(all_summaries, args.labels, args.output_dir,
                                   args.dpi, color_indices=args.color_indices)

    if plot_type in ("all", "spectrum"):
        plot_spectrum_distributions(all_layer_data, args.labels,
                                    args.output_dir, args.dpi,
                                    log_scale=args.log_scale)

    if plot_type in ("all", "curves"):
        plot_spectral_curves(all_layer_data, args.labels, args.output_dir,
                             layer_names=args.layer_names, dpi=args.dpi,
                             log_scale=args.log_scale,
                             color_indices=args.color_indices)

    if plot_type in ("all", "smoothness"):
        plot_smoothness_summary(all_summaries, args.labels,
                                args.output_dir, args.dpi,
                                log_scale=args.log_scale)
        plot_smoothness_per_layer(all_summaries, args.labels,
                                  args.output_dir, args.dpi,
                                  log_scale=args.log_scale,
                                  layer_names=args.layer_names,
                                  color_indices=args.color_indices)

    if plot_type in ("all", "sva"):
        plot_sva_curves(all_layer_data, all_summaries, args.labels,
                        args.output_dir, layer_names=args.layer_names,
                        dpi=args.dpi)

    print(f"\n[plot] All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
