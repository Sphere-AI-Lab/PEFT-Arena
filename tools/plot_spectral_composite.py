#!/usr/bin/env python3
"""
Compose a 2x3 overview figure for the layer-18 MLP down_proj analysis.

Layout:
  Row 1:
    [SFT spectral retention] [SFT spectral adaptation] [effective rank]
  Row 2:
    [RL spectral retention ] [RL spectral adaptation ] [smoothness]

All six subplot frames are created by a single ImageGrid so their black
bounding boxes stay exactly the same size.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("TMPDIR", str(_CACHE_ROOT / "tmp"))
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "tmp").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from plot_spectral_analysis import (
    COLORS,
    MARKERS,
    _label_style,
    load_analysis_results,
    setup_style,
)


DEFAULT_SFT_SPECTRUM_DIRS = [
    "analysis/math/sft/qwen2.5-7b/full/global_step_780",
    "analysis/math/sft/qwen2.5-7b/oft-b32/global_step_780",
    "analysis/math/sft/qwen2.5-7b/lora-r8/global_step_780",
]

DEFAULT_SFT_SPECTRUM_LABELS = [
    "SFT-FullFT",
    "SFT-OFT-b32",
    "SFT-LoRA-r8",
]

DEFAULT_RL_SPECTRUM_DIRS = [
    "analysis/math/rl/qwen2.5-7b/full/global_step_200",
    "analysis/math/rl/qwen2.5-7b/oft-b32/global_step_200",
    "analysis/math/rl/qwen2.5-7b/lora-r8/global_step_200",
]

DEFAULT_RL_SPECTRUM_LABELS = [
    "RL-FullFT",
    "RL-OFT-b32",
    "RL-LoRA-r8",
]

DEFAULT_COMPARISON_DIRS = [
    "analysis/math/sft/qwen2.5-7b/full/global_step_780",
    "analysis/math/sft/qwen2.5-7b/oft-b32/global_step_780",
    "analysis/math/sft/qwen2.5-7b/lora-r8/global_step_780",
    "analysis/math/sft/qwen2.5-7b/miss-r8/global_step_780",
    "analysis/math/sft/qwen2.5-7b/pissa-r8/global_step_780",
    "analysis/math/sft/qwen2.5-7b/vera-256/global_step_780",
    "analysis/math/sft/qwen2.5-7b/adalora-r8/global_step_780",
    "analysis/math/rl/qwen2.5-7b/full/global_step_200",
    "analysis/math/rl/qwen2.5-7b/oft-b32/global_step_200",
    "analysis/math/rl/qwen2.5-7b/lora-r8/global_step_200",
]

DEFAULT_COMPARISON_LABELS = [
    "SFT-FullFT",
    "SFT-OFT-b32",
    "SFT-LoRA-r8",
    "SFT-MiSS-r8",
    "SFT-PiSSA-r8",
    "SFT-VeRA-256",
    "SFT-AdaLoRA-r8",
    "RL-FullFT",
    "RL-OFT-b32",
    "RL-LoRA-r8",
]

DEFAULT_COMPARISON_COLOR_INDICES = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
DEFAULT_SPECTRUM_COLOR_INDICES = [0, 1, 2]


def _load_many(repo_root: Path, rel_dirs: list[str]):
    summaries = []
    layer_data = []
    for rel_dir in rel_dirs:
        input_dir = repo_root / rel_dir
        summary, data = load_analysis_results(str(input_dir))
        summaries.append(summary)
        layer_data.append(data)
    return summaries, layer_data


def _safe_layer_key(layer_name: str) -> str:
    return layer_name.replace(".", "_").replace("/", "_")


def _plot_spectral_metric_panel(
    ax,
    all_layer_data,
    labels,
    layer_name: str,
    metric_key: str,
    title: str,
    ylabel: str,
    color_indices: list[int],
    log_scale: bool,
    show_legend: bool,
):
    layer_key = _safe_layer_key(layer_name)
    pretrained_plotted = False

    for i, (layer_data, label) in enumerate(zip(all_layer_data, labels)):
        if layer_key not in layer_data:
            continue

        vals = layer_data[layer_key].get(metric_key)
        if vals is None:
            continue

        s_pre = layer_data[layer_key].get("S_pre")
        if s_pre is not None and len(s_pre) > 0:
            if metric_key == "delta_P_diag":
                norm_factor = float(s_pre[0]) if float(s_pre[0]) > 0 else 1.0
            else:
                norm_factor = float(s_pre.sum()) if float(s_pre.sum()) > 0 else 1.0
        else:
            norm_factor = 1.0

        if not pretrained_plotted and s_pre is not None:
            s_pre_np = s_pre.numpy().copy() / norm_factor
            if log_scale:
                s_pre_np = np.where(s_pre_np > 0, s_pre_np, 1e-10)
            ax.plot(
                np.arange(len(s_pre_np)),
                s_pre_np,
                "k--",
                linewidth=2,
                alpha=0.8,
                label="Pretrained SV",
            )
            pretrained_plotted = True

        if metric_key == "E_delta":
            vals = vals / vals.sum()
        vals_np = vals.numpy().copy()
        if log_scale:
            vals_np = np.where(vals_np > 0, vals_np, 1e-10)

        ax.plot(
            np.arange(len(vals_np)),
            vals_np,
            color=COLORS[color_indices[i] % len(COLORS)],
            linestyle="-",
            alpha=0.6,
            linewidth=1.5,
            label=label,
        )

    if log_scale:
        ax.set_yscale("log")
        ylabel = ylabel + r" (log scale)"
    # ax.set_title(title, fontsize=13)
    ax.set_xlabel(r"Singular Value Index $i$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=10, framealpha=0.3, loc="best")


def _plot_effective_rank_curve_panel(
    ax,
    all_summaries,
    labels,
    module_type: str,
    color_indices: list[int],
    show_legend: bool,
):
    per_method_data = []
    for summary in all_summaries:
        method_data = {}
        for layer_name, layer_info in summary["per_layer"].items():
            match = re.search(r"layers\.(\d+)\.(.+?)(?:\.weight)?$", layer_name)
            if match:
                layer_idx = int(match.group(1))
                parsed_module_type = match.group(2)
            else:
                continue
            if parsed_module_type != module_type:
                continue
            method_data[layer_idx] = layer_info["effective_rank"]
        per_method_data.append(method_data)

    for i, (method_data, label) in enumerate(zip(per_method_data, labels)):
        if not method_data:
            continue
        indices = sorted(method_data.keys())
        eranks = [method_data[idx] for idx in indices]
        linestyle, _ = _label_style(label)
        ax.plot(
            indices,
            eranks,
            color=COLORS[color_indices[i] % len(COLORS)],
            linestyle=linestyle,
            alpha=0.7,
            linewidth=1.5,
            marker=MARKERS[i % len(MARKERS)],
            markersize=5,
            markevery=max(1, len(indices) // 10),
            label=label,
        )

    # ax.set_title("Effective Rank", fontsize=13)
    ax.set_xlabel(r"Layer Index", fontsize=12)
    ax.set_ylabel(r"Effective Rank", fontsize=12)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=9, framealpha=0.3, loc="best")


def _plot_smoothness_panel(
    ax,
    all_summaries,
    labels,
    layer_name: str,
    color_indices: list[int],
    log_scale: bool,
    show_legend: bool,
):
    ret_scores = []
    adap_scores = []
    bar_colors = []

    for i, summary in enumerate(all_summaries):
        if layer_name in summary["per_layer"]:
            ret_scores.append(summary["per_layer"][layer_name]["smoothness_retention"])
            adap_scores.append(summary["per_layer"][layer_name]["smoothness_adaptation"])
        else:
            ret_scores.append(0)
            adap_scores.append(0)
        bar_colors.append(COLORS[color_indices[i] % len(COLORS)])

    n_methods = len(labels)
    width = 0.8 / n_methods
    group_positions = np.array([0.0, 1.2])

    for i, label in enumerate(labels):
        offset = (i - n_methods / 2 + 0.5) * width
        _, hatch = _label_style(label)
        ax.bar(
            group_positions[0] + offset,
            ret_scores[i],
            width,
            color=bar_colors[i],
            alpha=0.85,
            edgecolor="black" if hatch else "white",
            linewidth=0.5,
            hatch=hatch,
            label=label,
        )
        ax.bar(
            group_positions[1] + offset,
            adap_scores[i],
            width,
            color=bar_colors[i],
            alpha=0.85,
            edgecolor="black" if hatch else "white",
            linewidth=0.5,
            hatch=hatch,
        )

    # ax.set_title("Smoothness", fontsize=13)
    ax.set_xticks(group_positions)
    ax.set_xticklabels(
        [
            r"Retention $|\Delta\mathcal{P}_{\mathrm{diag}}|$",
            r"Adaptation $\mathcal{E}_{\Delta}$",
        ],
        fontsize=11,
    )
    ax.set_ylabel("Fluctuation Score", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    if log_scale:
        ax.set_yscale("log")
    if show_legend:
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(handles[:n_methods], legend_labels[:n_methods], fontsize=9, framealpha=0.3, loc="best")


def build_composite_figure(
    repo_root: Path,
    output_path: Path,
    layer_name: str,
    dpi: int,
    log_scale: bool,
):
    sft_summaries, sft_layer_data = _load_many(repo_root, DEFAULT_SFT_SPECTRUM_DIRS)
    rl_summaries, rl_layer_data = _load_many(repo_root, DEFAULT_RL_SPECTRUM_DIRS)
    comp_summaries, _ = _load_many(repo_root, DEFAULT_COMPARISON_DIRS)

    setup_style()
    matplotlib.rcParams["text.usetex"] = False
    plt.rcParams["text.usetex"] = False

    fig = plt.figure(figsize=(18, 9))
    outer = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.22,
        hspace=0.28,
    )

    # Use temporary axes to obtain exact subplot rectangles from GridSpec.
    tmp_top_left = fig.add_subplot(outer[0, 0])
    tmp_top_mid = fig.add_subplot(outer[0, 1])
    tmp_top_span = fig.add_subplot(outer[0, :2])
    tmp_bottom_span = fig.add_subplot(outer[1, :2])
    gap_in = (tmp_top_mid.get_position().x0 - tmp_top_left.get_position().x1) * fig.get_figwidth()
    top_rect = tmp_top_span.get_position().bounds
    bottom_rect = tmp_bottom_span.get_position().bounds
    tmp_top_left.remove()
    tmp_top_mid.remove()
    tmp_top_span.remove()
    tmp_bottom_span.remove()

    top_grid = ImageGrid(
        fig,
        top_rect,
        nrows_ncols=(1, 2),
        axes_pad=gap_in,
        share_all=False,
        label_mode="all",
        aspect=False,
    )
    top_grid[0]._shared_axes["y"].remove(top_grid[1])
    bottom_grid = ImageGrid(
        fig,
        bottom_rect,
        nrows_ncols=(1, 2),
        axes_pad=gap_in,
        share_all=False,
        label_mode="all",
        aspect=False,
    )
    bottom_grid[0]._shared_axes["y"].remove(bottom_grid[1])
    erank_ax = fig.add_subplot(outer[0, 2])
    smooth_ax = fig.add_subplot(outer[1, 2])

    _plot_spectral_metric_panel(
        top_grid[0],
        sft_layer_data,
        DEFAULT_SFT_SPECTRUM_LABELS,
        layer_name,
        metric_key="delta_P_diag",
        title="SFT Retention Profiling",
        ylabel=r"Normalized $|\Delta\mathcal{P}_{\mathrm{diag}}(i)|$",
        color_indices=DEFAULT_SPECTRUM_COLOR_INDICES,
        log_scale=log_scale,
        show_legend=False,
    )
    _plot_spectral_metric_panel(
        top_grid[1],
        sft_layer_data,
        DEFAULT_SFT_SPECTRUM_LABELS,
        layer_name,
        metric_key="E_delta",
        title="SFT Adaptation Profiling",
        ylabel=r"Normalized $\mathcal{E}_{\Delta}(i)$",
        color_indices=DEFAULT_SPECTRUM_COLOR_INDICES,
        log_scale=log_scale,
        show_legend=True,
    )
    _plot_effective_rank_curve_panel(
        erank_ax,
        comp_summaries,
        DEFAULT_COMPARISON_LABELS,
        module_type="mlp.down_proj",
        color_indices=DEFAULT_COMPARISON_COLOR_INDICES,
        show_legend=True,
    )

    _plot_spectral_metric_panel(
        bottom_grid[0],
        rl_layer_data,
        DEFAULT_RL_SPECTRUM_LABELS,
        layer_name,
        metric_key="delta_P_diag",
        title="RL Retention Profiling",
        ylabel=r"Normalized $|\Delta\mathcal{P}_{\mathrm{diag}}(i)|$",
        color_indices=DEFAULT_SPECTRUM_COLOR_INDICES,
        log_scale=log_scale,
        show_legend=False,
    )
    _plot_spectral_metric_panel(
        bottom_grid[1],
        rl_layer_data,
        DEFAULT_RL_SPECTRUM_LABELS,
        layer_name,
        metric_key="E_delta",
        title="RL Adaptation Profiling",
        ylabel=r"Normalized $\mathcal{E}_{\Delta}(i)$",
        color_indices=DEFAULT_SPECTRUM_COLOR_INDICES,
        log_scale=log_scale,
        show_legend=True,
    )
    _plot_smoothness_panel(
        smooth_ax,
        comp_summaries,
        DEFAULT_COMPARISON_LABELS,
        layer_name=layer_name,
        color_indices=DEFAULT_COMPARISON_COLOR_INDICES,
        log_scale=log_scale,
        show_legend=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"[plot] Saved composite figure: {output_path}")


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compose the layer-18 spectral overview figure.")
    parser.add_argument(
        "--output",
        default=str(repo_root / "analysis" / "plot_composite" / "composite_layer18_mlp_down_proj.pdf"),
        help="Output path for the composite PDF.",
    )
    parser.add_argument(
        "--layer_name",
        default="model.layers.18.mlp.down_proj.weight",
        help="Layer name to visualize.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument(
        "--linear_scale",
        action="store_true",
        help="Use linear y-axis instead of log-scale for spectral/smoothness panels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    build_composite_figure(
        repo_root=repo_root,
        output_path=Path(args.output),
        layer_name=args.layer_name,
        dpi=args.dpi,
        log_scale=not args.linear_scale,
    )


if __name__ == "__main__":
    main()
