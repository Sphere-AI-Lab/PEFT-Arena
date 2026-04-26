#!/usr/bin/env python3
"""
Compose a 4x5 panel figure for per-method spectral profiling.

Layout:
  - 10 methods total, arranged in two blocks of 5 columns
  - for each block:
      row A: retention profiling
      row B: adaptation profiling

Each subplot contains:
  - pretrained singular value dashed black line
  - one colored method-specific curve
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("TMPDIR", str(_CACHE_ROOT / "tmp"))
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "tmp").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from plot_spectral_analysis import COLORS, load_analysis_results, setup_style


DEFAULT_INPUT_DIRS = [
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

DEFAULT_LABELS = [
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

DEFAULT_COLOR_INDICES = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]


def _safe_layer_key(layer_name: str) -> str:
    return layer_name.replace(".", "_").replace("/", "_")


def _load_many(repo_root: Path, rel_dirs: list[str]):
    summaries = []
    layer_data = []
    for rel_dir in rel_dirs:
        summary, data = load_analysis_results(str(repo_root / rel_dir))
        summaries.append(summary)
        layer_data.append(data)
    return summaries, layer_data


def _plot_single_metric(
    ax,
    layer_tensor_dict,
    color: str,
    metric_key: str,
    title: str,
    log_scale: bool,
    show_xlabel: bool,
    show_ylabel: bool,
):
    s_pre = layer_tensor_dict.get("S_pre")
    vals = layer_tensor_dict.get(metric_key)
    if vals is None or s_pre is None:
        return

    if metric_key == "delta_P_diag":
        norm_factor = float(s_pre[0]) if float(s_pre[0]) > 0 else 1.0
        ylabel = r"Normalized $|\Delta\mathcal{P}_{\mathrm{diag}}(i)|$"
    else:
        norm_factor = float(s_pre.sum()) if float(s_pre.sum()) > 0 else 1.0
        ylabel = r"Normalized $\mathcal{E}_{\Delta}(i)$"
        vals = vals / vals.sum()

    s_pre_np = s_pre.numpy().copy() / norm_factor
    vals_np = vals.numpy().copy()

    if log_scale:
        s_pre_np = np.where(s_pre_np > 0, s_pre_np, 1e-10)
        vals_np = np.where(vals_np > 0, vals_np, 1e-10)
        ax.set_yscale("log")

    x = np.arange(len(vals_np))
    ax.plot(x, s_pre_np, "k--", linewidth=1.8, alpha=0.8)
    ax.plot(x, vals_np, color=color, linewidth=1.3, alpha=0.9)

    ax.set_title(title, fontsize=11)
    if show_xlabel:
        ax.set_xlabel(r"Singular Value Index $i$", fontsize=10)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    else:
        ax.set_ylabel("")
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)


def build_method_grid(repo_root: Path, output_path: Path, layer_name: str, dpi: int, log_scale: bool):
    _, all_layer_data = _load_many(repo_root, DEFAULT_INPUT_DIRS)
    layer_key = _safe_layer_key(layer_name)

    setup_style()
    matplotlib.rcParams["text.usetex"] = False
    plt.rcParams["text.usetex"] = False

    fig = plt.figure(figsize=(14, 11))
    grid = GridSpec(
        5,
        5,
        figure=fig,
        left=0.055,
        right=0.995,
        bottom=0.055,
        top=0.92,
        width_ratios=[1, 1, 1, 1, 1],
        height_ratios=[1, 1, 0.05, 1, 1],
        wspace=0.18,
        hspace=0.24,
    )
    axes = np.empty((5, 5), dtype=object)
    for col in range(5):
        axes[0, col] = fig.add_subplot(grid[0, col])
        axes[1, col] = fig.add_subplot(grid[1, col])
        axes[2, col] = fig.add_subplot(grid[3, col])
        axes[3, col] = fig.add_subplot(grid[4, col])

    for idx, (layer_data, label, color_idx) in enumerate(
        zip(all_layer_data, DEFAULT_LABELS, DEFAULT_COLOR_INDICES)
    ):
        if layer_key not in layer_data:
            continue

        block = idx // 5
        col = idx % 5
        retention_ax = axes[2 * block, col]
        adaptation_ax = axes[2 * block + 1, col]
        color = COLORS[color_idx % len(COLORS)]

        _plot_single_metric(
            retention_ax,
            layer_data[layer_key],
            color=color,
            metric_key="delta_P_diag",
            title=f"{label}\nRetention Profiling",
            log_scale=log_scale,
            show_xlabel=False,
            show_ylabel=(col == 0),
        )
        _plot_single_metric(
            adaptation_ax,
            layer_data[layer_key],
            color=color,
            metric_key="E_delta",
            title="Adaptation Profiling",
            log_scale=log_scale,
            show_xlabel=True,
            show_ylabel=(col == 0),
        )

    # fig.suptitle("Per-Method Spectral Profiling for model.layers.18.mlp.down_proj.weight", fontsize=16, y=0.995)
    # fig.text(0.5, 0.985, "Dashed black line: pretrained singular values", ha="right", va="bottom", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"[plot] Saved method grid: {output_path}")


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compose a 4x5 per-method spectral profiling figure.")
    parser.add_argument(
        "--output",
        default=str(repo_root / "analysis" / "plot_composite" / "method_grid_layer18_mlp_down_proj.pdf"),
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
        help="Use linear y-axis instead of log-scale.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    build_method_grid(
        repo_root=repo_root,
        output_path=Path(args.output),
        layer_name=args.layer_name,
        dpi=args.dpi,
        log_scale=not args.linear_scale,
    )


if __name__ == "__main__":
    main()
