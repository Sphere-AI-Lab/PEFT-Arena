"""Shared reward entrypoint for PEFTArena RL training."""

from __future__ import annotations

from verl.utils.reward_score import default_compute_score

from .math_rlvr import compute_score as compute_math_score


MATH_STYLE_SOURCES = {"openr1", "med_23k_think"}


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Dispatch PEFTArena-local math-style scoring for supported RL datasets."""
    if data_source in MATH_STYLE_SOURCES:
        return compute_math_score(solution_str=solution_str, ground_truth=ground_truth)
    return default_compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
