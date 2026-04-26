#!/usr/bin/env python3
"""Extract a compact table of newly added general benchmarks from summary.csv."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


OUTPUT_COLUMNS = [
    "model",
    "method",
    "train",
    "train_dataset",
    "HumanEval",
    "HellaSwag",
    "WinoGrande",
    "MMLU(avg)",
    "ARC",
    "GSM8K",
    "XCOPA",
]

BENCHMARK_PRIORITY = {
    "humaneval": [("humaneval", "group_avg"), ("openai_humaneval", "pass@1"), ("humaneval", "pass@1")],
    "hellaswag": [("hellaswag", "group_avg"), ("hellaswag", "score"), ("hellaswag", "accuracy"), ("hellaswag", "acc")],
    "winogrande": [("winogrande", "group_avg"), ("winogrande", "score"), ("winogrande", "accuracy"), ("winogrande", "acc")],
    "mmlu_avg": [("mmlu_avg", "group_avg"), ("mmlu", "group_avg")],
    "arc": [("arc", "group_avg"), ("ARC-c", "accuracy"), ("arc", "accuracy"), ("arc", "acc"), ("ARC-c", "acc")],
    "gsm8k": [("gsm8k", "group_avg"), ("gsm8k", "score"), ("gsm8k", "accuracy"), ("gsm8k", "acc")],
    "xcopa": [("xcopa", "group_avg"), ("xcopa", "score"), ("xcopa", "accuracy"), ("xcopa", "acc")],
}

CHECKPOINT_PATTERNS = (
    re.compile(r"^(?P<dataset>math|med)_(?P<train>sft|rl)_(?P<model>qwen2\.5-7b|llama3\.2-3b-ins)_(?P<method>.+)_(?P<step>\d+)$"),
    re.compile(r"^checkpoints_(?P<train>sft|rl)_(?P<dataset>math|med)_(?P<model>qwen2\.5-7b|llama3\.2-3b-ins)_(?P<method>.+)_(?P<step>\d+)$"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/summary.csv",
        help="Path to summary.csv",
    )
    parser.add_argument(
        "--output",
        default="results/new_benchmark_summary.csv",
        help="Path to output CSV",
    )
    return parser.parse_args()


def parse_checkpoint_name(name: str) -> dict[str, str] | None:
    for pattern in CHECKPOINT_PATTERNS:
        match = pattern.match(name)
        if match:
            info = match.groupdict()
            return {
                "model": info["model"],
                "method": info["method"],
                "train": info["train"],
                "train_dataset": info["dataset"],
            }
    return None


def format_value(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def collect_scores(rows: list[dict[str, str]]) -> dict[str, float]:
    score_map: dict[str, float] = {}
    for key, candidates in BENCHMARK_PRIORITY.items():
        value = None
        for benchmark, metric in candidates:
            for row in rows:
                if row["benchmark"] == benchmark and row["metric"] == metric:
                    try:
                        value = float(row["value"])
                    except ValueError:
                        value = None
                    break
            if value is not None:
                break
        score_map[key] = value
    return score_map


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    grouped_rows: dict[str, list[dict[str, str]]] = {}
    with input_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("domain") != "general":
                continue
            parsed = parse_checkpoint_name(row["checkpoint"])
            if parsed is None:
                continue
            grouped_rows.setdefault(row["checkpoint"], []).append(row)

    ordered_checkpoints = sorted(
        grouped_rows,
        key=lambda name: (
            parse_checkpoint_name(name)["model"],
            parse_checkpoint_name(name)["method"],
            parse_checkpoint_name(name)["train"],
            parse_checkpoint_name(name)["train_dataset"],
            name,
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for checkpoint in ordered_checkpoints:
            parsed = parse_checkpoint_name(checkpoint)
            assert parsed is not None
            scores = collect_scores(grouped_rows[checkpoint])
            writer.writerow(
                {
                    "model": parsed["model"],
                    "method": parsed["method"],
                    "train": parsed["train"],
                    "train_dataset": parsed["train_dataset"],
                    "HumanEval": format_value(scores["humaneval"]),
                    "HellaSwag": format_value(scores["hellaswag"]),
                    "WinoGrande": format_value(scores["winogrande"]),
                    "MMLU(avg)": format_value(scores["mmlu_avg"]),
                    "ARC": format_value(scores["arc"]),
                    "GSM8K": format_value(scores["gsm8k"]),
                    "XCOPA": format_value(scores["xcopa"]),
                }
            )

    print(f"Wrote {len(ordered_checkpoints)} rows to {output_path}")


if __name__ == "__main__":
    main()
