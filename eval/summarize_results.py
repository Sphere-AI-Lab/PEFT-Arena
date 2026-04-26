#!/usr/bin/env python3
"""Summarize evaluation results across all domains, grouped by model checkpoint.

Scans the results directory for per-checkpoint subdirectories and collects
math, medical, and general evaluation outputs into a unified summary table.

Average scores computed as follows:
  - math_avg  = mean(aime24, amc23, math500)  (acc)
  - med_avg   = mean(all medical benchmarks)   (acc, 0-100 scale)
  - general_avg = mean(all available grouped general benchmarks), where
      grouped components include:
      * ifeval      = IFEval Inst-level-strict-accuracy
      * bbh_avg     = mean(bbh-*)
      * nq          = NQ score
      * mmlu_avg    = mean(lukaemon_mmlu_*)
      * gsm8k       = GSM8K
      * arc         = ARC-Challenge / ARC-c
      * hellaswag   = HellaSwag
      * winogrande  = WinoGrande
      * xcopa       = XCOPA
      * humaneval   = HumanEval pass@1

Directory layout expected:
    results/<checkpoint>/math/<benchmark>/*_metrics.json
    results/<checkpoint>/med/<checkpoint>_score.json
    results/<checkpoint>/general/<checkpoint>_<benchmark>/<timestamp>/summary/*.csv

Legacy general layout is also supported:
    results/general/<checkpoint>_<benchmark>/<timestamp>/summary/*.csv

Usage:
    python eval/summarize_results.py --results_dir results/
    python eval/summarize_results.py --results_dir results/ --output results/summary.csv
"""

import argparse
import csv
import json
import os
import glob
from collections import defaultdict

# ---------------------------------------------------------------------------
# General benchmark helpers
# ---------------------------------------------------------------------------

GENERAL_GROUP_SPECS = (
    ("ifeval", lambda name: name == "IFEval"),
    ("bbh_avg", lambda name: name.startswith("bbh-")),
    ("nq", lambda name: name == "nq"),
    ("mmlu_avg", lambda name: name == "mmlu" or name.startswith("lukaemon_mmlu_")),
    ("gsm8k", lambda name: name == "gsm8k"),
    ("arc", lambda name: name.lower() in {"arc", "arc-c", "arc-challenge"}),
    ("hellaswag", lambda name: name == "hellaswag"),
    ("winogrande", lambda name: name == "winogrande"),
    ("xcopa", lambda name: name.lower() == "xcopa"),
    (
        "humaneval",
        lambda name: name.lower() in {"openai_humaneval", "humaneval"},
    ),
)


def _general_metric_priority(dataset: str, metric_name: str) -> int:
    """Return metric priority for a general benchmark row.

    Higher priority rows are preferred when a dataset appears multiple times in a
    summary CSV with different metrics.
    """
    dataset_lower = dataset.lower()
    metric_lower = (metric_name or "").lower()

    if dataset == "IFEval":
        return 100 if metric_name == "Inst-level-strict-accuracy" else -1

    if dataset_lower in {"openai_humaneval", "humaneval"}:
        if metric_lower in {"humaneval_pass@1", "pass@1"}:
            return 100
        return -1

    if metric_lower == "score":
        return 30
    if metric_lower in {"acc", "accuracy"}:
        return 25
    if "accuracy" in metric_lower:
        return 20
    return 10


def _iter_general_group_scores(general_results: dict):
    """Yield (group_name, avg_score) for grouped general benchmarks."""
    for group_name, matcher in GENERAL_GROUP_SPECS:
        scores = [
            data["score"]
            for benchmark, data in general_results.items()
            if matcher(benchmark)
        ]
        if scores:
            yield group_name, sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect_math_results(checkpoint_dir: str) -> dict:
    """Collect math evaluation results from a single checkpoint directory.

    Returns dict mapping benchmark_name -> metrics dict.
    Each metrics dict has keys: acc, pass@1, pass@k (full dict).
    """
    math_dir = os.path.join(checkpoint_dir, "math")
    results = {}

    if not os.path.isdir(math_dir):
        return results

    for metric_file in glob.glob(os.path.join(math_dir, "**", "*_metrics.json"), recursive=True):
        dataset_name = os.path.basename(os.path.dirname(metric_file))
        with open(metric_file) as f:
            metrics = json.load(f)
        results[dataset_name] = {
            "acc": metrics.get("acc", 0.0),
            "pass@k": metrics.get("pass@k", {}),
        }

    return results


def collect_med_results(checkpoint_dir: str) -> dict:
    """Collect medical evaluation results from a single checkpoint directory.

    Returns dict mapping benchmark_name -> {"acc": float}.
    """
    med_dir = os.path.join(checkpoint_dir, "med")
    results = {}

    if not os.path.isdir(med_dir):
        return results

    for score_file in glob.glob(os.path.join(med_dir, "**", "*_score.json"), recursive=True):
        try:
            with open(score_file) as f:
                data = json.load(f)
            for ds, ds_data in data.items():
                if ds == "average":
                    continue
                if isinstance(ds_data, dict):
                    results[ds] = {"acc": ds_data.get("acc", 0.0)}
                else:
                    results[ds] = {"acc": ds_data}
        except Exception:
            pass

    return results


def collect_general_results(results_dir: str) -> dict:
    """Collect general evaluation results from OpenCompass summary CSVs.

    For each model column found in the summary CSVs, collect one preferred
    metric per dataset:
      - IFEval -> Inst-level-strict-accuracy
      - HumanEval -> pass@1
      - other datasets -> prefer score / accuracy-like metrics

    Returns dict: checkpoint_name -> {benchmark: {"score": float, "metric": str}}
    """
    # checkpoint_name -> {benchmark -> {"score": float, "metric": str}}
    results = defaultdict(dict)

    summary_files = set()
    summary_files.update(
        glob.glob(
            os.path.join(results_dir, "general", "**", "summary", "*.csv"),
            recursive=True,
        ))
    summary_files.update(
        glob.glob(
            os.path.join(results_dir, "*", "general", "**", "summary", "*.csv"),
            recursive=True,
        ))

    for summary_file in sorted(summary_files):
        rel_parts = os.path.relpath(summary_file, results_dir).split(os.sep)
        checkpoint_hint = None
        if len(rel_parts) >= 2 and rel_parts[0] != "general" and rel_parts[1] == "general":
            checkpoint_hint = rel_parts[0]

        with open(summary_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row.get("dataset", "unknown")
                metric_name = row.get("metric", "score")

                metric_priority = _general_metric_priority(dataset, metric_name)
                if metric_priority < 0:
                    continue

                model_cols = [
                    k for k in row.keys()
                    if k not in ("dataset", "version", "metric", "mode")
                ]
                if checkpoint_hint:
                    if checkpoint_hint in model_cols:
                        model_targets = [(checkpoint_hint, checkpoint_hint)]
                    elif len(model_cols) == 1:
                        model_targets = [(checkpoint_hint, model_cols[0])]
                    else:
                        model_targets = [(model, model) for model in model_cols]
                else:
                    model_targets = [(model, model) for model in model_cols]

                for result_key, model_col in model_targets:
                    value = row.get(model_col)
                    try:
                        existing = results[result_key].get(dataset)
                        if existing and existing.get("_priority", -1) > metric_priority:
                            continue

                        results[result_key][dataset] = {
                            "score": float(value),
                            "metric": metric_name,
                            "_priority": metric_priority,
                        }
                    except (ValueError, TypeError):
                        pass

    cleaned_results = {}
    for checkpoint, checkpoint_results in results.items():
        cleaned_results[checkpoint] = {
            benchmark: {
                "score": data["score"],
                "metric": data["metric"],
            }
            for benchmark, data in checkpoint_results.items()
        }

    return cleaned_results


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(results_dir: str) -> list:
    """Discover checkpoint directories under results_dir.

    A checkpoint dir is any subdirectory containing a math/, med/, or general/ subdir
    (excluding the legacy top-level math/med/general dirs).
    """
    checkpoints = []
    if not os.path.isdir(results_dir):
        return checkpoints

    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry in ("math", "med", "general"):
            continue
        has_math = os.path.isdir(os.path.join(entry_path, "math"))
        has_med = os.path.isdir(os.path.join(entry_path, "med"))
        has_general = os.path.isdir(os.path.join(entry_path, "general"))
        if has_math or has_med or has_general:
            checkpoints.append(entry)

    return checkpoints


# ---------------------------------------------------------------------------
# Average score helpers
# ---------------------------------------------------------------------------

def compute_math_avg(math_results: dict) -> dict:
    """Compute math average across aime24, amc23, math500.

    Returns {"avg_acc": float, "pass@k": {k: avg_score}, "benchmarks_used": [...]}.
    """
    target_benchmarks = ["aime24", "amc23", "math500"]
    accs = []
    all_pass_k = defaultdict(list)

    for bm in target_benchmarks:
        if bm in math_results:
            accs.append(math_results[bm]["acc"])
            for k, v in math_results[bm].get("pass@k", {}).items():
                all_pass_k[k].append(v)

    if not accs:
        return {}

    avg_pass_k = {}
    for k in sorted(all_pass_k.keys(), key=lambda x: int(x)):
        vals = all_pass_k[k]
        if len(vals) == len(accs):  # only average if all benchmarks have this k
            avg_pass_k[k] = sum(vals) / len(vals)

    return {
        "avg_acc": sum(accs) / len(accs),
        "pass@k": avg_pass_k,
        "benchmarks_used": [bm for bm in target_benchmarks if bm in math_results],
    }


def compute_med_avg(med_results: dict) -> dict:
    """Compute medical average across all benchmarks.

    Returns {"avg_acc": float, "benchmarks_used": [...]}.
    acc values are on 0-1 scale in the source, converted to 0-100 for display.
    """
    if not med_results:
        return {}

    accs = [m["acc"] for m in med_results.values()]
    return {
        "avg_acc": sum(accs) / len(accs),  # still 0-1 scale
        "benchmarks_used": sorted(med_results.keys()),
    }


def compute_general_avg(general_results: dict) -> dict:
    """Compute grouped general averages and overall general average.

    Returns a dict containing available grouped components plus:
      - avg_score
      - components_used
    """
    components = {}
    scores_for_avg = []
    components_used = []

    for group_name, score in _iter_general_group_scores(general_results):
        components[group_name] = score
        scores_for_avg.append(score)
        components_used.append(group_name)

    if not scores_for_avg:
        return {}

    components["avg_score"] = sum(scores_for_avg) / len(scores_for_avg)
    components["components_used"] = components_used
    return components


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(all_results: dict):
    """Print a formatted summary of all results grouped by checkpoint."""
    print("\n" + "=" * 90)
    print("PEFTArena Evaluation Summary (grouped by checkpoint)")
    print("=" * 90)

    for checkpoint in sorted(all_results.keys()):
        data = all_results[checkpoint]
        print(f"\n🔖 Checkpoint: {checkpoint}")
        print("-" * 80)

        math = data.get("math", {})
        med = data.get("med", {})
        general = data.get("general", {})
        math_avg = data.get("math_avg", {})
        med_avg = data.get("med_avg", {})
        general_avg = data.get("general_avg", {})

        # --- Math ---
        if math:
            print("  📐 Math Benchmarks:")
            for dataset in sorted(math.keys()):
                metrics = math[dataset]
                acc = metrics.get("acc", 0.0)
                pass_at_k = metrics.get("pass@k", {})
                pass_str = ""
                if pass_at_k:
                    parts = [f"pass@{k}={v:.1f}" for k, v in
                             sorted(pass_at_k.items(), key=lambda x: int(x[0]))]
                    pass_str = "  " + " | ".join(parts)
                print(f"      {dataset:25s}  acc: {acc:.1f}{pass_str}")

            if math_avg:
                avg_acc = math_avg["avg_acc"]
                pass_k = math_avg.get("pass@k", {})
                pass_str = ""
                if pass_k:
                    parts = [f"pass@{k}={v:.1f}" for k, v in
                             sorted(pass_k.items(), key=lambda x: int(x[0]))]
                    pass_str = "  " + " | ".join(parts)
                print(f"      {'>>> math_avg':25s}  acc: {avg_acc:.1f}{pass_str}")

        # --- Medical ---
        if med:
            print("  🏥 Medical Benchmarks:")
            for task in sorted(med.keys()):
                metrics = med[task]
                acc = metrics.get("acc", 0.0) * 100
                print(f"      {task:25s}  acc: {acc:.1f}")

            if med_avg:
                avg_acc = med_avg["avg_acc"] * 100
                print(f"      {'>>> med_avg':25s}  acc: {avg_acc:.1f}")

        # --- General ---
        if general:
            print("  📊 General Benchmarks:")
            for benchmark in sorted(general.keys()):
                metrics = general[benchmark]
                score = metrics.get("score", 0.0)
                metric_name = metrics.get("metric", "score")
                print(f"      {benchmark:25s}  {metric_name}: {score:.2f}")

            if general_avg:
                avg = general_avg.get("avg_score")
                detail_parts = [
                    f"{name}={general_avg[name]:.2f}"
                    for name in general_avg.get("components_used", [])
                    if name in general_avg
                ]
                detail = ", ".join(detail_parts)
                print(f"      {'>>> general_avg':25s}  score: {avg:.2f}  ({detail})")

        if not math and not med and not general:
            print("  (no results found)")

    print("\n" + "=" * 90)


def save_csv(all_results: dict, output_path: str):
    """Save all results to a single CSV file with a checkpoint column."""
    rows = []

    for checkpoint in sorted(all_results.keys()):
        data = all_results[checkpoint]

        # Math per-benchmark
        for dataset in sorted(data.get("math", {}).keys()):
            metrics = data["math"][dataset]
            rows.append({
                "checkpoint": checkpoint,
                "domain": "math",
                "benchmark": dataset,
                "metric": "acc",
                "value": metrics.get("acc", 0.0),
            })
            # Also output pass@k values
            for k, v in sorted(metrics.get("pass@k", {}).items(), key=lambda x: int(x[0])):
                rows.append({
                    "checkpoint": checkpoint,
                    "domain": "math",
                    "benchmark": dataset,
                    "metric": f"pass@{k}",
                    "value": v,
                })

        # Math avg
        math_avg = data.get("math_avg", {})
        if math_avg:
            rows.append({
                "checkpoint": checkpoint,
                "domain": "math",
                "benchmark": "AVG",
                "metric": "acc",
                "value": math_avg["avg_acc"],
            })
            for k, v in sorted(math_avg.get("pass@k", {}).items(), key=lambda x: int(x[0])):
                rows.append({
                    "checkpoint": checkpoint,
                    "domain": "math",
                    "benchmark": "AVG",
                    "metric": f"pass@{k}",
                    "value": v,
                })

        # Medical per-benchmark
        for task in sorted(data.get("med", {}).keys()):
            metrics = data["med"][task]
            rows.append({
                "checkpoint": checkpoint,
                "domain": "medical",
                "benchmark": task,
                "metric": "acc",
                "value": metrics.get("acc", 0.0),
            })

        # Medical avg
        med_avg = data.get("med_avg", {})
        if med_avg:
            rows.append({
                "checkpoint": checkpoint,
                "domain": "medical",
                "benchmark": "AVG",
                "metric": "acc",
                "value": med_avg["avg_acc"],
            })

        # General per-benchmark
        for benchmark in sorted(data.get("general", {}).keys()):
            metrics = data["general"][benchmark]
            rows.append({
                "checkpoint": checkpoint,
                "domain": "general",
                "benchmark": benchmark,
                "metric": metrics.get("metric", "score"),
                "value": metrics.get("score", 0.0),
            })

        # General avg
        general_avg = data.get("general_avg", {})
        if general_avg:
            for group_name in general_avg.get("components_used", []):
                rows.append({
                    "checkpoint": checkpoint,
                    "domain": "general",
                    "benchmark": group_name,
                    "metric": "group_avg",
                    "value": general_avg[group_name],
                })
            rows.append({
                "checkpoint": checkpoint,
                "domain": "general",
                "benchmark": "AVG",
                "metric": (
                    "mean("
                    + ",".join(general_avg.get("components_used", []))
                    + ")"
                ),
                "value": general_avg["avg_score"],
            })

    if rows:
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["checkpoint", "domain", "benchmark", "metric", "value"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[summary] Results saved to {output_path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Summarize PEFTArena evaluation results")
    parser.add_argument("--results_dir", default="results", help="Root results directory")
    parser.add_argument("--output", default=None, help="Output CSV file path")
    args = parser.parse_args()

    results_dir = args.results_dir
    all_results = {}

    # --- Per-checkpoint math & med results ---
    checkpoints = discover_checkpoints(results_dir)
    for ckpt in checkpoints:
        ckpt_dir = os.path.join(results_dir, ckpt)
        math_results = collect_math_results(ckpt_dir)
        med_results = collect_med_results(ckpt_dir)
        if math_results or med_results:
            entry = {"math": math_results, "med": med_results}
            entry["math_avg"] = compute_math_avg(math_results)
            entry["med_avg"] = compute_med_avg(med_results)
            all_results[ckpt] = entry

    # --- Legacy top-level math/ and med/ ---
    legacy_math = collect_math_results(results_dir)
    legacy_med = collect_med_results(results_dir)
    if legacy_math or legacy_med:
        entry = all_results.setdefault("(legacy/unknown)", {})
        entry["math"] = legacy_math
        entry["med"] = legacy_med
        entry["math_avg"] = compute_math_avg(legacy_math)
        entry["med_avg"] = compute_med_avg(legacy_med)

    # --- General results (keyed by model name in CSV columns) ---
    general_by_model = collect_general_results(results_dir)
    for model, benchmarks in general_by_model.items():
        entry = all_results.setdefault(model, {})
        entry["general"] = benchmarks
        entry["general_avg"] = compute_general_avg(benchmarks)

    # --- Print & save ---
    print_summary(all_results)

    output_path = args.output or os.path.join(results_dir, "summary.csv")
    save_csv(all_results, output_path)


if __name__ == "__main__":
    main()
