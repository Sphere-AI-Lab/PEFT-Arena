from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run import derive_eval_model_name, resolve_eval_output_dir


def test_derive_eval_model_name_strips_checkpoint_prefix():
    checkpoint = str(REPO_ROOT / "checkpoints" / "sft" / "math" / "qwen2.5-7b" / "lora-r16" / "global_step_780")
    expected = "sft_math_qwen2.5-7b_lora-r16_780"
    assert derive_eval_model_name(checkpoint) == expected


def test_resolve_eval_output_dir_appends_checkpoint_under_results_root():
    checkpoint = str(REPO_ROOT / "checkpoints" / "sft" / "math" / "qwen2.5-7b" / "lora-r16" / "global_step_780")
    output_dir = resolve_eval_output_dir(checkpoint, "/tmp/results", "math")
    assert output_dir == "/tmp/results/sft_math_qwen2.5-7b_lora-r16_780/math"


def test_resolve_eval_output_dir_rewrites_legacy_domain_root():
    checkpoint = str(REPO_ROOT / "checkpoints" / "sft" / "math" / "qwen2.5-7b" / "lora-r16" / "global_step_780")
    output_dir = resolve_eval_output_dir(checkpoint, "/tmp/results/math", "math")
    assert output_dir == "/tmp/results/sft_math_qwen2.5-7b_lora-r16_780/math"


def test_resolve_eval_output_dir_keeps_explicit_leaf():
    checkpoint = str(REPO_ROOT / "checkpoints" / "sft" / "math" / "qwen2.5-7b" / "lora-r16" / "global_step_780")
    output_dir = resolve_eval_output_dir(checkpoint, "/tmp/custom_leaf", "math")
    assert output_dir == "/tmp/custom_leaf"
