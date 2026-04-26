import csv
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'eval' / 'summarize_results.py'


def _load_module():
    spec = importlib.util.spec_from_file_location(
        'summarize_results', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary_csv(path: Path, model_col: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['dataset', 'version', 'metric', 'mode', model_col])
        writer.writeheader()
        writer.writerows(rows)


def test_collect_general_results_from_checkpoint_general_layout(tmp_path):
    module = _load_module()
    results_dir = tmp_path / 'results'
    checkpoint = 'med_sft_qwen2.5-7b_adalora-r8_364'
    summary_path = (
        results_dir / checkpoint / 'general' /
        f'{checkpoint}_ifeval_nq' / '20260306_075541' / 'summary' /
        'summary_20260306_075541.csv')

    _write_summary_csv(
        summary_path,
        'model_alias',
        [
            dict(
                dataset='IFEval',
                version='3321a3',
                metric='Prompt-level-strict-accuracy',
                mode='gen',
                model_alias='1.0',
            ),
            dict(
                dataset='IFEval',
                version='3321a3',
                metric='Inst-level-strict-accuracy',
                mode='gen',
                model_alias='77.5',
            ),
            dict(
                dataset='nq',
                version='c788f6',
                metric='score',
                mode='gen',
                model_alias='41.2',
            ),
        ],
    )

    results = module.collect_general_results(str(results_dir))

    assert checkpoint in results
    assert 'Prompt-level-strict-accuracy' not in {
        item['metric'] for item in results[checkpoint].values()
    }
    assert results[checkpoint]['IFEval'] == {
        'score': 77.5,
        'metric': 'Inst-level-strict-accuracy',
    }
    assert results[checkpoint]['nq'] == {
        'score': 41.2,
        'metric': 'score',
    }


def test_collect_general_results_from_legacy_general_layout(tmp_path):
    module = _load_module()
    results_dir = tmp_path / 'results'
    summary_path = (
        results_dir / 'general' / 'legacy_model_bbh' / '20260306_070151' /
        'summary' / 'summary_20260306_073855.csv')

    _write_summary_csv(
        summary_path,
        'legacy_model',
        [
            dict(
                dataset='bbh-temporal_sequences',
                version='e43931',
                metric='score',
                mode='gen',
                legacy_model='64.8',
            ),
        ],
    )

    results = module.collect_general_results(str(results_dir))

    assert results == {
        'legacy_model': {
            'bbh-temporal_sequences': {
                'score': 64.8,
                'metric': 'score',
            }
        }
    }


def test_discover_checkpoints_includes_general_only_dirs(tmp_path):
    module = _load_module()
    results_dir = tmp_path / 'results'
    (results_dir / 'general_only_ckpt' / 'general').mkdir(parents=True)

    checkpoints = module.discover_checkpoints(str(results_dir))

    assert checkpoints == ['general_only_ckpt']


def test_collect_general_results_prefers_humaneval_pass_at_1(tmp_path):
    module = _load_module()
    results_dir = tmp_path / 'results'
    checkpoint = 'math_sft_qwen2.5-7b_full_500'
    summary_path = (
        results_dir / checkpoint / 'general' /
        f'{checkpoint}_humaneval' / '20260406_010101' / 'summary' /
        'summary_20260406_010101.csv')

    _write_summary_csv(
        summary_path,
        'model_alias',
        [
            dict(
                dataset='openai_humaneval',
                version='dcae0e',
                metric='pass@10',
                mode='gen',
                model_alias='42.0',
            ),
            dict(
                dataset='openai_humaneval',
                version='dcae0e',
                metric='pass@1',
                mode='gen',
                model_alias='17.5',
            ),
            dict(
                dataset='openai_humaneval',
                version='dcae0e',
                metric='pass@100',
                mode='gen',
                model_alias='88.0',
            ),
        ],
    )

    results = module.collect_general_results(str(results_dir))

    assert results[checkpoint]['openai_humaneval'] == {
        'score': 17.5,
        'metric': 'pass@1',
    }


def test_compute_general_avg_groups_new_opencompass_benchmarks():
    module = _load_module()
    general_results = {
        'IFEval': {'score': 71.0, 'metric': 'Inst-level-strict-accuracy'},
        'bbh-temporal_sequences': {'score': 60.0, 'metric': 'score'},
        'bbh-word_sorting': {'score': 80.0, 'metric': 'score'},
        'nq': {'score': 40.0, 'metric': 'score'},
        'lukaemon_mmlu_college_biology': {'score': 50.0, 'metric': 'accuracy'},
        'lukaemon_mmlu_machine_learning': {'score': 70.0, 'metric': 'accuracy'},
        'gsm8k': {'score': 65.0, 'metric': 'score'},
        'ARC-c': {'score': 55.0, 'metric': 'accuracy'},
        'hellaswag': {'score': 75.0, 'metric': 'accuracy'},
        'winogrande': {'score': 68.0, 'metric': 'accuracy'},
        'XCOPA': {'score': 72.0, 'metric': 'accuracy'},
        'openai_humaneval': {'score': 20.0, 'metric': 'pass@1'},
    }

    avg = module.compute_general_avg(general_results)

    assert avg['ifeval'] == 71.0
    assert avg['bbh_avg'] == 70.0
    assert avg['nq'] == 40.0
    assert avg['mmlu_avg'] == 60.0
    assert avg['gsm8k'] == 65.0
    assert avg['arc'] == 55.0
    assert avg['hellaswag'] == 75.0
    assert avg['winogrande'] == 68.0
    assert avg['xcopa'] == 72.0
    assert avg['humaneval'] == 20.0
    assert avg['components_used'] == [
        'ifeval',
        'bbh_avg',
        'nq',
        'mmlu_avg',
        'gsm8k',
        'arc',
        'hellaswag',
        'winogrande',
        'xcopa',
        'humaneval',
    ]
    assert avg['avg_score'] == 59.6
