#!/usr/bin/env python3
"""Auto-generate OpenCompass evaluation configs for PEFTArena.

This script generates standalone OpenCompass configs based on:
- selected benchmarks
- whether the model is base vs instruct/chat
- whether the selected benchmark set requires mixed model wrappers

The current benchmark set covers:
- `bbh`
- `ifeval_nq`
- `humaneval`
- `hellaswag`
- `winogrande`
- `mmlu`
- `arc`
- `gsm8k`
- `xcopa`
"""

import argparse
import json
import os
from pathlib import Path


MODEL_CLASS_VLLM = 'opencompass.models.VLLM'
MODEL_CLASS_CHAT = 'opencompass.models.VLLMwithChatTemplate'

# Known instruct model identifiers (case-insensitive substring match)
INSTRUCT_INDICATORS = [
    'instruct', '-ins', 'ins-', 'chat', 'it-',
]


def is_instruct_model(model_path: str) -> bool:
    """Auto-detect whether a model is an instruct/chat model based on its path or config."""
    path_lower = model_path.lower()

    for indicator in INSTRUCT_INDICATORS:
        if indicator in path_lower:
            return True

    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get('_name_or_path', '').lower()
            for indicator in INSTRUCT_INDICATORS:
                if indicator in model_type:
                    return True
        except Exception:
            pass

    return False


def find_base_model_path(checkpoint_path: str) -> str:
    """Try to find the base model path from a checkpoint or adapter config."""
    adapter_config = os.path.join(checkpoint_path, 'adapter_config.json')
    if os.path.exists(adapter_config):
        with open(adapter_config) as f:
            config = json.load(f)
        return config.get('base_model_name_or_path', '')

    model_config = os.path.join(checkpoint_path, 'config.json')
    if os.path.exists(model_config):
        with open(model_config) as f:
            config = json.load(f)
        return config.get('_name_or_path', checkpoint_path)

    return checkpoint_path


def _get_oc_datasets_dir() -> Path:
    """Find the OpenCompass datasets config directory at generation time."""
    repo_oc_datasets_dir = (
        Path(__file__).resolve().parents[1] / 'third_party' / 'opencompass' /
        'opencompass' / 'configs' / 'datasets')
    if repo_oc_datasets_dir.exists():
        return repo_oc_datasets_dir

    workspace_oc_datasets_dir = (
        Path(__file__).resolve().parents[2] / 'opencompass' / 'opencompass' /
        'configs' / 'datasets')
    if workspace_oc_datasets_dir.exists():
        return workspace_oc_datasets_dir

    try:
        import opencompass
    except ImportError:
        raise

    if getattr(opencompass, '__file__', None):
        return Path(opencompass.__file__).resolve().parent / 'configs' / 'datasets'

    raise FileNotFoundError(
        'Unable to resolve OpenCompass dataset config directory from installed '
        'package or bundled repository copy.')


def _get_arena_datasets_dir() -> Path:
    """Return PEFTArena-owned OpenCompass dataset config copies."""
    return Path(__file__).resolve().parent / 'opencompass_configs' / 'datasets'


def _get_dataset_config_path(relative_path: tuple[str, ...]) -> Path:
    """Resolve a dataset config, preferring PEFTArena-local copies."""
    arena_config_path = _get_arena_datasets_dir().joinpath(*relative_path)
    if arena_config_path.exists():
        return arena_config_path

    oc_config_path = _get_oc_datasets_dir().joinpath(*relative_path)
    if oc_config_path.exists():
        return oc_config_path

    raise FileNotFoundError(
        'OpenCompass dataset config not found in PEFTArena or OpenCompass: '
        f'{" / ".join(relative_path)}')


def _rewrite_bbh_source(source: str, config_path: Path) -> str:
    """Rewrite BBH prompt lookups so inlined configs work outside the source tree."""
    source = source.replace('os.path.dirname(__file__)',
                            repr(str(config_path.parent)))
    source = source.replace("__file__.rsplit('/', 1)[0] + '/lib_prompt'",
                            repr(str(config_path.parent / 'lib_prompt')))
    cleanup = (
        "\nfor _tmp in ('_name', '_hint', 'bbh_infer_cfg', 'bbh_eval_cfg',\n"
        "             'f', '_prompt_file', 'os', 'PromptTemplate',\n"
        "             'ZeroRetriever', 'GenInferencer', 'AccEvaluator',\n"
        "             'BBHDataset', 'BBHEvaluator', 'bbh_mcq_postprocess',\n"
        "             'BBHEvaluator_mcq', 'bbh_reader_cfg',\n"
        "             'bbh_multiple_choice_sets', 'bbh_free_form_sets',\n"
        "             'bbh_prompt_dir'):\n"
        "    globals().pop(_tmp, None)\n"
        "del _tmp\n"
    )
    if cleanup.strip() not in source:
        source = f'{source.rstrip()}\n{cleanup}'
    return source


def _inline_dataset_config(relative_path: tuple[str, ...],
                           section_title: str,
                           rewrite_source=None) -> str:
    """Load and inline an OpenCompass dataset config into the generated config."""
    config_path = _get_dataset_config_path(relative_path)
    source = config_path.read_text(encoding='utf-8')
    if rewrite_source is not None:
        source = rewrite_source(source, config_path)

    return (
        f'\n# {section_title} datasets - inlined into PEFTArena config\n'
        f'# source: {config_path}\n'
        f'{source.rstrip()}\n'
    )


def generate_bbh_datasets_source():
    return _inline_dataset_config(
        ('bbh', 'bbh_gen_5b92b0.py'),
        'BBH',
        rewrite_source=_rewrite_bbh_source,
    )


def generate_ifeval_nq_datasets_source():
    return ''.join([
        _inline_dataset_config(('IFEval', 'IFEval_gen_3321a3.py'), 'IFEval'),
        _inline_dataset_config(('nq', 'nq_gen_c788f6.py'), 'NQ'),
    ])


def generate_humaneval_datasets_source():
    return _inline_dataset_config(
        ('humaneval', 'humaneval_openai_sample_evals_gen_dcae0e.py'),
        'HumanEval',
    )


def generate_hellaswag_datasets_source():
    return _inline_dataset_config(
        ('hellaswag', 'hellaswag_10shot_gen_e42710.py'),
        'HellaSwag',
    )


def generate_winogrande_datasets_source():
    return _inline_dataset_config(
        ('winogrande', 'winogrande_gen_458220.py'),
        'WinoGrande',
    )


def generate_mmlu_datasets_source():
    return _inline_dataset_config(
        ('mmlu', 'mmlu_gen_4d595a.py'),
        'MMLU',
    )


def generate_arc_datasets_source():
    return _inline_dataset_config(
        ('ARC_c', 'ARC_c_gen_1e0de5.py'),
        'ARC-Challenge',
    )


def generate_gsm8k_datasets_source():
    return _inline_dataset_config(
        ('gsm8k', 'gsm8k_gen_1d7fe4.py'),
        'GSM8K',
    )


def generate_xcopa_datasets_source():
    return _inline_dataset_config(
        ('XCOPA', 'XCOPA_ppl_54058d.py'),
        'XCOPA',
    )


def _base_or_chat_model_class(is_instruct: bool) -> str:
    return MODEL_CLASS_CHAT if is_instruct else MODEL_CLASS_VLLM


def _chat_only_model_class(is_instruct: bool) -> str:
    del is_instruct
    return MODEL_CLASS_CHAT


BENCHMARK_SPECS = {
    'bbh': dict(
        dataset_source=generate_bbh_datasets_source,
        dataset_vars=['bbh_datasets'],
        model_class=_base_or_chat_model_class,
        max_out_len=512,
    ),
    'ifeval_nq': dict(
        dataset_source=generate_ifeval_nq_datasets_source,
        dataset_vars=['ifeval_datasets', 'nq_datasets'],
        model_class=_chat_only_model_class,
        max_out_len=1025,
    ),
    'humaneval': dict(
        dataset_source=generate_humaneval_datasets_source,
        dataset_vars=['humaneval_datasets'],
        model_class=_base_or_chat_model_class,
        max_out_len=1024,
    ),
    'hellaswag': dict(
        dataset_source=generate_hellaswag_datasets_source,
        dataset_vars=['hellaswag_datasets'],
        model_class=_base_or_chat_model_class,
        max_out_len=32,
    ),
    'winogrande': dict(
        dataset_source=generate_winogrande_datasets_source,
        dataset_vars=['winogrande_datasets'],
        model_class=_base_or_chat_model_class,
        max_out_len=32,
    ),
    'mmlu': dict(
        dataset_source=generate_mmlu_datasets_source,
        dataset_vars=['mmlu_datasets'],
        model_class=_base_or_chat_model_class,
        max_out_len=32,
    ),
    'arc': dict(
        dataset_source=generate_arc_datasets_source,
        dataset_vars=['ARC_c_datasets'],
        model_class=_base_or_chat_model_class,
        max_out_len=32,
    ),
    'gsm8k': dict(
        dataset_source=generate_gsm8k_datasets_source,
        dataset_vars=['gsm8k_datasets'],
        model_class=_base_or_chat_model_class,
        max_out_len=512,
    ),
    'xcopa': dict(
        dataset_source=generate_xcopa_datasets_source,
        dataset_vars=['XCOPA_datasets'],
        model_class=_base_or_chat_model_class,
        max_out_len=32,
    ),
}

BENCHMARK_ALIASES = {
    'arc_c': 'arc',
}


def normalize_benchmarks(benchmarks: list[str]) -> list[str]:
    """Normalize, validate, and deduplicate benchmark names."""
    normalized = []
    seen = set()
    for benchmark in benchmarks:
        key = BENCHMARK_ALIASES.get(benchmark.strip().lower(),
                                    benchmark.strip().lower())
        if not key:
            continue
        if key not in BENCHMARK_SPECS:
            valid = ', '.join(sorted(BENCHMARK_SPECS))
            raise ValueError(
                f'Unknown benchmark "{benchmark}". Valid choices: {valid}')
        if key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    if not normalized:
        raise ValueError('No benchmarks selected.')
    return normalized


def generate_model_config(model_path: str, abbr: str, model_class: str,
                          num_gpus: int = 1, batch_size: int = 256,
                          max_out_len: int = 1024) -> str:
    """Generate a model config dict string."""
    return f"""dict(
        abbr='{abbr}',
        type='{model_class}',
        path='{model_path}',
        model_kwargs=dict(
            tensor_parallel_size=1,
            trust_remote_code=True,
        ),
        generation_kwargs=dict(temperature=0),
        batch_size={batch_size},
        max_out_len={max_out_len},
        run_cfg=dict(num_gpus={num_gpus}),
    )"""


def _write_config_file(output_path: str,
                       dataset_source: str,
                       model_cfg: str,
                       dataset_expr: str) -> None:
    """Write a standalone OpenCompass evaluation config."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(dataset_source.rstrip())
        f.write(f"\n\nmodels = [\n    {model_cfg}\n]\n")
        f.write(f"\ndatasets = {dataset_expr}\n")


def generate_config(
    model_path: str,
    benchmarks: list[str],
    output_config: str,
    abbr: str,
    num_gpus: int = 1,
    batch_size: int = 256,
):
    """Generate one or more standalone OpenCompass configs."""
    benchmarks = normalize_benchmarks(benchmarks)
    is_instruct = is_instruct_model(model_path)
    base_model = find_base_model_path(model_path)
    is_instruct = is_instruct or is_instruct_model(base_model)

    print(f'[config-gen] Model path:     {model_path}')
    print(f'[config-gen] Base model:     {base_model}')
    print(f'[config-gen] Is instruct:    {is_instruct}')
    print(f'[config-gen] Benchmarks:     {benchmarks}')

    groups = []
    group_index_by_class = {}
    for benchmark in benchmarks:
        spec = BENCHMARK_SPECS[benchmark]
        model_class = spec['model_class'](is_instruct)
        if model_class not in group_index_by_class:
            group_index_by_class[model_class] = len(groups)
            groups.append(dict(
                model_class=model_class,
                benchmarks=[],
                dataset_sections=[],
                dataset_vars=[],
                max_out_len=0,
            ))
        group = groups[group_index_by_class[model_class]]
        group['benchmarks'].append(benchmark)
        group['dataset_sections'].append(spec['dataset_source']())
        group['dataset_vars'].extend(spec['dataset_vars'])
        group['max_out_len'] = max(group['max_out_len'], spec['max_out_len'])
        print(f'[config-gen] {benchmark} model class: {model_class}')

    if len(groups) == 1:
        group = groups[0]
        model_cfg = generate_model_config(
            model_path,
            abbr,
            group['model_class'],
            num_gpus,
            batch_size,
            max_out_len=group['max_out_len'],
        )
        _write_config_file(
            output_config,
            ''.join(group['dataset_sections']),
            model_cfg,
            ' + '.join(group['dataset_vars']),
        )
        print(f'[config-gen] Config → {output_config}')
        return [output_config]

    print('[config-gen] WARNING: benchmark set requires mixed model wrappers.')
    print('[config-gen] Generating one config per model-class group.')

    generated_configs = []
    for group in groups:
        suffix = '_'.join(group['benchmarks'])
        grouped_output = output_config.replace('.py', f'_{suffix}.py')
        model_cfg = generate_model_config(
            model_path,
            abbr,
            group['model_class'],
            num_gpus,
            batch_size,
            max_out_len=group['max_out_len'],
        )
        _write_config_file(
            grouped_output,
            ''.join(group['dataset_sections']),
            model_cfg,
            ' + '.join(group['dataset_vars']),
        )
        print(f'[config-gen] Config → {grouped_output}')
        generated_configs.append(grouped_output)

    return generated_configs


def main():
    parser = argparse.ArgumentParser(
        description='Generate OpenCompass config for PEFTArena eval')
    parser.add_argument('--model_path', required=True,
                        help='Path to (merged) model')
    parser.add_argument(
        '--benchmarks',
        default='bbh,ifeval_nq',
        help=('Comma-separated benchmark names: '
              'bbh, ifeval_nq, humaneval, hellaswag, winogrande, '
              'mmlu, arc, gsm8k, xcopa'),
    )
    parser.add_argument('--output_config', required=True,
                        help='Output .py config file')
    parser.add_argument('--abbr', default=None,
                        help='Model abbreviation for OpenCompass')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(',')]

    if args.abbr is None:
        parts = args.model_path.rstrip('/').split('/')
        args.abbr = '_'.join(parts[-2:]).replace('global_step_', '')

    configs = generate_config(
        model_path=args.model_path,
        benchmarks=benchmarks,
        output_config=args.output_config,
        abbr=args.abbr,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
    )

    print(f'\n[config-gen] Generated {len(configs)} config file(s):')
    for config_path in configs:
        print(f'  {config_path}')


if __name__ == '__main__':
    main()
