import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
MODULE_PATH = REPO_ROOT / 'eval' / 'generate_opencompass_config.py'
ARENA_DATASETS_DIR = (
    REPO_ROOT / 'eval' / 'opencompass_configs' / 'datasets')
OPENCOMPASS_DATASETS_DIR = (
    REPO_ROOT / 'third_party' / 'opencompass' / 'opencompass' / 'configs' /
    'datasets')
if not (OPENCOMPASS_DATASETS_DIR / 'nq' / 'nq_gen_c788f6.py').exists():
    OPENCOMPASS_DATASETS_DIR = (
        WORKSPACE_ROOT / 'opencompass' / 'opencompass' / 'configs' /
        'datasets')


def _load_module():
    spec = importlib.util.spec_from_file_location(
        'generate_opencompass_config', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._get_oc_datasets_dir = lambda: OPENCOMPASS_DATASETS_DIR
    return module


def _write_model_config(model_dir: Path, model_name: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'config.json').write_text(
        json.dumps({'_name_or_path': model_name}))


def test_generate_split_configs_inline_dataset_sources(tmp_path):
    module = _load_module()
    model_dir = tmp_path / 'models' / 'qwen-base'
    _write_model_config(model_dir, 'Qwen/Qwen2.5-7B')

    output_config = tmp_path / 'configs' / 'arena.py'
    configs = module.generate_config(
        model_path=str(model_dir),
        benchmarks=['bbh', 'ifeval_nq'],
        output_config=str(output_config),
        abbr='qwen_base',
    )

    assert configs == [
        str(output_config).replace('.py', '_bbh.py'),
        str(output_config).replace('.py', '_ifeval_nq.py'),
    ]

    bbh_text = Path(configs[0]).read_text()
    ifeval_text = Path(configs[1]).read_text()

    assert 'Config.fromfile' not in bbh_text
    assert 'Config.fromfile' not in ifeval_text
    assert '_oc_cfg' not in bbh_text
    assert '_oc_cfg' not in ifeval_text
    assert 'os.path.dirname(__file__)' not in bbh_text
    assert str(ARENA_DATASETS_DIR / 'bbh') in bbh_text
    assert 'datasets = bbh_datasets' in bbh_text
    assert 'datasets = ifeval_datasets + nq_datasets' in ifeval_text
    assert "type='opencompass.models.VLLM'" in bbh_text
    assert "type='opencompass.models.VLLMwithChatTemplate'" in ifeval_text
    assert (
        str(ARENA_DATASETS_DIR / 'nq') in ifeval_text
        or str(OPENCOMPASS_DATASETS_DIR / 'nq') in ifeval_text
    )


def test_generated_bbh_config_roundtrips_through_mmengine_dump(tmp_path):
    mmengine_config = pytest.importorskip('mmengine.config')
    module = _load_module()
    model_dir = tmp_path / 'models' / 'qwen-base'
    _write_model_config(model_dir, 'Qwen/Qwen2.5-7B')

    output_config = tmp_path / 'configs' / 'arena.py'
    configs = module.generate_config(
        model_path=str(model_dir),
        benchmarks=['bbh'],
        output_config=str(output_config),
        abbr='qwen_base',
    )

    dumped_config = tmp_path / 'configs' / 'roundtrip.py'
    cfg = mmengine_config.Config.fromfile(configs[0], format_python_code=False)
    cfg.dump(dumped_config)
    reloaded = mmengine_config.Config.fromfile(
        dumped_config, format_python_code=False)

    assert len(reloaded.datasets) == len(cfg.datasets)


def test_generate_single_config_inlines_all_dataset_sources(tmp_path):
    module = _load_module()
    model_dir = tmp_path / 'models' / 'qwen-instruct'
    _write_model_config(model_dir, 'Qwen/Qwen2.5-7B-Instruct')

    output_config = tmp_path / 'configs' / 'arena.py'
    configs = module.generate_config(
        model_path=str(model_dir),
        benchmarks=['bbh', 'ifeval_nq'],
        output_config=str(output_config),
        abbr='qwen_instruct',
    )

    assert configs == [str(output_config)]

    text = output_config.read_text()
    assert 'Config.fromfile' not in text
    assert '_oc_cfg' not in text
    assert 'bbh_datasets = []' in text
    assert 'ifeval_datasets = [' in text
    assert 'nq_datasets = [' in text
    assert 'datasets = bbh_datasets + ifeval_datasets + nq_datasets' in text
    assert "type='opencompass.models.VLLMwithChatTemplate'" in text


def test_generate_single_config_for_new_benchmarks(tmp_path):
    module = _load_module()
    model_dir = tmp_path / 'models' / 'qwen-instruct'
    _write_model_config(model_dir, 'Qwen/Qwen2.5-7B-Instruct')

    output_config = tmp_path / 'configs' / 'arena.py'
    configs = module.generate_config(
        model_path=str(model_dir),
        benchmarks=['humaneval', 'hellaswag', 'mmlu', 'gsm8k'],
        output_config=str(output_config),
        abbr='qwen_instruct',
    )

    assert configs == [str(output_config)]

    text = output_config.read_text()
    assert 'humaneval_datasets = [' in text
    assert 'hellaswag_datasets = [' in text
    assert 'mmlu_datasets = []' in text
    assert 'gsm8k_datasets = [' in text
    assert ('datasets = humaneval_datasets + hellaswag_datasets + '
            'mmlu_datasets + gsm8k_datasets') in text
    assert "type='opencompass.models.VLLMwithChatTemplate'" in text
    assert 'Config.fromfile' not in text


def test_generate_split_config_for_base_model_with_new_benchmarks(tmp_path):
    module = _load_module()
    model_dir = tmp_path / 'models' / 'qwen-base'
    _write_model_config(model_dir, 'Qwen/Qwen2.5-7B')

    output_config = tmp_path / 'configs' / 'arena.py'
    configs = module.generate_config(
        model_path=str(model_dir),
        benchmarks=['humaneval', 'xcopa', 'ifeval_nq'],
        output_config=str(output_config),
        abbr='qwen_base',
    )

    assert configs == [
        str(output_config).replace('.py', '_humaneval_xcopa.py'),
        str(output_config).replace('.py', '_ifeval_nq.py'),
    ]

    standard_text = Path(configs[0]).read_text()
    chat_text = Path(configs[1]).read_text()

    assert 'humaneval_datasets = [' in standard_text
    assert 'XCOPA_datasets = [' in standard_text
    assert "type='opencompass.models.VLLM'" in standard_text
    assert 'ifeval_datasets = [' in chat_text
    assert 'nq_datasets = [' in chat_text
    assert "type='opencompass.models.VLLMwithChatTemplate'" in chat_text
