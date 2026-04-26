import sys
from pathlib import Path


ARENA_DIR = Path(__file__).resolve().parents[1]
VERL_DIR = ARENA_DIR / "third_party" / "verl"
for path in (ARENA_DIR, VERL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from verl.utils.fsdp_utils import normalize_peft_param_name


def test_normalize_peft_param_name_drops_oft_adapter_tensors():
    params = {
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": 1,
        "base_model.model.model.layers.0.self_attn.q_proj.oft_R.default.weight": 2,
        "base_model.model.model.layers.0.self_attn.q_proj.oft_dropout.default.p": 3,
    }

    normalized = normalize_peft_param_name(params)

    assert normalized == {
        "model.layers.0.self_attn.q_proj.weight": 1,
    }
