#!/bin/bash
# =============================================================================
# PEFTArena Environment Setup Script
# =============================================================================
# Installs the Python-side dependencies required by the release workflows on
# top of an existing Python + PyTorch environment.
#
# Usage:
#   bash setup_env.sh
#
# Optional environment variables:
#   PYTHON_BIN=python3
#   OPENCOMPASS_DATA_AUTO_DOWNLOAD=0
#   INSTALL_LIGER_KERNEL=0
#   INSTALL_VLLM=1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="${SCRIPT_DIR}/third_party"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_LIGER_KERNEL="${INSTALL_LIGER_KERNEL:-1}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"

pip_install() {
    "$PYTHON_BIN" -m pip install "$@"
}

require_dir() {
    local path="$1"
    if [ ! -d "$path" ]; then
        echo "[setup] ERROR: required directory not found: ${path}"
        exit 1
    fi
}

echo "============================================"
echo "[PEFTArena] Environment Setup"
echo "============================================"

echo "[setup] Using Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "[setup] Python version: $("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])')"

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    echo "[setup] WARNING: torch is not installed in the current environment."
    echo "[setup] Install a CUDA-compatible torch build before running training or vLLM evaluation."
fi

echo "[setup] Validating bundled third-party components..."
bash "${SCRIPT_DIR}/setup_third_party.sh"

require_dir "${THIRD_PARTY}/math_eval/latex2sympy"
require_dir "${THIRD_PARTY}/opencompass"
require_dir "${THIRD_PARTY}/verl"
require_dir "${THIRD_PARTY}/human-eval"

echo "[setup] Installing shared release dependencies..."
pip_install "setuptools<70"
pip_install \
    psutil \
    pandas \
    matplotlib \
    nltk \
    pebble \
    sympy \
    word2number \
    "antlr4-python3-runtime==4.9.3"

echo "[setup] Installing math_eval parser package..."
pip_install -e "${THIRD_PARTY}/math_eval/latex2sympy"

echo "[setup] Installing VeRL..."
pip_install -e "${THIRD_PARTY}/verl"

if [ "$INSTALL_VLLM" = "1" ]; then
    echo "[setup] Installing vLLM..."
    pip_install "vllm==0.11.0"
else
    echo "[setup] Skipping vLLM (INSTALL_VLLM=0)."
fi

echo "[setup] Installing OpenCompass runtime requirements..."
pip_install -r "${THIRD_PARTY}/opencompass/requirements/runtime.txt"
echo "[setup] Installing OpenCompass..."
pip_install -e "${THIRD_PARTY}/opencompass"

if [ ! -d "${THIRD_PARTY}/opencompass/data" ]; then
    echo "[setup] NOTE: OpenCompass benchmark datasets are not bundled in this release tree."
    echo "[setup] NOTE: many datasets download automatically on first use, but benchmarks such as"
    echo "[setup] NOTE: IFEval and MMLU still expect local dataset preparation under"
    echo "[setup] NOTE: ${THIRD_PARTY}/opencompass/data."
fi

echo "[setup] Installing human-eval..."
pip_install -e "${THIRD_PARTY}/human-eval"

if [ -f "${THIRD_PARTY}/human-eval/evalplus/setup.py" ] || \
   [ -f "${THIRD_PARTY}/human-eval/evalplus/pyproject.toml" ]; then
    echo "[setup] Installing vendored evalplus..."
    pip_install -e "${THIRD_PARTY}/human-eval/evalplus"
else
    echo "[setup] Vendored evalplus not present; installing evalplus from PyPI..."
    pip_install evalplus
fi

if [ "$INSTALL_LIGER_KERNEL" = "1" ]; then
    echo "[setup] Installing liger-kernel..."
    pip_install liger-kernel
else
    echo "[setup] Skipping liger-kernel (INSTALL_LIGER_KERNEL=0)."
fi

echo "[setup] Installing wandb..."
pip_install wandb

echo "[setup] Downloading NLTK tokenizer data..."
"$PYTHON_BIN" - <<'PY'
import nltk

nltk.download("punkt_tab", quiet=False)
PY

echo "[setup] Re-pinning antlr4 runtime for OmegaConf + patched latex2sympy..."
pip_install --force-reinstall "antlr4-python3-runtime==4.9.3"

echo ""
echo "============================================"
echo "[PEFTArena] Environment setup complete"
echo "============================================"
echo "You can now use:"
echo "  python run.py --help"
