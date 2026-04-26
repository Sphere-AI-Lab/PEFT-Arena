#!/bin/bash
# =============================================================================
# PEFTArena Third-Party Setup Script
# =============================================================================
# Validates the bundled third-party trees and backfills them from common local
# sources when a required component is missing or incomplete.
#
# Usage:
#   bash setup_third_party.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="${SCRIPT_DIR}/third_party"
PYTHON_BIN="${PYTHON_BIN:-python}"
OPENCOMPASS_DATA_AUTO_DOWNLOAD="${OPENCOMPASS_DATA_AUTO_DOWNLOAD:-1}"

mkdir -p "$THIRD_PARTY"

echo "============================================"
echo "[PEFTArena] Third-Party Setup"
echo "============================================"

check_paths() {
    local base="$1"
    shift
    local rel
    for rel in "$@"; do
        if [ ! -e "${base}/${rel}" ]; then
            return 1
        fi
    done
    return 0
}

sync_tree() {
    local label="$1"
    local src="$2"
    local dst="$3"
    local item
    local base

    echo "[setup] Syncing ${label} from ${src}..."
    mkdir -p "$dst"
    shopt -s dotglob nullglob
    for item in "${src}"/*; do
        base="$(basename "$item")"
        if [ "$base" = ".git" ]; then
            continue
        fi
        cp -a "$item" "${dst}/"
    done
    shopt -u dotglob nullglob
}

clone_into_tree() {
    local label="$1"
    local url="$2"
    local dst="$3"
    local tmp_dir

    tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/peftarena_${label}_XXXXXX")"
    echo "[setup] Cloning ${label} from ${url}..."
    git clone "$url" "$tmp_dir"
    sync_tree "$label" "$tmp_dir" "$dst"
    rm -rf "$tmp_dir"
}

ensure_verl() {
    local dst="${THIRD_PARTY}/verl"
    local src
    local required=("setup.py" "verl" "recipe")
    local sources=(
        "${SCRIPT_DIR}/../peft_arena/third_party/verl"
    )

    if check_paths "$dst" "${required[@]}"; then
        echo "[setup] VeRL ready at ${dst}"
        return
    fi

    echo "[setup] VeRL is missing or incomplete at ${dst}"
    for src in "${sources[@]}"; do
        if [ -d "$src" ] && check_paths "$src" "${required[@]}"; then
            sync_tree "VeRL" "$src" "$dst"
            break
        fi
    done

    if ! check_paths "$dst" "${required[@]}"; then
        clone_into_tree "VeRL" "https://github.com/volcengine/verl.git" "$dst"
    fi

    check_paths "$dst" "${required[@]}" || {
        echo "[setup] ERROR: VeRL is still incomplete at ${dst}"
        exit 1
    }
}

ensure_math_eval() {
    local dst="${THIRD_PARTY}/math_eval"
    local src
    local required=("math_eval.py" "parser.py" "data" "latex2sympy/setup.py")
    local sources=(
        "${SCRIPT_DIR}/../peft_arena/third_party/math_eval"
        "${SCRIPT_DIR}/../limit-of-rl/math/examples/math_eval"
    )

    if check_paths "$dst" "${required[@]}"; then
        echo "[setup] math_eval ready at ${dst}"
        return
    fi

    echo "[setup] math_eval is missing or incomplete at ${dst}"
    for src in "${sources[@]}"; do
        if [ -d "$src" ] && check_paths "$src" "${required[@]}"; then
            sync_tree "math_eval" "$src" "$dst"
            break
        fi
    done

    check_paths "$dst" "${required[@]}" || {
        echo "[setup] ERROR: math_eval is still incomplete at ${dst}"
        echo "[setup] Expected: ${required[*]}"
        exit 1
    }
}

ensure_med_eval() {
    local dst="${THIRD_PARTY}/med_eval"
    local src
    local required=("main.py" "data/m1_eval_data_processed.json" "utils/extract_answer.py")
    local sources=(
        "${SCRIPT_DIR}/../peft_arena/third_party/med_eval"
        "${SCRIPT_DIR}/../med_eval_ours"
    )

    if check_paths "$dst" "${required[@]}"; then
        echo "[setup] med_eval ready at ${dst}"
        return
    fi

    echo "[setup] med_eval is missing or incomplete at ${dst}"
    for src in "${sources[@]}"; do
        if [ -d "$src" ] && check_paths "$src" "${required[@]}"; then
            sync_tree "med_eval" "$src" "$dst"
            break
        fi
    done

    check_paths "$dst" "${required[@]}" || {
        echo "[setup] ERROR: med_eval is still incomplete at ${dst}"
        echo "[setup] Expected: ${required[*]}"
        exit 1
    }
}

ensure_opencompass() {
    local dst="${THIRD_PARTY}/opencompass"
    local src
    local required=("requirements/runtime.txt" "opencompass" "opencompass/configs/datasets")
    local sources=(
        "${SCRIPT_DIR}/../peft_arena/third_party/opencompass"
    )

    if check_paths "$dst" "${required[@]}"; then
        echo "[setup] OpenCompass ready at ${dst}"
    else
        echo "[setup] OpenCompass is missing or incomplete at ${dst}"
        for src in "${sources[@]}"; do
            if [ -d "$src" ] && check_paths "$src" "${required[@]}"; then
                sync_tree "OpenCompass" "$src" "$dst"
                break
            fi
        done

        if ! check_paths "$dst" "${required[@]}"; then
            clone_into_tree "OpenCompass" "https://github.com/open-compass/opencompass.git" "$dst"
        fi
    fi

    check_paths "$dst" "${required[@]}" || {
        echo "[setup] ERROR: OpenCompass is still incomplete at ${dst}"
        exit 1
    }

    if [ ! -d "${dst}/data" ]; then
        if [ "$OPENCOMPASS_DATA_AUTO_DOWNLOAD" = "1" ]; then
            echo "[setup] NOTE: bundled OpenCompass does not ship a single download script in this version."
            echo "[setup] NOTE: many datasets are fetched automatically on first use, but some benchmarks"
            echo "[setup] NOTE: such as IFEval and MMLU still require manual dataset preparation under"
            echo "[setup] NOTE: ${dst}/data according to the upstream OpenCompass dataset docs."
        else
            echo "[setup] OpenCompass dataset preparation skipped (OPENCOMPASS_DATA_AUTO_DOWNLOAD=0)."
        fi
    fi
}

ensure_human_eval() {
    local dst="${THIRD_PARTY}/human-eval"
    local src
    local required=("setup.py" "requirements.txt" "human_eval/data/HumanEval.jsonl.gz")
    local sources=(
        "${SCRIPT_DIR}/../peft_arena/third_party/human-eval"
    )

    if check_paths "$dst" "${required[@]}"; then
        echo "[setup] human-eval ready at ${dst}"
        return
    fi

    echo "[setup] human-eval is missing or incomplete at ${dst}"
    for src in "${sources[@]}"; do
        if [ -d "$src" ] && check_paths "$src" "${required[@]}"; then
            sync_tree "human-eval" "$src" "$dst"
            break
        fi
    done

    if ! check_paths "$dst" "${required[@]}"; then
        clone_into_tree "human-eval" "https://github.com/openai/human-eval.git" "$dst"
    fi

    check_paths "$dst" "${required[@]}" || {
        echo "[setup] ERROR: human-eval is still incomplete at ${dst}"
        exit 1
    }
}

ensure_verl
ensure_math_eval
ensure_med_eval
ensure_opencompass
ensure_human_eval

echo ""
echo "============================================"
echo "[PEFTArena] Third-party setup complete"
echo "============================================"
echo "  - VeRL:         ${THIRD_PARTY}/verl"
echo "  - math_eval:    ${THIRD_PARTY}/math_eval"
echo "  - med_eval:     ${THIRD_PARTY}/med_eval"
echo "  - OpenCompass:  ${THIRD_PARTY}/opencompass"
echo "  - human-eval:   ${THIRD_PARTY}/human-eval"
