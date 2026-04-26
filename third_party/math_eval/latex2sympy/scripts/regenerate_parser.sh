#!/bin/bash
# Regenerate ANTLR parser files for latex2sympy using antlr4 4.9.3
# This ensures compatibility with omegaconf/hydra which require antlr4-python3-runtime==4.9.3
#
# Prerequisites: Java must be installed (java command available)
#
# Usage: cd to the latex2sympy root dir and run:
#   bash scripts/regenerate_parser.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

ANTLR_VERSION="4.9.3"
ANTLR_JAR="${PROJECT_DIR}/antlr-${ANTLR_VERSION}-complete.jar"
ANTLR_URL="https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar"

# Download the ANTLR jar if not present
if [ ! -f "${ANTLR_JAR}" ]; then
    echo "[regenerate] Downloading antlr-${ANTLR_VERSION}-complete.jar..."
    wget -q "${ANTLR_URL}" -O "${ANTLR_JAR}" || curl -sL "${ANTLR_URL}" -o "${ANTLR_JAR}"
fi

echo "[regenerate] Using ANTLR ${ANTLR_VERSION}"

# Back up existing gen/ directory
if [ -d "${PROJECT_DIR}/gen" ]; then
    echo "[regenerate] Backing up existing gen/ to gen.bak/"
    rm -rf "${PROJECT_DIR}/gen.bak"
    cp -r "${PROJECT_DIR}/gen" "${PROJECT_DIR}/gen.bak"
fi

# Regenerate parser
echo "[regenerate] Generating parser from PS.g4..."
cd "${PROJECT_DIR}"
java -jar "${ANTLR_JAR}" PS.g4 -o gen

echo "[regenerate] Done! Parser files regenerated in gen/"
echo "[regenerate] Old files backed up in gen.bak/"
echo ""
echo "Next steps:"
echo "  1. Verify: pip install antlr4-python3-runtime==4.9.3"
echo "  2. Test:   python -c 'from latex2sympy.latex2sympy2 import latex2sympy; print(latex2sympy(\"x^2 + 1\"))'"
