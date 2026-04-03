#!/bin/bash
# =============================================================================
# CUTLASS FP8 Patch for GB10 (sm_121)
# =============================================================================
# Disables CUTLASS FP8 kernels which are not prebuilt for sm_121.
# vLLM falls back to Triton-based FP8 kernels which support sm_121.
#
# Usage: ./cutlass-fp8-sm121.sh [/path/to/vllm-env]
# =============================================================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[OK]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

VENV="${1:-$HOME/vllm-env}"
PYTHON="$VENV/bin/python"

# Find vLLM package path
PYVER=$($PYTHON --version 2>&1 | grep -oP '3\.\d+')
VLLM_PKG="$VENV/lib/python${PYVER}/site-packages/vllm"
[ -d "$VLLM_PKG" ] || err "vLLM not found at $VLLM_PKG"

TARGET="$VLLM_PKG/model_executor/layers/quantization/utils/w8a8_utils.py"
[ -f "$TARGET" ] || err "w8a8_utils.py not found at $TARGET"

# Check if already patched
if grep -q 'return False  # Patched: sm_121' "$TARGET" 2>/dev/null; then
    log "CUTLASS FP8 already patched"
    exit 0
fi

# Backup
cp "$TARGET" "${TARGET}.bak"

# Patch functions to return False
python3 << PYEOF
import re

with open("$TARGET") as f:
    content = f.read()

# Patch cutlass_fp8_supported()
content = re.sub(
    r'(def cutlass_fp8_supported\(\)[^:]*:\n)',
    r'\1    return False  # Patched: sm_121 not in prebuilt CUTLASS\n',
    content
)

# Patch cutlass_block_fp8_supported()
content = re.sub(
    r'(def cutlass_block_fp8_supported\(\)[^:]*:\n)',
    r'\1    return False  # Patched: sm_121 not in prebuilt CUTLASS\n',
    content
)

# Patch module-level constants
content = re.sub(
    r'CUTLASS_FP8_SUPPORTED\s*=\s*cutlass_fp8_supported\(\)',
    'CUTLASS_FP8_SUPPORTED = False  # Patched: sm_121',
    content
)
content = re.sub(
    r'CUTLASS_BLOCK_FP8_SUPPORTED\s*=\s*cutlass_block_fp8_supported\(\)',
    'CUTLASS_BLOCK_FP8_SUPPORTED = False  # Patched: sm_121',
    content
)

with open("$TARGET", 'w') as f:
    f.write(content)

print("  Patched w8a8_utils.py")
PYEOF

# Clear pyc caches
find "$VLLM_PKG" -name "*.pyc" -path "*/quantization/*" -delete 2>/dev/null || true
find "$VLLM_PKG" -name "__pycache__" -path "*/quantization/*" -exec rm -rf {} + 2>/dev/null || true

log "CUTLASS FP8 disabled — Triton fallback active"
