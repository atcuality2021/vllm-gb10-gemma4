#!/bin/bash
# =============================================================================
# CUTLASS FP8 Patch for GB10 (sm_121)
# =============================================================================
# Disables CUTLASS FP8 kernels which are not prebuilt for sm_121.
# vLLM falls back to Triton-based FP8 kernels which support sm_121.
#
# Usage: ./cutlass-fp8-sm121.sh [/path/to/vllm-env]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

VENV="${1:-$HOME/vllm-env}"
PYTHON="$VENV/bin/python"

# Find vLLM package path
PYVER=$(detect_pyver "$VENV")
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
CUTLASS_TARGET="$TARGET" python3 << 'PYEOF'
import os, re, sys

target = os.environ["CUTLASS_TARGET"]
with open(target) as f:
    content = f.read()

changed = False

# Patch cutlass_fp8_supported()
new_content = re.sub(
    r'(def cutlass_fp8_supported\(\)[^:]*:\n)',
    r'\1    return False  # Patched: sm_121 not in prebuilt CUTLASS\n',
    content
)
if new_content != content:
    changed = True
    content = new_content

# Patch cutlass_block_fp8_supported()
new_content = re.sub(
    r'(def cutlass_block_fp8_supported\(\)[^:]*:\n)',
    r'\1    return False  # Patched: sm_121 not in prebuilt CUTLASS\n',
    content
)
if new_content != content:
    changed = True
    content = new_content

# Patch module-level constants
for old, new in [
    (r'CUTLASS_FP8_SUPPORTED\s*=\s*cutlass_fp8_supported\(\)',
     'CUTLASS_FP8_SUPPORTED = False  # Patched: sm_121'),
    (r'CUTLASS_BLOCK_FP8_SUPPORTED\s*=\s*cutlass_block_fp8_supported\(\)',
     'CUTLASS_BLOCK_FP8_SUPPORTED = False  # Patched: sm_121'),
]:
    new_content = re.sub(old, new, content)
    if new_content != content:
        changed = True
        content = new_content

if changed:
    with open(target, 'w') as f:
        f.write(content)
    print("  Patched w8a8_utils.py")
elif 'return False  # Patched: sm_121' in content:
    print("  Already patched")
else:
    print("  ERROR: Could not find patch targets — vLLM version may have changed", file=sys.stderr)
    sys.exit(1)
PYEOF

# Clear pyc caches
find "$VLLM_PKG" -name "*.pyc" -path "*/quantization/*" -delete 2>/dev/null || true
find "$VLLM_PKG" -name "__pycache__" -path "*/quantization/*" -exec rm -rf {} + 2>/dev/null || true

log "CUTLASS FP8 disabled — Triton fallback active"
