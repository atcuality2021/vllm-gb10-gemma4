#!/bin/bash
# =============================================================================
# Gemma 4 Backport Patch for vLLM 0.18.x on GB10
# =============================================================================
# Backports Gemma 4 model support (PR #38826) from vLLM main to 0.18.x.
#
# What it does:
#   1. Upgrades huggingface_hub + installs transformers from GitHub main
#      (gemma4 model_type not in any stable transformers release yet)
#   2. Clones vLLM main to get native gemma4 model files
#   3. Copies gemma4 model, RoPE, reasoning, and tool parser files
#   4. Patches vLLM registry to register Gemma4 architectures
#   5. Patches base.py to handle null sub_configs (audio_config=null)
#   6. Patches utils.py to load named buffers (layer_scalar)
#
# Usage: ./gemma4-backport.sh [/path/to/vllm-env]
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

VENV="${1:-$HOME/vllm-env}"
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

# Find vLLM package path
PYVER=$($PYTHON --version 2>&1 | grep -oP '3\.\d+')
VLLM_PKG="$VENV/lib/python${PYVER}/site-packages/vllm"

[ -f "$PIP" ]       || err "pip not found at $PIP"
[ -d "$VLLM_PKG" ]  || err "vLLM not found at $VLLM_PKG"

VLLM_VER=$($PIP show vllm 2>/dev/null | grep Version | awk '{print $2}')
echo ""
echo "============================================="
echo "  Gemma 4 Backport Patch"
echo "  vLLM version: $VLLM_VER"
echo "  venv: $VENV"
echo "============================================="
echo ""

# Check if already patched
if grep -q 'Gemma4ForCausalLM' "$VLLM_PKG/model_executor/models/registry.py" 2>/dev/null; then
    warn "Gemma4 already registered in vLLM registry. Re-patching anyway."
fi

# ---- Step 1: Upgrade transformers ----
echo "[1/6] Installing transformers with gemma4 support..."
$PIP install --upgrade huggingface_hub -q 2>&1 | tail -1
$PIP install git+https://github.com/huggingface/transformers.git --no-deps -q 2>&1 | tail -1

# Verify
$PYTHON -c "from transformers.models.gemma4 import Gemma4Config; print('  Gemma4Config: OK')" || \
    err "transformers still doesn't have gemma4 support"
log "transformers upgraded"

# ---- Step 2: Clone vLLM main ----
echo "[2/6] Cloning vLLM main branch (shallow)..."
TMPDIR=$(mktemp -d)
git clone --depth 1 https://github.com/vllm-project/vllm.git "$TMPDIR/vllm-src" 2>&1 | tail -1

SRC="$TMPDIR/vllm-src/vllm"
[ -f "$SRC/model_executor/models/gemma4.py" ] || err "gemma4.py not found in vLLM main — PR may not be merged yet"
log "vLLM source cloned"

# ---- Step 3: Copy gemma4 files ----
echo "[3/6] Copying gemma4 model files..."
MODELS="$VLLM_PKG/model_executor/models"
ROPE="$VLLM_PKG/model_executor/layers/rotary_embedding"

# Model files
cp "$SRC/model_executor/models/gemma4.py"       "$MODELS/"
cp "$SRC/model_executor/models/gemma4_mm.py"     "$MODELS/"
cp "$SRC/model_executor/models/gemma4_utils.py"  "$MODELS/"

# Fix import path for 0.18.x compatibility
sed -i 's|from vllm.inputs import MultiModalDataDict|from vllm.multimodal.inputs import MultiModalDataDict|' \
    "$MODELS/gemma4_mm.py"

# RoPE
cp "$SRC/model_executor/layers/rotary_embedding/gemma4_rope.py" "$ROPE/"
cp "$SRC/model_executor/layers/rotary_embedding/__init__.py"    "$ROPE/__init__.py"

# Copy telechat3 rope if present (new dependency in main's __init__.py)
if [ -f "$SRC/model_executor/layers/rotary_embedding/telechat3_scaling_rope.py" ]; then
    cp "$SRC/model_executor/layers/rotary_embedding/telechat3_scaling_rope.py" "$ROPE/"
fi

# Reasoning & tool parsers
cp "$SRC/reasoning/gemma4_reasoning_parser.py"  "$VLLM_PKG/reasoning/"
cp "$SRC/reasoning/gemma4_utils.py"             "$VLLM_PKG/reasoning/"
cp "$SRC/reasoning/__init__.py"                 "$VLLM_PKG/reasoning/__init__.py"
cp "$SRC/tool_parsers/gemma4_utils.py"          "$VLLM_PKG/tool_parsers/"
cp "$SRC/tool_parsers/gemma4_tool_parser.py"    "$VLLM_PKG/tool_parsers/"
cp "$SRC/tool_parsers/__init__.py"              "$VLLM_PKG/tool_parsers/__init__.py"

# Config convertor
cp "$SRC/transformers_utils/model_arch_config_convertor.py" \
   "$VLLM_PKG/transformers_utils/model_arch_config_convertor.py"

log "All gemma4 files copied"

# ---- Step 4: Patch registry ----
echo "[4/6] Patching model registry..."
REGISTRY="$VLLM_PKG/model_executor/models/registry.py"
if ! grep -q 'Gemma4ForCausalLM' "$REGISTRY"; then
    sed -i '/"Gemma3ForConditionalGeneration".*gemma3_mm/a\    "Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM"),\n    "Gemma4ForConditionalGeneration": ("gemma4_mm", "Gemma4ForConditionalGeneration"),  # noqa: E501' \
        "$REGISTRY"
    log "Registry patched"
else
    log "Registry already has Gemma4 entries"
fi

# ---- Step 5: Patch base.py (null sub_config) ----
echo "[5/6] Patching base.py for null sub_configs..."
BASEPY="$VLLM_PKG/model_executor/models/transformers/base.py"
if ! grep -q 'sub_config is None' "$BASEPY" 2>/dev/null; then
    sed -i '/if sub_config.dtype != (dtype := self.config.dtype):/i\            if sub_config is None:\n                continue' "$BASEPY"
    log "base.py patched"
else
    log "base.py already patched"
fi

# ---- Step 6: Patch utils.py (named buffers) ----
echo "[6/6] Patching utils.py for named buffer loading..."
UTILSPY="$VLLM_PKG/model_executor/models/utils.py"
if ! grep -q 'named_buffers' "$UTILSPY" 2>/dev/null; then
    python3 << 'PYEOF'
import sys, os
path = os.environ.get("UTILSPY_PATH", sys.argv[1] if len(sys.argv) > 1 else "")
with open(path) as f:
    content = f.read()

old = '''                child_params[stat_name] = module_state_dict[stat_name]'''
new = '''                child_params[stat_name] = module_state_dict[stat_name]

        # Also include named buffers (e.g. layer_scalar in Gemma4)
        for buf_name, buf_tensor in module.named_buffers(recurse=False):
            if buf_name not in child_params:
                child_params[buf_name] = buf_tensor'''

idx = content.rfind(old)
if idx != -1:
    content = content[:idx] + new + content[idx+len(old):]
    with open(path, 'w') as f:
        f.write(content)
    print("  Patched utils.py")
else:
    print("  Could not find patch target in utils.py — may already be patched")
PYEOF
    log "utils.py patched"
else
    log "utils.py already patched"
fi

# ---- Cleanup ----
rm -rf "$TMPDIR"

echo ""
echo "============================================="
echo "  Gemma 4 backport complete!"
echo "============================================="
echo ""
echo "  Verify with:"
echo "    $PYTHON -c 'from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM; print(\"OK\")'"
echo ""
echo "  Launch with:"
echo "    vllm serve /path/to/gemma-4-31B-it \\"
echo "      --trust-remote-code --enforce-eager \\"
echo "      --gpu-memory-utilization 0.55 --max-model-len 8192 \\"
echo "      --max-num-seqs 4 --enable-prefix-caching"
echo ""
