#!/bin/bash
# =============================================================================
# ManthanQuant — 3-bit Lloyd-Max KV cache compression for GB10
# =============================================================================
# Installs the vendored ManthanQuant package (third_party/manthanquant) into a
# target venv and patches the vLLM attention backends so KV is compressed on
# the ARM CPU cores (no CUDA kernels — avoids the Triton/_C load conflict on
# sm_121). On gemma-4-* the active backend is triton_attn (vLLM hard-forces it
# for Gemma 4's heterogeneous head dims); flash_attn is also patched for
# ASR/Whisper paths. The experimental flashinfer backend is left alone.
#
# Compression: per head-dim-256 vector, bf16 512B -> fp16 radius + 3-bit packed
# = ~98-100B, i.e. ~5.1x, at ~0.978 cosine similarity. On GB10 the compressed
# shadow cache is built CPU-side; activation is gated at runtime by
# MANTHANQUANT_ENABLED=1 (see scripts/launch-gemma4-mtp.sh --manthanquant).
#
# Usage:
#   ./patches/manthanquant-kv-compression.sh [VENV_DIR]
#     VENV_DIR  default: ~/vllm-mtp-env  (the from-source MTP build)
#
#   ./patches/manthanquant-kv-compression.sh ~/vllm-mtp-env --revert
#
# Idempotent: re-running re-patches from the .manthanquant_orig backups.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MQ_DIR="$REPO_ROOT/third_party/manthanquant"
VENV="${1:-$HOME/vllm-mtp-env}"
REVERT=0
[ "${2:-}" = "--revert" ] && REVERT=1

[ -d "$MQ_DIR" ] || err "vendored manthanquant not found at $MQ_DIR"
[ -f "$VENV/bin/python" ] || err "venv not found at $VENV (build it first: ./scripts/build-from-source.sh)"

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo ""
echo "============================================================"
echo "  ManthanQuant KV Compression"
echo "============================================================"
echo "  source:  $MQ_DIR"
echo "  venv:    $VENV"
echo "  action:  $([ "$REVERT" = 1 ] && echo revert || echo install)"
echo "============================================================"
echo ""

if [ "$REVERT" = 1 ]; then
    VLLM_ENV="$VENV" "$PY" "$MQ_DIR/install_vllm_patch.py" --revert all
    log "Reverted vLLM attention backends. (manthanquant package left installed; pip uninstall manthanquant to remove.)"
    exit 0
fi

# --- 1. Install the pure-Python package into the venv ------------------------
# Editable install of the vendored source so `import manthanquant.vllm_patch`
# resolves. The CUDA _C extension is NOT built (and NOT needed) on GB10 — the
# active path is pure numpy (manthanquant/cpu_quantize.py).
info "Installing vendored manthanquant (CPU path — no nvcc build)..."
"$PIP" install -e "$MQ_DIR" 2>&1 | tail -3

"$PY" -c "import manthanquant.vllm_patch; print('manthanquant import OK')" \
    || err "manthanquant package did not import after install"

# --- 2. Patch the vLLM attention backends (flash_attn + triton_attn) ---------
info "Patching vLLM attention backends in $VENV ..."
VLLM_ENV="$VENV" "$PY" "$MQ_DIR/install_vllm_patch.py"

echo ""
log "ManthanQuant installed."
echo "  Activate at serve time with MANTHANQUANT_ENABLED=1, e.g.:"
echo "    MANTHANQUANT=1 VLLM_VENV=\"$VENV\" ./scripts/launch-gemma4-mtp.sh \\"
echo "      ~/hf_models/gemma-4-26B-A4B-it ~/hf_models/gemma-4-26B-A4B-it-assistant"
echo ""
echo "  Verify it is actually running (honest signal):"
echo "    cat ~/logs/manthanquant_active.flag    # one 'kv_hook_first' line per worker pid"
echo ""
