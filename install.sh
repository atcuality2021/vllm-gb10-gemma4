#!/bin/bash
# =============================================================================
# vLLM GB10 + Gemma 4 — Complete Installer
# =============================================================================
# One-command setup: applies all GB10 fixes + Gemma 4 backport + benchmark tool
#
# Usage:
#   ./install.sh [/path/to/vllm-env]
#
# Default venv: ~/vllm-env
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
VENV="${1:-$HOME/vllm-env}"

echo ""
echo "============================================================"
echo "  vLLM GB10 + Gemma 4 — Complete Installer"
echo "============================================================"
echo "  venv:       $VENV"
echo "  patches:    $SCRIPT_DIR/patches/"
echo "  platform:   $(uname -m)"
echo "============================================================"
echo ""

# --- Validate environment ---
[ -f "$VENV/bin/pip" ] || err "vLLM virtualenv not found at $VENV. Create one first:
  python3.12 -m venv $VENV
  $VENV/bin/pip install vllm"

PYVER=$(detect_pyver "$VENV")
[ -d "$VENV/lib/python${PYVER}/site-packages/vllm" ] || \
    err "vLLM not installed in $VENV. Install with: $VENV/bin/pip install vllm"

VLLM_VER=$($VENV/bin/pip show vllm 2>/dev/null | grep Version | awk '{print $2}')
info "Detected vLLM version: $VLLM_VER"

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    warn "This machine is $ARCH, not aarch64. GB10 patches are designed for ARM."
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# --- Step 1: CUTLASS FP8 patch ---
echo ""
info "Step 1/4: Applying CUTLASS FP8 sm_121 patch..."
bash "$SCRIPT_DIR/patches/cutlass-fp8-sm121.sh" "$VENV"

# --- Step 2: Ray unified memory patch ---
echo ""
info "Step 2/4: Applying Ray unified memory patch..."
bash "$SCRIPT_DIR/patches/ray-unified-memory.sh" "$VENV"

# --- Step 3: NCCL sm_121 build (optional, multi-node only) ---
echo ""
if [ -f "$HOME/nccl/build/lib/libnccl.so" ]; then
    log "NCCL custom build already exists at $HOME/nccl/build/lib/"
else
    warn "NCCL custom build not found. This is only needed for multi-node setups."
    read -p "Build NCCL from source for sm_121? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Step 3/4: Building NCCL from source..."
        bash "$SCRIPT_DIR/patches/nccl-sm121-build.sh"
    else
        info "Step 3/4: Skipping NCCL build (single-node only)"
    fi
fi

# --- Step 4: Gemma 4 backport ---
echo ""
info "Step 4/4: Applying Gemma 4 backport patch..."
bash "$SCRIPT_DIR/patches/gemma4-backport.sh" "$VENV"

# --- Install benchmark dependencies ---
echo ""
info "Installing benchmark dependencies..."
$VENV/bin/pip install requests -q 2>&1 | tail -1

# --- Summary ---
echo ""
echo "============================================================"
echo "  Installation Complete"
echo "============================================================"
echo ""
echo "  Patches applied:"
echo "    [x] CUTLASS FP8 disabled (sm_121 fallback to Triton)"
echo "    [x] Ray OOM killer disabled (unified memory)"
if [ -f "$HOME/nccl/build/lib/libnccl.so" ]; then
echo "    [x] NCCL built from source (sm_121)"
else
echo "    [ ] NCCL not built (single-node only)"
fi
echo "    [x] Gemma 4 backported from vLLM main"
echo ""
echo "  Launch scripts:"
echo "    ./scripts/launch-gemma4.sh /path/to/gemma-4-31B-it"
echo "    ./scripts/launch-qwen-omni.sh /path/to/Qwen3-Omni-30B-A3B-Instruct"
echo ""
echo "  Run benchmarks:"
echo "    ./scripts/run-benchmark.sh --model model-name --url http://localhost:8000"
echo ""
