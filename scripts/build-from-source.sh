#!/bin/bash
# =============================================================================
# Build vLLM from source for DGX Spark GB10 (aarch64) — native Gemma 4 + MTP
# =============================================================================
# Builds the atcuality2021/vllm fork natively for the NVIDIA DGX Spark
# (GB10, ARM aarch64, sm_121 / CUDA cap 12.1) so that Gemma 4 support —
# including gemma4_mtp (MTP speculative decoding) and gemma4_unified
# (audio/video) — is baked into the wheel, instead of the file-copy backport
# in patches/gemma4-backport.sh.
#
# Why build instead of backport?
#   gemma4_mtp / gemma4_assistant is FORK-ONLY. It is NOT in upstream vLLM
#   main, so neither a stock `pip install vllm` nor an upstream-main build can
#   serve a Gemma 4 MTP draft (e.g. gemma-4-26B-A4B-it-assistant). The fork at
#   FORK_COMMIT carries gemma4 + gemma4_mtp + gemma4_unified directly.
#   Bonus: on vLLM 0.22.x the Blackwell/unified-memory assert that the older
#   0.18.x line tripped over is fixed upstream — no separate assert patch.
#
# Why torch 2.11.0+cu130?
#   The fork @ FORK_COMMIT pins torch==2.11.0 (requirements/build/cuda.txt).
#   The aarch64 SBSA wheel of that exact version is published on
#   download.pytorch.org/whl/cu130, so the build and runtime ABIs match.
#
# Why TORCH_CUDA_ARCH_LIST="12.0+PTX"?
#   torch's bundled arch list stops at sm_120; GB10 is sm_121. Building with
#   sm_120 + embedded PTX lets the driver JIT-compile to sm_121 at load time.
#   (Verified: CUDA ops run on GB10 capability (12,1) with this setting.)
#
# RAM SAFETY (IMPORTANT)
#   A full-parallel CUDA compile can use many GiB per translation unit and
#   WILL OOM a box that is also serving a model. Keep MAX_JOBS low (2-3) and
#   watch `free -g` during the build. Everything happens in an ISOLATED venv
#   and produces a wheel — nothing touches an existing runtime env.
#
# Usage:
#   ./scripts/build-from-source.sh [VENV_DIR] [MAX_JOBS]
#     VENV_DIR  default: ~/vllm-mtp-env
#     MAX_JOBS  default: 2   (raise to 3 only with comfortable `free -g` headroom)
#
# Overridable via env: SRC_DIR, WHEEL_DIR, TORCH_CUDA_ARCH_LIST, CUDA_HOME.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

# --- Parameters --------------------------------------------------------------
FORK_REPO="${FORK_REPO:-https://github.com/atcuality2021/vllm.git}"
FORK_COMMIT="${FORK_COMMIT:-2a983c79a}"          # pinned, reproducible
SRC_DIR="${SRC_DIR:-$HOME/vllm-fork-build}"
VENV_DIR="${1:-$HOME/vllm-mtp-env}"
MAX_JOBS="${2:-2}"
WHEEL_DIR="${WHEEL_DIR:-$HOME/vllm-mtp-wheels}"
TORCH_VER="${TORCH_VER:-2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

# CUDA toolkit (13.0 ships at /usr/local/cuda-13.0 on the DGX Spark image)
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
[ -d "$CUDA_HOME" ] || CUDA_HOME=/usr/local/cuda
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

echo ""
echo "============================================================"
echo "  vLLM From-Source Build for GB10 (native Gemma 4 + MTP)"
echo "============================================================"
echo "  fork:     $FORK_REPO"
echo "  commit:   $FORK_COMMIT"
echo "  venv:     $VENV_DIR"
echo "  torch:    $TORCH_VER (cu130, aarch64 SBSA)"
echo "  arch:     $ARCH_LIST"
echo "  MAX_JOBS: $MAX_JOBS"
echo "  platform: $(uname -m)"
echo "============================================================"
echo ""

# --- Prereq checks -----------------------------------------------------------
[ "$(uname -m)" = "aarch64" ] || warn "Not aarch64 — this build targets GB10/ARM."
command -v nvcc >/dev/null 2>&1 || err "nvcc not found (need CUDA toolkit at $CUDA_HOME/bin)"
nvcc --version | tail -2
gcc --version | head -1
PYBIN="$(command -v python3.12 || command -v python3)"
"$PYBIN" -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version' \
  || warn "python is not 3.12 — wheel tag may differ"

# --- 1. Source: clone fork @ pinned commit -----------------------------------
if [ -d "$SRC_DIR/.git" ]; then
    log "Source exists: $SRC_DIR (fetching)"
    git -C "$SRC_DIR" fetch --all -q
else
    git clone -q "$FORK_REPO" "$SRC_DIR"
    log "Cloned fork -> $SRC_DIR"
fi
git -C "$SRC_DIR" checkout -q "$FORK_COMMIT"
log "Checked out $(git -C "$SRC_DIR" log --oneline -1)"
ls "$SRC_DIR/vllm/model_executor/models/" | grep -q gemma4_mtp \
    || err "gemma4_mtp.py absent — wrong commit (MTP support missing)"
log "gemma4_mtp.py present in source"

# --- 2. Isolated venv + torch + build deps -----------------------------------
[ -d "$VENV_DIR" ] || "$PYBIN" -m venv "$VENV_DIR"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"
"$PIP" install -q -U pip wheel
info "Installing torch ${TORCH_VER} (cu130 aarch64 SBSA wheel)"
"$PIP" install "torch==${TORCH_VER}" --index-url "$TORCH_INDEX"
"$PIP" install -q numpy
"$PIP" install -q -r "$SRC_DIR/requirements/build/cuda.txt"
"$PY" -c "import torch; print('torch', torch.__version__, torch.version.cuda)"

# --- 3. Build the wheel (RAM-capped) -----------------------------------------
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export MAX_JOBS NVCC_THREADS=1 VLLM_TARGET_DEVICE=cuda
mkdir -p "$WHEEL_DIR"
warn "Compiling vLLM CUDA extensions — LONG at MAX_JOBS=$MAX_JOBS (~2h on GB10)."
warn "Monitor with: watch -n30 'free -g; ls -lh $WHEEL_DIR'"
( cd "$SRC_DIR" && "$PIP" wheel . --no-build-isolation --no-deps -w "$WHEEL_DIR" )
WHEEL="$(ls -t "$WHEEL_DIR"/vllm-*.whl | head -1)"
[ -n "$WHEEL" ] || err "wheel not produced"
log "Built wheel: $WHEEL"

# --- 4. Install wheel + validate ---------------------------------------------
"$PIP" install -q "$WHEEL"
"$PY" -c "import vllm; print('vllm', getattr(vllm, '__version__', '(editable)'))"
"$PY" - <<'PYEOF'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert "Gemma4MTPModel" in archs, "Gemma4MTPModel not registered"
assert "Gemma4ForCausalLM" in archs, "Gemma4ForCausalLM not registered"
print("OK: Gemma4MTPModel + Gemma4ForCausalLM registered (MTP draft available)")
PYEOF
log "vLLM installed into $VENV_DIR"

echo ""
log "BUILD COMPLETE."
echo "  Single-node Gemma 4 still needs the GB10 runtime fixes (CUTLASS FP8,"
echo "  Ray OOM threshold); multi-node also needs the NCCL sm_121 build:"
echo "    ./patches/cutlass-fp8-sm121.sh \"$VENV_DIR\""
echo "    ./patches/ray-unified-memory.sh \"$VENV_DIR\""
echo "  The gemma4-backport patch is NOT needed — Gemma 4 is native in this build."
echo ""
echo "  Launch the MoE Gemma 4 with MTP speculative decoding:"
echo "    VLLM_VENV=\"$VENV_DIR\" ./scripts/launch-gemma4-mtp.sh \\"
echo "      ~/hf_models/gemma-4-26B-A4B-it ~/hf_models/gemma-4-26B-A4B-it-assistant"
echo ""
