#!/bin/bash
# =============================================================================
# NCCL Build from Source for sm_121 (GB10 Blackwell)
# =============================================================================
# Pre-built NCCL packages lack GPU kernels for sm_121. This builds NCCL
# v2.28.9 from source with explicit sm_121 support.
#
# Required for multi-node tensor parallelism on GB10.
# Single-node setups can skip this.
#
# Usage: ./nccl-sm121-build.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

NCCL_DIR="$HOME/nccl"
NCCL_VERSION="v2.28.9-1"

# Check for CUDA
[ -d "/usr/local/cuda" ] || err "CUDA not found at /usr/local/cuda"

# Check if already built
if [ -f "$NCCL_DIR/build/lib/libnccl.so" ]; then
    EXISTING_VER=$(strings "$NCCL_DIR/build/lib/libnccl.so" | grep "NCCL version" | head -1 || echo "unknown")
    warn "NCCL already built at $NCCL_DIR/build/lib/"
    echo "  $EXISTING_VER"
    read -p "Rebuild? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || { log "Skipping rebuild"; exit 0; }
fi

echo ""
echo "Building NCCL $NCCL_VERSION for sm_121..."
echo ""

# Clone
if [ -d "$NCCL_DIR/.git" ]; then
    cd "$NCCL_DIR"
    git fetch origin
    git checkout "$NCCL_VERSION"
else
    git clone https://github.com/NVIDIA/nccl.git "$NCCL_DIR"
    cd "$NCCL_DIR"
    git checkout "$NCCL_VERSION"
fi

# Build
NPROC=$(nproc)
echo "Building with $NPROC threads..."

make -j"$NPROC" src.build \
    NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" \
    CUDA_HOME=/usr/local/cuda

# Verify
[ -f "$NCCL_DIR/build/lib/libnccl.so" ] || err "Build failed — libnccl.so not found"

log "NCCL built successfully at $NCCL_DIR/build/lib/"
echo ""
echo "  Add to your environment:"
echo "    export LD_LIBRARY_PATH=$NCCL_DIR/build/lib:\$LD_LIBRARY_PATH"
echo ""
echo "  This must be set on ALL nodes in a multi-node cluster."
echo ""
