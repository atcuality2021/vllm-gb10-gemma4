#!/bin/bash
# =============================================================================
# Launch Qwen3-Omni-30B on DGX Spark GB10
# =============================================================================
# Usage: ./launch-qwen-omni.sh [/path/to/Qwen3-Omni-30B-A3B-Instruct] [port]
# =============================================================================

set -euo pipefail

MODEL_PATH="${1:-$HOME/hf_models/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${2:-8000}"
VENV="${VLLM_VENV:-$HOME/vllm-env}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Usage: $0 /path/to/Qwen3-Omni-30B-A3B-Instruct [port]"
    exit 1
fi

# Source GB10 environment if available
[ -f "$VENV/bin/gb10-env.sh" ] && source "$VENV/bin/gb10-env.sh"

# GB10 required environment
export RAY_memory_usage_threshold=1.0
export HF_HUB_OFFLINE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Activate venv
source "$VENV/bin/activate"

echo ""
echo "============================================="
echo "  Launching Qwen3-Omni-30B on GB10"
echo "============================================="
echo "  Model:   $MODEL_PATH"
echo "  Port:    $PORT"
echo "  GPU Mem: 0.85"
echo "  Context: 16384"
echo "  Seqs:    16"
echo "============================================="
echo ""

exec vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --trust-remote-code \
    --enforce-eager \
    --enable-prefix-caching \
    --max-num-batched-tokens 4096
