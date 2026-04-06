#!/bin/bash
# =============================================================================
# Launch Gemma 4 31B on DGX Spark GB10
# =============================================================================
# Usage: ./launch-gemma4.sh [/path/to/gemma-4-31B-it] [port]
# =============================================================================

set -euo pipefail

MODEL_PATH="${1:-$HOME/hf_models/gemma-4-31B-it}"
PORT="${2:-8000}"
VENV="${VLLM_VENV:-$HOME/vllm-env}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Usage: $0 /path/to/gemma-4-31B-it [port]"
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
echo "  Launching Gemma 4 31B on GB10"
echo "============================================="
echo "  Model:   $MODEL_PATH"
echo "  Port:    $PORT"
echo "  GPU Mem: 0.55"
echo "  Context: 8192"
echo "  Seqs:    4"
echo "============================================="
echo ""
echo "  NOTE: Gemma 4 is a dense 31B model."
echo "  Expect ~3.8 tok/s on GB10."
echo "  For faster inference, use Qwen3-Omni-30B (~28 tok/s)."
echo ""

exec vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --gpu-memory-utilization 0.55 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --trust-remote-code \
    --enforce-eager \
    --enable-prefix-caching
