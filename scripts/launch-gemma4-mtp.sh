#!/bin/bash
# =============================================================================
# Launch Gemma 4 26B-A4B (MoE) with MTP speculative decoding on DGX Spark GB10
# =============================================================================
# Requires a from-source fork build (see scripts/build-from-source.sh) — MTP /
# gemma4_mtp is fork-only and is NOT present in stock or upstream-main vLLM.
#
# Usage:
#   ./launch-gemma4-mtp.sh [/path/to/gemma-4-26B-A4B-it] [/path/to/draft] [port]
#
# The draft is the matching MTP assistant checkpoint
# (gemma-4-26B-A4B-it-assistant, model_type gemma4_assistant). Pass "none" to
# launch the target model without speculative decoding.
#
# Env:
#   VLLM_VENV   venv with the from-source build   (default ~/vllm-mtp-env)
#   API_KEY     bearer key for the OpenAI server  (optional)
# =============================================================================

set -euo pipefail

MODEL_PATH="${1:-$HOME/hf_models/gemma-4-26B-A4B-it}"
DRAFT_PATH="${2:-$HOME/hf_models/gemma-4-26B-A4B-it-assistant}"
PORT="${3:-8000}"
VENV="${VLLM_VENV:-$HOME/vllm-mtp-env}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Usage: $0 /path/to/gemma-4-26B-A4B-it [/path/to/draft|none] [port]"
    exit 1
fi

# Source GB10 environment if available
[ -f "$VENV/bin/gb10-env.sh" ] && source "$VENV/bin/gb10-env.sh"

# GB10 required environment
export RAY_memory_usage_threshold=1.0
export HF_HUB_OFFLINE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

source "$VENV/bin/activate"

# Speculative decoding config (omit if draft is "none" or missing)
SPEC_ARGS=()
SPEC_NOTE="disabled"
if [ "$DRAFT_PATH" != "none" ] && [ -d "$DRAFT_PATH" ]; then
    SPEC_ARGS=(--speculative-config "{\"model\": \"$DRAFT_PATH\", \"method\": \"mtp\", \"num_speculative_tokens\": 1}")
    SPEC_NOTE="MTP draft: $DRAFT_PATH"
elif [ "$DRAFT_PATH" != "none" ]; then
    echo "WARN: draft not found at $DRAFT_PATH — launching without MTP."
fi

# Optional API key
KEY_ARGS=()
[ -n "${API_KEY:-}" ] && KEY_ARGS=(--api-key "$API_KEY")

echo ""
echo "============================================="
echo "  Launching Gemma 4 26B-A4B (MoE) on GB10"
echo "============================================="
echo "  Model:    $MODEL_PATH"
echo "  Port:     $PORT"
echo "  Spec:     $SPEC_NOTE"
echo "  GPU Mem:  0.55"
echo "  Context:  32768"
echo "  Seqs:     6"
echo "  Parser:   gemma4 (tool calling + JSON)"
echo "============================================="
echo ""
echo "  MoE (4B active) — far faster than the dense 31B."
echo "  With MTP, expect ~1.6x decode speedup at ~80% draft acceptance."
echo ""

exec vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --served-model-name gemma-4-26B \
    --gpu-memory-utilization 0.55 \
    --max-model-len 32768 \
    --max-num-seqs 6 \
    --trust-remote-code \
    --enforce-eager \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    "${SPEC_ARGS[@]}" \
    "${KEY_ARGS[@]}"
