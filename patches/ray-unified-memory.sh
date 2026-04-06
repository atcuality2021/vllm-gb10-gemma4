#!/bin/bash
# =============================================================================
# Ray Unified Memory Patch for GB10
# =============================================================================
# Creates a wrapper script that sets RAY_memory_usage_threshold=1.0
# to prevent Ray from killing workers on unified memory systems.
#
# On GB10, CPU and GPU share 128GB. Model weights loaded to GPU count
# as system memory, so Ray's default 0.95 threshold always triggers.
#
# Usage: ./ray-unified-memory.sh [/path/to/vllm-env]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

VENV="${1:-$HOME/vllm-env}"

# Create activation hook
ACTIVATE_HOOK="$VENV/bin/gb10-env.sh"

cat > "$ACTIVATE_HOOK" << 'ENVEOF'
#!/bin/bash
# GB10 Unified Memory Environment Variables
# Source this before launching vLLM or Ray on DGX Spark GB10

# Disable Ray OOM killer (unified memory makes it trigger falsely)
export RAY_memory_usage_threshold=1.0

# NCCL settings for GB10
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Custom NCCL library (if built from source)
if [ -d "$HOME/nccl/build/lib" ]; then
    export LD_LIBRARY_PATH="$HOME/nccl/build/lib:$LD_LIBRARY_PATH"
fi

# HuggingFace offline mode (use local models)
export HF_HUB_OFFLINE=1
ENVEOF

chmod +x "$ACTIVATE_HOOK"

log "Ray unified memory patch applied"
log "Environment script created at $ACTIVATE_HOOK"
echo "  Source it before launching: source $ACTIVATE_HOOK"
