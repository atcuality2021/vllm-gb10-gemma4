#!/bin/bash
# =============================================================================
# Run Benchmark Suite Against Any Running vLLM Model
# =============================================================================
# Usage:
#   ./run-benchmark.sh --model MODEL_NAME --url http://localhost:8000
#   ./run-benchmark.sh --model gemma-4-31B-it --url http://192.168.29.252:8000
#   ./run-benchmark.sh --model Qwen3-Omni-30B --compare previous_report.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK="$REPO_DIR/benchmarks/model_benchmark.py"
OUTPUT_DIR="$REPO_DIR/benchmarks/reports"
VENV="${VLLM_VENV:-$HOME/vllm-env}"

# Defaults
URL="http://localhost:8000"
API_KEY="not-needed"
MODEL=""
COMPARE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --url)
            URL="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --api-key)
            API_KEY="$2"; shift 2 ;;
        --compare)
            COMPARE="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --model MODEL_NAME [--url URL] [--api-key KEY] [--compare REPORT.json]"
            echo ""
            echo "Options:"
            echo "  --model       Model name (required)"
            echo "  --url         Base URL (default: http://localhost:8000)"
            echo "  --api-key     API key (default: not-needed)"
            echo "  --compare     Path to previous report JSON for comparison"
            echo "  --output-dir  Output directory (default: benchmarks/reports/)"
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required"
    echo "Usage: $0 --model MODEL_NAME [--url URL]"
    exit 1
fi

# Find Python
if [ -f "$VENV/bin/python" ]; then
    PYTHON="$VENV/bin/python"
else
    PYTHON="python3"
fi

# Check benchmark script exists
[ -f "$BENCHMARK" ] || { echo "ERROR: Benchmark script not found at $BENCHMARK"; exit 1; }

echo ""
echo "============================================="
echo "  Running Benchmark Suite"
echo "============================================="
echo "  Model:   $MODEL"
echo "  URL:     $URL"
echo "  Output:  $OUTPUT_DIR"
echo "============================================="
echo ""

ARGS="--url $URL --api-key $API_KEY --model $MODEL --output-dir $OUTPUT_DIR"

if [ -n "$COMPARE" ]; then
    ARGS="$ARGS --compare $COMPARE"
fi

exec $PYTHON "$BENCHMARK" $ARGS
