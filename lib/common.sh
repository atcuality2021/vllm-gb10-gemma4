#!/usr/bin/env bash
# =============================================================================
# Shared logging helpers for vllm-gb10-gemma4 scripts
# Source this at the top of any script that needs colored output.
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Detect Python version in a venv
detect_pyver() {
    local venv="$1"
    "$venv/bin/python" --version 2>&1 | sed -n 's/.*Python 3\.\([0-9]*\).*/3.\1/p'
}
