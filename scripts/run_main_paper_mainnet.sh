#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Recommended mode:
# - mainnet market data
# - no real orders (paper)
# - use engine optimal sizing/leverage for paper positions
export BYBIT_TESTNET=0
export ENABLE_LIVE_ORDERS=0
export PAPER_TRADING=1
export PAPER_USE_ENGINE_SIZING=1
export PAPER_EXIT_POLICY_ONLY="${PAPER_EXIT_POLICY_ONLY:-1}"
export LOG_STDOUT=1
# export MC_VERBOSE_PRINT=1
# ✅ Score-based entry (bypass EV/Win/CVaR gates)
export SCORE_ONLY_MODE=0
# ✅ JAX Memory Optimization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export JAX_PLATFORMS=cpu
# export JAX_MC_DEVICE=cpu

PY="${PYTHON:-./.venv311/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

exec "$PY" -u main.py "$@"

