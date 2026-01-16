#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "probe" && "$MODE" != "live" ]]; then
  echo "Usage: $0 {probe|live}"
  echo
  echo "Environment:"
  echo "  - Credentials: set BYBIT_API_KEY/BYBIT_API_SECRET or put them in state/bybit.env (or state/bybit.env.example)."
  echo "  - Testnet: set BYBIT_TESTNET=1 (default in this script)."
  echo "  - Override any PMAKER_* vars by exporting them before running."
  exit 2
fi
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export LOG_STDOUT="${LOG_STDOUT:-1}"
export BYBIT_TESTNET="${BYBIT_TESTNET:-1}"
# Market-data feed: default to mainnet so LIVE matches paper's data distribution.
# Orders/positions still use BYBIT_TESTNET.
export DATA_BYBIT_TESTNET="${DATA_BYBIT_TESTNET:-0}"
export EXEC_MODE="${EXEC_MODE:-maker_then_market}"
export PMAKER_ENABLE="${PMAKER_ENABLE:-1}"
export MAX_TOTAL_LEVERAGE="${MAX_TOTAL_LEVERAGE:-10}"

if [[ "$MODE" == "probe" ]]; then
  export TRAIN_PMAKER_ONLY="${TRAIN_PMAKER_ONLY:-1}"
  export TRAIN_PMAKER_ONLY_STRICT="${TRAIN_PMAKER_ONLY_STRICT:-1}"
  export PMAKER_DEBUG="${PMAKER_DEBUG:-1}"
  export PMAKER_PROBE_MAX_RUN_SEC="${PMAKER_PROBE_MAX_RUN_SEC:-90}"
  export PMAKER_PROBE_INTERVAL_SEC="${PMAKER_PROBE_INTERVAL_SEC:-0.6}"
  export PMAKER_PROBE_TIMEOUT_MS="${PMAKER_PROBE_TIMEOUT_MS:-8000}"
  export PMAKER_PROBE_POLL_MS="${PMAKER_PROBE_POLL_MS:-120}"
  export PMAKER_PROBE_IMPROVE_TICKS="${PMAKER_PROBE_IMPROVE_TICKS:-10}"
  export PMAKER_PROBE_QTY_USD="${PMAKER_PROBE_QTY_USD:-40}"
  export PMAKER_PROBE_SYMBOLS_CSV="${PMAKER_PROBE_SYMBOLS_CSV:-ETH/USDT:USDT,SOL/USDT:USDT}"
  # Optional auto transition to live mode:
  # export PMAKER_AUTO_START_TRADING=1
  # export PMAKER_AUTO_ENABLE_ORDERS=0
  # export PMAKER_AUTO_MIN_ATTEMPTS=200
  # export PMAKER_AUTO_MIN_FILLS=10
  # export PMAKER_AUTO_MIN_FILL_RATE=0.05
else
  export TRAIN_PMAKER_ONLY="${TRAIN_PMAKER_ONLY:-0}"
  export TRAIN_PMAKER_ONLY_STRICT="${TRAIN_PMAKER_ONLY_STRICT:-0}"
  export PMAKER_DEBUG="${PMAKER_DEBUG:-0}"
  # ✅ 권장 설정: engine 모드에서도 pmaker 학습 활성화
  export PMAKER_TRAIN_STEPS="${PMAKER_TRAIN_STEPS:-2}"
  export PMAKER_PROBE_DURING_ENGINE="${PMAKER_PROBE_DURING_ENGINE:-1}"
  export PMAKER_BATCH="${PMAKER_BATCH:-64}"
fi

PY_BIN="python3"
if [[ -x "$ROOT/.venv311/bin/python" ]]; then
  PY_BIN="$ROOT/.venv311/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY_BIN="$ROOT/.venv/bin/python"
fi

exec "$PY_BIN" -u main.py "$@"
