import os
from pathlib import Path

from utils.helpers import _env_bool, _env_int, _env_float, _load_env_file

def _env_symbols(defaults: list[str]) -> list[str]:
    raw = str(os.environ.get("SYMBOLS_CSV", "")).strip()
    if not raw:
        return list(defaults)
    parts = [p.strip() for p in raw.split(",")]
    syms = [p for p in parts if p]
    return syms or list(defaults)

# -------------------------------------------------------------------
# Constants & Configuration
# -------------------------------------------------------------------
PORT = 9999
DASHBOARD_HISTORY_MAX = 5000
DASHBOARD_TRADE_TAPE_MAX = 800
DASHBOARD_INCLUDE_DETAILS = _env_bool("DASHBOARD_INCLUDE_DETAILS", False)

DECISION_REFRESH_SEC = _env_float("DECISION_REFRESH_SEC", 2.0)
DECISION_MAX_INFLIGHT = _env_int("DECISION_MAX_INFLIGHT", 10)
LOG_STDOUT = _env_bool("LOG_STDOUT", False)
DEBUG_MU_SIGMA = _env_bool("DEBUG_MU_SIGMA", False)
DEBUG_TPSL_META = _env_bool("DEBUG_TPSL_META", False)
DEBUG_ROW = _env_bool("DEBUG_ROW", False)
DECISION_LOG_EVERY = _env_int("DECISION_LOG_EVERY", 10)
MC_N_PATHS_LIVE = _env_int("MC_N_PATHS_LIVE", 10000)

SYMBOLS = _env_symbols([
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT",
    "AVAX/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"
])

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
DASHBOARD_FILE = BASE_DIR / "dashboard_v2.html"


# OHLCV settings
TIMEFRAME = "1m"
OHLCV_PRELOAD_LIMIT = 240
OHLCV_REFRESH_LIMIT = 2
OHLCV_SLEEP_SEC = _env_float("OHLCV_SLEEP_SEC", 30.0)
PRELOAD_ON_START = _env_bool("PRELOAD_ON_START", True)

# Orderbook settings
ORDERBOOK_DEPTH = 5
ORDERBOOK_SLEEP_SEC = _env_float("ORDERBOOK_SLEEP_SEC", 2.0)
ORDERBOOK_SYMBOL_INTERVAL_SEC = _env_float("ORDERBOOK_SYMBOL_INTERVAL_SEC", 0.35)

# Networking / Retry
CCXT_TIMEOUT_MS = 20000
MAX_RETRY = 4
RETRY_BASE_SEC = 0.5
MAX_INFLIGHT_REQ = 1

# Risk / Execution settings
ENABLE_LIVE_ORDERS = _env_bool("ENABLE_LIVE_ORDERS", False)
DEFAULT_LEVERAGE = 5.0
MAX_LEVERAGE = 50.0
DEFAULT_SIZE_FRAC = 0.10
MAX_POSITION_HOLD_SEC = _env_int("MAX_POSITION_HOLD_SEC", 600)
POSITION_HOLD_MIN_SEC = _env_int("POSITION_HOLD_MIN_SEC", 120)
POSITION_HOLD_HARD_CAP_SEC = _env_int("POSITION_HOLD_HARD_CAP_SEC", max(1800, MAX_POSITION_HOLD_SEC))
POSITION_CAP_ENABLED = False
EXPOSURE_CAP_ENABLED = True
MAX_CONCURRENT_POSITIONS = 99999
MAX_NOTIONAL_EXPOSURE = 5.0
REBALANCE_THRESHOLD_FRAC = 0.02
EV_DROP_THRESHOLD = 0.0003
K_LEV = 4.0
EV_EXIT_FLOOR = {"bull": -0.0003, "bear": -0.0003, "chop": -0.0002, "volatile": -0.0002}
EV_DROP = {"bull": 0.0010, "bear": 0.0010, "chop": 0.0008, "volatile": 0.0008}
PSL_RISE = {"bull": 0.05, "bear": 0.05, "chop": 0.03, "volatile": 0.03}

EXEC_MODE = str(os.environ.get("EXEC_MODE", "market")).strip().lower()
MAKER_TIMEOUT_MS = _env_int("MAKER_TIMEOUT_MS", 1500)
MAKER_RETRIES = _env_int("MAKER_RETRIES", 2)
MAKER_POLL_MS = _env_int("MAKER_POLL_MS", 200)

CONSENSUS_THRESHOLD = 1.0
RSI_PERIOD = 14
RSI_LONG = 60.0
RSI_SHORT = 40.0
SPREAD_LOOKBACK = 60
SPREAD_Z_ENTRY = 2.0
SPREAD_Z_EXIT = 0.5
SPREAD_SIZE_FRAC = 0.02
SPREAD_HOLD_SEC = 600
SPREAD_ENABLED = False
SPREAD_PAIRS = [
    ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ("SOL/USDT:USDT", "BNB/USDT:USDT"),
]
SPREAD_PCT_MAX = 0.0005
SPREAD_ENTRY_MAX = _env_float("SPREAD_ENTRY_MAX", 0.0)

BYBIT_TAKER_FEE = 0.0006
BYBIT_MAKER_FEE = 0.0001

EV_TUNE_WINDOW_SEC = 30 * 60
EV_TUNE_PCTL = 95
EV_TUNE_MIN_SAMPLES = 40
EV_ENTER_FLOOR_MIN = 0.0008
EV_ENTER_FLOOR_MAX = 0.0025
EV_ENTRY_THRESHOLD = _env_float("EV_ENTRY_THRESHOLD", 0.002)

COOLDOWN_SEC = _env_int("COOLDOWN_SEC", 60)
ENTRY_STREAK_MIN = 1
COOLDOWN_TP_SEC = 30
COOLDOWN_RISK_SEC = 120
