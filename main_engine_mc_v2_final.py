from __future__ import annotations
import asyncio
import json
import time
import math
import random
import os
import numpy as np
from pathlib import Path
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt

from engines.engine_hub import EngineHub
from trainers.online_alpha_trainer import OnlineAlphaTrainer, AlphaTrainerConfig
from utils.alpha_features import build_alpha_features

# -------------------------------------------------------------------
# aiohttp 일부 macOS 환경에서 TCP keepalive 설정 시 OSError(22)가 날 수 있다.
# (dashboard 접속 불가 증상). best-effort로 무시 처리.
# -------------------------------------------------------------------
try:
    import aiohttp.tcp_helpers as _aiohttp_tcp_helpers

    _orig_tcp_keepalive = _aiohttp_tcp_helpers.tcp_keepalive

    def _tcp_keepalive_safe(transport):
        try:
            return _orig_tcp_keepalive(transport)
        except OSError:
            return None

    _aiohttp_tcp_helpers.tcp_keepalive = _tcp_keepalive_safe
except Exception:
    pass
from engines.mc_engine import mc_first_passage_tp_sl_jax
from engines.mc_risk import compute_cvar, kelly_with_cvar, PyramidTracker, ExitPolicy, should_exit_position
from regime import adjust_mu_sigma, time_regime, get_regime_mu_sigma
from engines.running_stats import RunningStats
from engines.mc_engine import MonteCarloEngine
try:
    from trainers.online_alpha_trainer import OnlineAlphaTrainer, AlphaTrainerConfig
except ImportError:
    OnlineAlphaTrainer = None
    AlphaTrainerConfig = None

# Import from our new modules
from config import *
from core.dashboard_server import DashboardServer
from core.data_manager import DataManager
from utils.helpers import now_ms, _safe_float, _sanitize_for_json, _calc_rsi, _load_env_file, _env_bool, _env_int, _env_float

# ---- PMaker survival model (optional) ----


from engines.pmaker_manager import PMakerManager

class LiveOrchestrator:
    def __init__(self, exchange, symbols=None):
        self.hub = EngineHub()
        # EngineHub loads MC engine automatically
        self.mc_engine_by_symbol = {}

        # ✅ Alpha Trainer
        self.alpha_enable = _env_bool("ALPHA_ENABLE", False)
        if self.alpha_enable and OnlineAlphaTrainer:
            self.alpha_trainer = OnlineAlphaTrainer(AlphaTrainerConfig())
        else:
            self.alpha_trainer = None

        # ✅ Configs for analyze_symbol
        self.atr_n = 14
        self.policy_min_ev_gap = 0.0002
        self.policy_w_ev_beta = 0.5
        self.pmaker_delay_penalty_k = 0.0001
        self.tp_r_by_h = {600: 0.004, 1800: 0.008}
        self.trail_atr_mult = 3.0
        self.sl_r_fixed = -0.005
        self.PMAKER_USE_SURVIVAL = True
        self.PMAKER_PREDICT_EXIT = True

        self.exchange = exchange
        self._net_sem = asyncio.Semaphore(MAX_INFLIGHT_REQ)
        self.symbols = symbols if symbols is not None else SYMBOLS
        
        # Initialize PMaker Manager
        self.pmaker = PMakerManager(self)
        self.dashboard = None # Set in main
        self.data = DataManager(self, self.symbols)
        self._last_ok = {"tickers": 0, "ohlcv": {s: 0 for s in SYMBOLS}, "ob": {s: 0 for s in SYMBOLS}}

        self.clients = set()
        self.logs = deque(maxlen=300)
        self.exec_stats = {}  # ✅ Initialize exec_stats for tracking execution statistics

        self.balance = 10_000.0
        self.positions = {}  # sym -> position dict (demo/paper)
        self.leverage = DEFAULT_LEVERAGE
        self.max_leverage = MAX_LEVERAGE
        self.enable_orders = ENABLE_LIVE_ORDERS
        self.max_positions = MAX_CONCURRENT_POSITIONS
        self.max_notional_frac = MAX_NOTIONAL_EXPOSURE
        self.position_cap_enabled = POSITION_CAP_ENABLED
        self.exposure_cap_enabled = EXPOSURE_CAP_ENABLED
        self.default_size_frac = DEFAULT_SIZE_FRAC
        # 수수료 설정 (Bybit taker/maker)
        self.fee_taker = BYBIT_TAKER_FEE
        self.fee_maker = BYBIT_MAKER_FEE
        self.fee_mode = "taker"  # 기본 taker 가정
        # ---- PMaker survival model (optional / online) ----
        # Canonical envs:
        #   PMAKER_ENABLE, PMAKER_MODEL_PATH, PMAKER_DEVICE, PMAKER_GRID_MS, PMAKER_MAX_MS,
        #   PMAKER_LR, PMAKER_TRAIN_STEPS, PMAKER_BATCH, PMAKER_CONSERVATIVE, PMAKER_DEBUG
        # Back-compat aliases:
        #   PMAKER_MLP_ENABLE -> PMAKER_ENABLE
        #   PMAKER_MODEL_PATH -> PMAKER_MODEL_PATH (same)
        # PMaker fields moved to self.pmaker

        # Autopilot wiring moved to self.pmaker

        # PMaker auto-start logic moved to PMakerManager

        self._decision_log_every = max(1, int(DECISION_LOG_EVERY))
        self._decision_cycle = 0
        self.spread_pairs = SPREAD_PAIRS
        self.spread_enabled = SPREAD_ENABLED
        self._last_actions = {}
        self._cooldown_until = {s: 0.0 for s in SYMBOLS}
        self._entry_streak = {s: 0 for s in SYMBOLS}
        self._last_exit_kind = {s: "NONE" for s in SYMBOLS}
        self._streak = {s: 0 for s in SYMBOLS}
        self._ev_tune_hist = {s: deque(maxlen=2000) for s in SYMBOLS}  # (ts, ev)
        self._ev_hist = {s: deque(maxlen=400) for s in SYMBOLS}
        self._cvar_hist = {s: deque(maxlen=400) for s in SYMBOLS}
        self._ofi_regime_hist: dict[tuple[str, str], deque] = {}
        self._ev_regime_hist: dict[tuple[str, str], deque] = {}
        self._ev_thr_ema: dict[tuple[str, str], float] = {}
        self._ev_thr_ema_ts: dict[tuple[str, str], int] = {}
        self._ev_drop_state: dict[str, dict] = {s: {"prev": None, "streak": 0} for s in SYMBOLS}
        self._spread_regime_hist: dict[tuple[str, str], deque] = {}
        self._liq_regime_hist: dict[tuple[str, str], deque] = {}
        self._ofi_resid_hist: dict[tuple[str, str], deque] = {}
        self._ev_regime_hist_rs: dict[tuple[str, str], deque] = {}
        self._cvar_regime_hist_rs: dict[tuple[str, str], deque] = {}
        self._ema_ev: dict[str, tuple[float, int]] = {}
        self._ema_psl: dict[str, tuple[float, int]] = {}
        self._exit_bad_ticks: dict[str, int] = {s: 0 for s in SYMBOLS}
        self._dyn_leverage = {s: self.leverage for s in SYMBOLS}
        self.trade_tape = deque(maxlen=20_000)
        self.eval_history = deque(maxlen=5_000)  # 예측 vs 실제 품질 평가용
        self._last_rows = []
        self._broadcast_cycle = 0
        self._last_decision_loop_log_ms = 0
        self._decision_cache: dict[str, tuple[int, dict]] = {}  # sym -> (ts_ms, decision)
        # ✅ 병렬 처리: 모든 심볼이 동시에 실행될 수 있도록 세마포어 제한을 심볼 수만큼 증가
        # DECISION_MAX_INFLIGHT가 명시적으로 설정되지 않으면 심볼 수만큼 허용
        sem_limit = int(DECISION_MAX_INFLIGHT) if DECISION_MAX_INFLIGHT > 0 else len(SYMBOLS)
        self._decision_sem = asyncio.Semaphore(max(1, sem_limit))
        print(f"[EV_DEBUG] decision_sem limit={self._decision_sem._value} (DECISION_MAX_INFLIGHT={DECISION_MAX_INFLIGHT}, SYMBOLS count={len(SYMBOLS)})")
        self._decision_tasks: dict[str, asyncio.Task] = {}
        self._loop_ms = None
        self._mc_ready = False
        self._exec_stats: dict[str, dict] = {}  # sym -> execution stats (maker fill/cancel/latency)
        self._pmaker_last_save_ms = 0
        self._pmaker_save_interval_ms = int(os.environ.get("PMAKER_SAVE_INTERVAL_SEC", "300")) * 1000
        self._pmaker_dirty = False

        self.exit_policy = ExitPolicy(
            min_event_ev_r=-0.0005,
            max_event_p_sl=0.55,
            min_event_p_tp=0.30,
            grace_sec=20,
            max_hold_sec=600,
            time_stop_mult=2.2,
            max_abs_event_cvar_r=0.010,
        )
        # ---- Exit mode (unify exit with entry logic) ----
        self._exit_mode = str(os.environ.get("EXIT_MODE", "unified")).strip().lower()
        self._exit_unified_enabled = self._exit_mode in ("unified", "entry", "entry_like", "symmetric")
        # When unified is enabled, disable extra legacy exits to avoid double-trigger / churn.
        self._exit_unified_disable_event_mc = _env_bool("EXIT_UNIFIED_DISABLE_EVENT_MC", True)
        self._exit_unified_disable_ev_drop = _env_bool("EXIT_UNIFIED_DISABLE_EV_DROP", True)
        self._exit_unified_disable_ema_exit = _env_bool("EXIT_UNIFIED_DISABLE_EMA_EXIT", True)

        # [TEST] unified_min_hold_sec를 25 → 180으로 변경 - 빠른 청산 테스트
        self._exit_unified_min_hold_sec = _env_float("EXIT_UNIFIED_MIN_HOLD_SEC", 180.0)
        self._exit_unified_bad_streak_need = _env_int("EXIT_UNIFIED_BAD_STREAK", 2)
        self._exit_unified_wait_streak_need = _env_int("EXIT_UNIFIED_WAIT_STREAK", 4)
        self._exit_unified_flip_streak_need = _env_int("EXIT_UNIFIED_FLIP_STREAK", 2)
        self._exit_unified_ev_drop = _env_float("EXIT_UNIFIED_EV_DROP", 0.0012)
        self._exit_unified_flip_margin = _env_float("EXIT_UNIFIED_FLIP_MARGIN", 0.0012)
        self._exit_unified_hold_ev_floor_raw = str(os.environ.get("EXIT_UNIFIED_HOLD_EV_FLOOR", "")).strip()
        self._exit_unified_hold_win_floor_raw = str(os.environ.get("EXIT_UNIFIED_HOLD_WIN_FLOOR", "")).strip()

        # Hard stop (still applies in unified mode, but with grace + configurable)
        self._exit_stop_grace_sec = _env_float("EXIT_STOP_GRACE_SEC", 20.0)
        self._exit_stop_roe = _env_float("EXIT_STOP_ROE", -0.08)
        self.mc_cache = {}  # (sym, side, regime, price_bucket) -> (ts, meta)
        self.mc_cache_ttl = 2.0  # seconds


        # 1m close buffer (preload로 한 번에 채움)

        # orderbook 상태(대시보드 표기용)

        # OHLCV freshness / dedupe

        self._equity_history = deque(maxlen=20_000)
        # persistence
        self.state_dir = BASE_DIR / "state"
        self.state_dir.mkdir(exist_ok=True)
        self.state_files = {
            "equity": self.state_dir / "equity_history.json",
            "trade": self.state_dir / "trade_tape.json",
            "eval": self.state_dir / "eval_history.json",
            "positions": self.state_dir / "positions.json",
            "balance": self.state_dir / "balance.json",
        }
        self._last_state_persist_ms = 0
        self._load_persistent_state()
        
        # --- NEW: AlphaHit online trainer (GPU) ---
        self.alpha_enable = os.getenv("ALPHA_HIT_ENABLE", "1") == "1"
        self.alpha_horizons = [60, 180, 300, 600, 900, 1800]  # (3s,10s 제외)
        # feature dim MUST match build_alpha_features()
        self.alpha_feat_dim = int(os.getenv("ALPHA_HIT_FEAT_DIM", "12"))
        self.alpha_lr = float(os.getenv("ALPHA_HIT_LR", "0.0002"))
        self.alpha_bs = int(os.getenv("ALPHA_HIT_BS", "256"))
        self.alpha_steps_per_tick = int(os.getenv("ALPHA_HIT_STEPS_PER_TICK", "2"))
        self.alpha_data_half_life_sec = float(os.getenv("ALPHA_HIT_DATA_HALF_LIFE_SEC", "3600"))
        self.alpha_ckpt_path = os.getenv("ALPHA_HIT_CKPT", "state/alpha_hit_mlp.pt")

        self.alpha_trainer = OnlineAlphaTrainer(AlphaTrainerConfig(
            horizons_sec=self.alpha_horizons,
            n_features=self.alpha_feat_dim,
            lr=self.alpha_lr,
            batch_size=self.alpha_bs,
            steps_per_tick=self.alpha_steps_per_tick,
            data_half_life_sec=self.alpha_data_half_life_sec,
            ckpt_path=self.alpha_ckpt_path,
            enable=self.alpha_enable,
        ))

        # --- knobs for EV integration ---
        self.policy_min_ev_gap = float(os.getenv("POLICY_MIN_EV_GAP", "0.005"))
        self.policy_w_ev_beta = float(os.getenv("POLICY_W_EV_BETA", "200"))
        self.pmaker_delay_penalty_k = float(os.getenv("PMAKER_DELAY_PENALTY_K", "1.0"))
        
        # [DIFF 6] PMaker survival prediction settings
        self.PMAKER_USE_SURVIVAL = os.getenv("PMAKER_USE_SURVIVAL", "1") == "1"
        self.PMAKER_DELAY_PENALTY_MULT = float(os.getenv("PMAKER_DELAY_PENALTY_MULT", "1.0"))
        self.PMAKER_SURVIVAL_DEBUG = os.getenv("PMAKER_SURVIVAL_DEBUG", "0") == "1"
        # entry/exit 분리 prediction on/off
        self.PMAKER_PREDICT_EXIT = os.getenv("PMAKER_PREDICT_EXIT", "1") == "1"

        # TP table per horizon (fractional return)
        # Provide env like: TP_R_BY_H="60:0.0012,180:0.0018,300:0.0024,600:0.0032,900:0.0040,1800:0.0060"
        self.tp_r_by_h = {}
        s = os.getenv("TP_R_BY_H", "")
        if s:
            for kv in s.split(","):
                try:
                    k, v = kv.split(":")
                    self.tp_r_by_h[int(k.strip())] = float(v.strip())
                except Exception:
                    pass

        # ATR trailing params (used for SL_r approximation and real exit)
        self.atr_tf_sec = int(os.getenv("ATR_TF_SEC", "60"))
        self.atr_n = int(os.getenv("ATR_N", "14"))
        self.trail_atr_mult = float(os.getenv("TRAIL_ATR_MULT", "2.0"))
        self.sl_r_fixed = float(os.getenv("SL_R_FIXED", "0.0020"))
        
        # MC engine per symbol (for injecting alpha predictions)
        self.mc_engine_by_symbol = {}
        
        # 러닝 통계
        self.stats = RunningStats(maxlen=5000)

    def _compute_atr_proxy(self, closes: np.ndarray, n: int = 14) -> float:
        """
        Compute ATR proxy from closes using EMA of absolute price changes.
        Falls back to simple EMA if not enough data.
        """
        if len(closes) < 2:
            return 0.0
        try:
            # Use absolute price changes as proxy for true range
            changes = np.abs(np.diff(closes))
            if len(changes) < n:
                # Simple average if not enough data
                return float(np.mean(changes)) if len(changes) > 0 else 0.0
            # EMA of changes
            alpha = 2.0 / (n + 1.0)
            ema = changes[0]
            for c in changes[1:]:
                ema = alpha * c + (1.0 - alpha) * ema
            return float(ema)
        except Exception:
            return 0.0

    def _exec_stats_for(self, sym: str) -> dict:
        st = self._exec_stats.get(sym)
        if st is None:
            st = {
                "ts0": now_ms(),
                "order_requests": 0,
                "order_requests_market": 0,
                "order_requests_maker_path": 0,
                "order_requests_maker_filled": 0,
                "order_requests_fallback_market": 0,
                "maker_limit_orders": 0,
                "maker_limit_filled": 0,
                "maker_limit_timeout": 0,
                "maker_limit_cancel_ok": 0,
                "maker_limit_cancel_err": 0,
                "maker_reject": 0,
                "maker_other_err": 0,
                "lat_ms_maker_filled": deque(maxlen=500),
                "lat_ms_maker_first_fill": deque(maxlen=500),
                "lat_ms_fallback_market": deque(maxlen=500),
            }
            self._exec_stats[sym] = st
        return st



    def _rows_snapshot(self, ts: int) -> list[dict]:
        rows: list[dict] = []
        for sym in self.symbols:
            try:
                price = self.data.market.get(sym, {}).get("price")
                candles = len(list(self.data.ohlcv_buffer.get(sym) or []))
                cached = self._decision_cache.get(sym)
                decision = cached[1] if cached else None
                rows.append(self._row(sym, price, ts, decision, candles))
            except Exception:
                rows.append(self._row(sym, self.data.market.get(sym, {}).get("price"), ts, None, len(list(self.data.ohlcv_buffer.get(sym) or []))))
        return rows

    async def broadcast_loop(self):
        while True:
            try:
                if not self._mc_ready:
                    await asyncio.sleep(0.1)
                    continue
                ts = now_ms()
                # ✅ 항상 최신 캐시에서 읽어서 반영 (0.5초마다 최신 데이터로 업데이트)
                rows = self._rows_snapshot(ts)  # _last_rows 대신 항상 최신 캐시에서 읽기
                await self.dashboard.broadcast(rows)
                # ✅ broadcast 후 _last_rows 업데이트 (신규 WS 연결 시 사용)
                self._last_rows = rows
            except Exception as e:
                print(f"[ERR] broadcast_loop: {e}")
                self._log_err(f"[ERR] broadcast_loop: {e}")
            await asyncio.sleep(0.1)  # ✅ 0.5초 → 0.1초로 단축 (더 빠른 반영)

    async def analyze_symbol(self, sym: str, ts: int, log_this_cycle: bool) -> dict:
        """
        심볼별 분석을 수행하는 비동기 함수.
        각 심볼의 MC 시뮬레이션 및 의사결정을 병렬로 처리하기 위해 사용됩니다.
        
        Returns:
            dict: row_data (심볼 분석 결과)
        """
        try:
            price = self.data.market[sym].get("price")
            closes = list(self.data.ohlcv_buffer[sym])
            candles = len(closes)

            # ✅ price가 None이어도 row는 추가 (대시보드에 데이터 표시)
            if price is None:
                return self._row(sym, None, ts, None, candles)

            # ✅ Step 5: mu/sigma 업데이트를 decision 계산 전에 먼저 수행 (타이밍 문제 해결)
            mu_bar, sigma_bar = self._compute_returns_and_vol(closes)
            
            # ✅ mu_bar/sigma_bar가 None이거나 0이면 기본값 사용 (모든 심볼 반영 보장)
            # ✅ closes가 부족하면 기본값 사용 (warm-up 기간)
            if mu_bar is None or sigma_bar is None or sigma_bar <= 0:
                closes_len = len(closes) if closes else 0
                if sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
                    self._log(f"[WARMUP] {sym} | mu_bar={mu_bar}, sigma_bar={sigma_bar}, closes_len={closes_len} - using defaults (will improve as data accumulates)")
                # 기본값 사용하여 계속 진행 (모든 심볼이 반영되도록)
                # closes가 부족하면 기본값 사용, 시간이 지나면서 데이터가 쌓이면 정상 계산됨
                mu_bar = mu_bar if mu_bar is not None else 0.0
                sigma_bar = max(0.0001, sigma_bar if sigma_bar is not None else 0.01)  # 최소값 보장
            
            regime = self._infer_regime(closes)
            
            # ✅ Step 1: returns 계산 정보 추적
            closes_len = len(closes) if closes else 0
            rets_n = None
            if closes and len(closes) >= 2:
                rets_n = len(closes) - 1  # log returns 개수

            spread_pct = None
            ob = self.data.orderbook.get(sym)
            bid = None
            ask = None
            bids = None
            asks = None
            if ob and ob.get("ready"):
                bids = ob.get("bids") or []
                asks = ob.get("asks") or []
                if bids and asks and len(bids) > 0 and len(asks) > 0 and len(bids[0]) >= 2 and len(asks[0]) >= 2:
                    bid = float(bids[0][0])
                    ask = float(asks[0][0])
                    mid = (bid + ask) / 2.0 if (bid and ask) else 0.0
                    if mid > 0:
                        spread_pct = (ask - bid) / mid
            else:
                # ✅ orderbook이 준비되지 않은 경우 로그 출력 (원인 파악용)
                if sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
                    self._log(f"[ORDERBOOK_WARN] {sym} | orderbook not ready - ob={ob is not None} ready={ob.get('ready') if ob else False} ts={ob.get('ts') if ob else None}")
            if (bid is None or ask is None):
                ticker = self.data.market.get(sym, {})
                t_bid = ticker.get("bid")
                t_ask = ticker.get("ask")
                if t_bid is not None and t_ask is not None and t_bid > 0 and t_ask > 0:
                    bid = float(t_bid)
                    ask = float(t_ask)
                    mid = (bid + ask) / 2.0
                    if mid > 0:
                        spread_pct = (ask - bid) / mid

            # 1) realized (annualized)
            mu_base, sigma = (0.0, 0.0)
            vol_src_main = "none"
            if mu_bar is not None and sigma_bar is not None:
                mu_base, sigma = self._annualize_mu_sigma(mu_bar, sigma_bar, bar_seconds=60.0)
                vol_src_main = f"realized_n={rets_n}"

            # 2) regime μ/σ table blend (session aware)
            session = time_regime()
            mu_tab, sig_tab = get_regime_mu_sigma(regime, session, symbol=sym)
            if mu_tab is not None and sig_tab is not None:
                w = 0.35
                mu_base = float((1.0 - w) * float(mu_base) + w * float(mu_tab))
                sigma = float(max(1e-6, (1.0 - w) * float(sigma) + w * float(sig_tab)))
                vol_src_main = f"{vol_src_main}_blended_with_regime_table(w={w})"
            
            # ✅ Step 1: mu_base, sigma 계산 결과 로그 (필요 시에만)
            if DEBUG_MU_SIGMA and sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
                self._log(
                    f"[MU_SIGMA_ORCH] {sym} | "
                    f"mu_bar={mu_bar}, sigma_bar={sigma_bar} | "
                    f"mu_base={mu_base:.8f}, sigma={sigma:.8f} | "
                    f"closes_len={closes_len}, rets_n={rets_n}, vol_src={vol_src_main} | "
                    f"regime={regime}, session={session}"
                )

            ofi_score = float(self._compute_ofi_score(sym))
            tuner = getattr(self, "tuner", None)
            if tuner and hasattr(tuner, "get_params"):
                try:
                    regime_params = tuner.get_params(regime)
                except Exception:
                    regime_params = None
            else:
                regime_params = None
            # MC 경량화: live에서는 n_paths를 낮춰 UI/루프가 멈추지 않게 한다.
            # (필요하면 env MC_N_PATHS_LIVE로 조정)
            ctx_n_paths = int(max(200, min(200000, int(MC_N_PATHS_LIVE))))
            if isinstance(regime_params, dict):
                # tuner가 n_paths를 크게 줘도 live override가 우선하도록 강제
                regime_params = dict(regime_params)
                regime_params["n_paths"] = ctx_n_paths

            # bootstrap returns for tail (optional)
            bootstrap_returns = None
            if closes is not None and len(closes) >= 64:
                try:
                    x = np.asarray(closes, dtype=np.float64)
                    bootstrap_returns = np.diff(np.log(np.maximum(x, 1e-12))).astype(np.float64)[-512:]
                except Exception:
                    bootstrap_returns = None

            # ✅ Step 1: mu_sim, sigma_sim을 명시적으로 전달 (mu_base, sigma와 동일값이지만 명확성을 위해)
            mu_sim_ctx = float(mu_base)
            sigma_sim_ctx = float(max(sigma, 0.0))
            
            # [D] Build features (CPU) and predict TP/SL hit probs (GPU)
            closes_arr = np.asarray(closes, dtype=np.float64)
            vols_arr = np.asarray([c.get("volume", 0.0) if isinstance(c, dict) else 0.0 for c in closes], dtype=np.float64) if closes else np.zeros_like(closes_arr)
            rets_arr = np.diff(closes_arr) if len(closes_arr) > 1 else np.array([])
            
            # orderbook spread_pct (20bp cap is your rule)
            spread_pct_book = float(spread_pct) if spread_pct is not None else 0.0
            
            # ofi_z / regime_id
            ofi_z = float(ofi_score)  # OFI score as z-score proxy
            regime_id_map = {"bull": 1.0, "bear": -1.0, "chop": 0.0, "volatile": 0.5}
            regime_id = float(regime_id_map.get(regime, 0.0))
            
            # ATR fraction for SL scaling (ATR/price) – compute from closes as proxy
            atr = self._compute_atr_proxy(closes_arr, n=self.atr_n)
            atr_frac = float(atr / max(1e-12, closes_arr[-1])) if len(closes_arr) > 0 else 0.0
            
            # Build ctx first (needed for pmaker prediction)
            ctx = {
                "symbol": sym,
                "btc_corr": self.data.get_btc_corr(sym),  # ✅ BTC 상관관계 추가
                "price": float(price),
                "bar_seconds": 60.0,
                "closes": closes,
                "direction": self._direction_bias(closes),
                "regime": regime,
                "ofi_score": float(ofi_score),
                "liquidity_score": self._liquidity_score(sym),
                "leverage": None,  # placeholder; set below
                "mu_base": float(mu_base),
                "sigma": float(max(sigma, 0.0)),
                "mu_sim": mu_sim_ctx,  # ✅ Step 1: 명시적 전달
                "sigma_sim": sigma_sim_ctx,  # ✅ Step 1: 명시적 전달
                "regime_params": regime_params,
                "n_paths": ctx_n_paths,
                "session": session,
                "spread_pct": spread_pct,
                # provide best bid/ask for execution models (optional)
                "bid": bid,
                "ask": ask,
                # ---- MC tail + JAX knobs ----
                "use_jax": True,
                "tail_mode": "student_t",
                "tail_model": "student_t",
                "tail_df": 6.0,
                "student_t_df": 6.0,
                "bootstrap_returns": bootstrap_returns,
                "ev": None,  # filled post decision
                # ✅ Step 1/2: 디버깅용 정보 추가
                "rets_n": rets_n,  # returns 샘플 수
                "vol_src": vol_src_main,  # vol 추정 소스
                "atr_frac": atr_frac,  # [D] Add ATR fraction to ctx
                # ✅ PMaker values: will be set below
                "pmaker_entry": 0.0,
                "pmaker_entry_delay_sec": 0.0,
                "pmaker_exit": 0.0,
                "pmaker_exit_delay_sec": 0.0,
                # ✅ [FIX 2] PMaker survival model 전달 (mu_alpha boost에 사용)
                "pmaker_surv": self.pmaker.surv if self.pmaker.enabled and self.pmaker.surv is not None else None,
            }

            # [DIFF 7] PMaker prediction (used by mc_engine cost model) must be present at decision time,
            # not only after placing an order.
            # Always set pmaker_entry/pmaker_exit in ctx, even if bid/ask are invalid (use defaults)
            pmaker_entry = 0.0
            pmaker_delay_sec = 0.0
            print(f"[PMAKER_DEBUG] {sym} | _decision_loop: bid={bid} ask={ask} price={price} _pmaker_enabled={self.pmaker.enabled} pmaker_surv={self.pmaker.surv is not None} PMAKER_USE_SURVIVAL={self.PMAKER_USE_SURVIVAL}")
            # Try prediction if bid/ask available, or fallback to price-based estimation
            use_price_fallback = False
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                spread_pct_val = (float(ask) - float(bid)) / float(bid) if bid > 0 else 0.0
                best_bid_val = float(bid)
                best_ask_val = float(ask)
                entry_price = float(bid)  # Assume buy entry
            elif price is not None and price > 0:
                # Fallback: use price with estimated spread
                use_price_fallback = True
                spread_pct_val = float(spread_pct) if spread_pct is not None and spread_pct > 0 else 0.0002  # default 2bp
                estimated_spread = price * spread_pct_val
                best_bid_val = price - estimated_spread / 2.0
                best_ask_val = price + estimated_spread / 2.0
                entry_price = best_bid_val  # Use estimated bid for buy entry
                print(f"[PMAKER_DEBUG] {sym} | Using price fallback: price={price} spread_pct={spread_pct_val:.6f} bid={best_bid_val:.2f} ask={best_ask_val:.2f}")
            else:
                best_bid_val = 0.0
                best_ask_val = 0.0
                spread_pct_val = 0.0002
                entry_price = 0.0
            
            if entry_price > 0 and best_bid_val > 0 and best_ask_val > 0:
                # ✅ [FIX 1] PMaker ctx 전달 개선: fallback_sync=False일 때도 최소한의 동기 실행 또는 통계 기반 fallback 제공
                # 먼저 캐시/비슷한 캐시를 시도하고, 없으면 동기 실행
                pm = await self.pmaker.request_prediction(
                    symbol=sym,
                    side="buy",  # entry side (will be determined by direction, but default to buy)
                    price=entry_price,
                    best_bid=best_bid_val,
                    best_ask=best_ask_val,
                    spread_pct=spread_pct_val,
                    qty=1.0,  # placeholder qty
                    maker_timeout_ms=int(os.environ.get("MAKER_TIMEOUT_MS", str(MAKER_TIMEOUT_MS))),
                    prefix="pmaker_entry",
                    use_cache=True,
                    fallback_sync=True,  # ✅ FIX: 캐시 없을 때도 동기 실행하여 결과 보장
                )
                if os.getenv("PMAKER_DEBUG", "0") == "1":
                    print(f"[PMAKER_DEBUG] {sym} | _pmaker_request_prediction (entry) returned: {pm} (fallback={use_price_fallback})")
                if pm:
                    pmaker_entry = float(pm.get("pmaker_entry", 0.0))
                    pmaker_delay_sec = float(pm.get("pmaker_entry_delay_sec", 0.0))
                    ctx["pmaker_entry"] = pmaker_entry
                    ctx["pmaker_entry_delay_sec"] = pmaker_delay_sec
                    ctx.setdefault("meta", {}).update(pm)
                    if os.getenv("PMAKER_DEBUG", "0") == "1":
                        print(f"[PMAKER_DEBUG] {sym} | ctx pmaker_entry={ctx['pmaker_entry']:.4f} delay={ctx['pmaker_entry_delay_sec']:.4f}")
                else:
                    # ✅ FIX: PMaker 통계 기반 fallback (sym_fill_mean 사용)
                    if self.pmaker.surv is not None:
                        try:
                            # 심볼별 평균 fill rate를 사용하여 기본값 제공
                            fill_rate_estimate = self.pmaker.surv.sym_fill_mean(sym)
                            # fill_rate를 기반으로 예상 delay 추정 (경험적 공식)
                            # fill_rate가 높으면 delay가 짧고, 낮으면 delay가 길다
                            estimated_delay_sec = max(0.5, min(5.0, 3.0 / max(0.1, fill_rate_estimate)))
                            ctx["pmaker_entry"] = float(fill_rate_estimate)
                            ctx["pmaker_entry_delay_sec"] = float(estimated_delay_sec)
                            if os.getenv("PMAKER_DEBUG", "0") == "1":
                                print(f"[PMAKER_DEBUG] {sym} | Using PMaker stats fallback: fill_rate={fill_rate_estimate:.4f} delay={estimated_delay_sec:.2f}s")
                        except Exception as e:
                            # 통계 fallback도 실패하면 기본값 사용
                            ctx["pmaker_entry"] = 0.0
                            ctx["pmaker_entry_delay_sec"] = 0.0
                            if os.getenv("PMAKER_DEBUG", "0") == "1":
                                print(f"[PMAKER_DEBUG] {sym} | PMaker stats fallback failed: {e}, using defaults")
                    else:
                        # Fallback: set default values
                        ctx["pmaker_entry"] = 0.0
                        ctx["pmaker_entry_delay_sec"] = 0.0
                        if os.getenv("PMAKER_DEBUG", "0") == "1":
                            print(f"[PMAKER_DEBUG] {sym} | No cached prediction and pmaker_surv is None, using defaults")
            else:
                print(f"[PMAKER_DEBUG] {sym} | Cannot predict: entry_price={entry_price} best_bid={best_bid_val} best_ask={best_ask_val}")
                ctx["pmaker_entry"] = 0.0
                ctx["pmaker_entry_delay_sec"] = 0.0
            
            # Build features and predict (using pmaker_entry from prediction above)
            # ✅ pmaker_entry는 ctx에서 가져오기 (예측 결과가 ctx에 저장됨)
            pmaker_entry_for_features = float(ctx.get("pmaker_entry", 0.0))
            pmaker_delay_sec_for_features = float(ctx.get("pmaker_entry_delay_sec", 0.0))
            x = build_alpha_features(
                closes=closes_arr,
                vols=vols_arr,
                returns=rets_arr,
                ofi_z=ofi_z,
                spread_pct=spread_pct_book,
                pmaker_entry=pmaker_entry_for_features,
                pmaker_delay_sec=pmaker_delay_sec_for_features,
                regime_id=regime_id,
            )
            pred = self.alpha_trainer.predict(x) if self.alpha_enable else None
            
            # [D] Inject to MC engine (per-symbol context)
            # Get MC engine from hub
            mc = None
            for engine in self.hub.engines:
                from engines.mc_engine import MonteCarloEngine
                if isinstance(engine, MonteCarloEngine):
                    mc = engine
                    break
            if mc is not None:
                mc._alpha_hit_pred = pred
                # ✅ ctx에서 pmaker_entry 가져오기 (예측 결과가 ctx에 저장됨)
                mc._pmaker_entry = float(ctx.get("pmaker_entry", 0.0))
                mc._pmaker_entry_delay_sec = float(ctx.get("pmaker_entry_delay_sec", 0.0))
                mc._atr_frac = atr_frac
                mc.POLICY_MIN_EV_GAP = self.policy_min_ev_gap
                mc.POLICY_W_EV_BETA = self.policy_w_ev_beta
                mc.PMAKER_DELAY_PENALTY_K = self.pmaker_delay_penalty_k
                mc.TP_R_BY_H = self.tp_r_by_h
                mc.TRAIL_ATR_MULT = self.trail_atr_mult
                mc.SL_R_FIXED = self.sl_r_fixed
                # Store MC engine per symbol for later access
                self.mc_engine_by_symbol[sym] = mc
                
                # [DIFF 6] Also add pmaker_exit prediction (for exit delay penalty)
                # Always try exit prediction if enabled, even if entry prediction failed
                if pm or True:  # Always set exit values
                    # Exit prediction uses same survival model but with exit-specific features
                    if self.PMAKER_PREDICT_EXIT and bid is not None and ask is not None:
                        try:
                            # For exit, we predict at current market price (exit would be opposite side)
                            # Use ask for long exit (sell), bid for short exit (buy)
                            exit_price = float(ask)  # Assume long position exit (sell at ask)
                            # ✅ FIX: exit 예측도 동기 실행으로 변경 (결과 보장)
                            pm_exit = await self.pmaker.request_prediction(
                                symbol=sym,
                                side="sell",  # exit side (opposite of entry)
                                price=exit_price,
                                best_bid=float(bid) if bid is not None else 0.0,
                                best_ask=float(ask) if ask is not None else 0.0,
                                spread_pct=spread_pct_val if spread_pct_val is not None else (ctx.get("spread_pct") or 0.0),
                                qty=1.0,  # placeholder qty
                                maker_timeout_ms=int(os.environ.get("MAKER_TIMEOUT_MS", str(MAKER_TIMEOUT_MS))),
                                prefix="pmaker_exit",
                                use_cache=True,
                                fallback_sync=True,  # ✅ FIX: 캐시 없을 때도 동기 실행하여 결과 보장
                            )
                            if pm_exit:
                                ctx["pmaker_exit"] = float(pm_exit.get("pmaker_exit", 0.0))
                                ctx["pmaker_exit_delay_sec"] = float(pm_exit.get("pmaker_exit_delay_sec", 0.0))
                                ctx.setdefault("meta", {}).update(pm_exit)
                                print(f"[PMAKER_DEBUG] {sym} | ctx pmaker_exit={ctx['pmaker_exit']:.4f} delay={ctx['pmaker_exit_delay_sec']:.4f}")
                                self._log(f"[PMAKER_DEBUG] {sym} | ctx pmaker_exit={ctx['pmaker_exit']:.4f} delay={ctx['pmaker_exit_delay_sec']:.4f}")
                            else:
                                print(f"[PMAKER_DEBUG] {sym} | pm_exit is None")
                                self._log(f"[PMAKER_DEBUG] {sym} | pm_exit is None")
                        except Exception as e:
                            print(f"[PMAKER_DEBUG] {sym} | exit prediction failed: {e}")
                            self._log_err(f"[PMAKER_DEBUG] {sym} | exit prediction failed: {e}")
                    else:
                        print(f"[PMAKER_DEBUG] {sym} | PMAKER_PREDICT_EXIT={self.PMAKER_PREDICT_EXIT} bid={bid} ask={ask}")
                        self._log(f"[PMAKER_DEBUG] {sym} | PMAKER_PREDICT_EXIT={self.PMAKER_PREDICT_EXIT} bid={bid} ask={ask}")
                        # Set default exit values if exit prediction is disabled
                        ctx["pmaker_exit"] = ctx.get("pmaker_entry", 0.0)
                        ctx["pmaker_exit_delay_sec"] = ctx.get("pmaker_entry_delay_sec", 0.0)
                
                    # Update meta with pm if it exists
                    if pm:
                        ctx.setdefault("meta", {}).update(pm)
                    else:
                        # Even if pm is None, ensure meta has pmaker_entry keys
                        ctx.setdefault("meta", {})["pmaker_entry"] = ctx.get("pmaker_entry", 0.0)
                        ctx.setdefault("meta", {})["pmaker_entry_delay_sec"] = ctx.get("pmaker_entry_delay_sec", 0.0)
                        # [C] Pass PMaker survival model and features to MC engine for horizon-specific delay penalty
                        if self.pmaker.surv is not None:
                            try:
                                # Extract PMaker features for survival prediction
                                pmaker_feats = self.pmaker._extract_feats(
                                    sym=sym,
                                    order_side="buy",  # placeholder, will be determined by direction
                                    maker_price=float(bid) if bid else 0.0,
                                    decision=None,
                                    attempt_idx=0,
                                    bid=bid,
                                    ask=ask,
                                    sigma=float(ctx.get("sigma_sim")) if ctx.get("sigma_sim") is not None else None,
                                )
                                pmaker_features_tensor = self.pmaker.surv.featurize(pmaker_feats)
                                ctx["pmaker_surv"] = self.pmaker.surv
                                ctx["pmaker_features"] = pmaker_features_tensor
                                ctx["pmaker_timeout_ms"] = int(os.environ.get("MAKER_TIMEOUT_MS", str(MAKER_TIMEOUT_MS)))
                            except Exception as e:
                                self._log_err(f"[PMAKER_CTX] Failed to prepare PMaker features: {e}")
            else:
                # If bid/ask are invalid, set default values to ensure ctx always has these keys
                print(f"[PMAKER_DEBUG] {sym} | bid/ask invalid (bid={bid} ask={ask}), setting default pmaker values")
                ctx["pmaker_entry"] = 0.0
                ctx["pmaker_entry_delay_sec"] = 0.0
                ctx["pmaker_exit"] = 0.0
                ctx["pmaker_exit_delay_sec"] = 0.0
                ctx.setdefault("meta", {})["pmaker_entry"] = 0.0
                ctx.setdefault("meta", {})["pmaker_entry_delay_sec"] = 0.0
                print(f"[PMAKER_DEBUG] {sym} | ctx keys after default pmaker values: {list(ctx.keys())}")

            # ✅ MC 의사결정은 무겁기 때문에:
            # - DECISION_REFRESH_SEC 주기로만 task를 스케줄링(중간에는 캐시 사용)
            # - task는 thread에서 실행 (이벤트루프 블로킹 방지)
            decision = None
            cached = self._decision_cache.get(sym)
            if cached is not None:
                decision = dict(cached[1])
                print(f"[EV_DEBUG] {sym} | _decision_loop: Using cached decision: ev={decision.get('ev') if isinstance(decision, dict) else None}")
            else:
                print(f"[EV_DEBUG] {sym} | _decision_loop: No cached decision, _decision_cache is empty")

            need_refresh = True
            if cached is not None:
                cached_ts = int(cached[0])
                cached_decision = cached[1]
                cached_ev = cached_decision.get('ev') if isinstance(cached_decision, dict) else None
                # ✅ 데이터 갱신 주기와 동기화: 데이터가 갱신되었을 때만 새로 계산
                # 캐시 유효 시간은 데이터 갱신 주기(ORDERBOOK_SLEEP_SEC, TICKER_SLEEP_SEC)와 맞춤
                # 최소 0.1초, 최대 DECISION_REFRESH_SEC * 0.5
                cache_ttl_ms = int(max(0.1, min(float(DECISION_REFRESH_SEC) * 0.5, float(ORDERBOOK_SLEEP_SEC) * 1.5)) * 1000)
                need_refresh = (ts - cached_ts) >= cache_ttl_ms
                print(f"[EV_DEBUG] {sym} | _decision_loop: cached decision exists: ev={cached_ev} cached_ts={cached_ts} current_ts={ts} cache_ttl_ms={cache_ttl_ms} need_refresh={need_refresh}")
            else:
                # ✅ 캐시가 없으면 즉시 새로 계산해야 함
                need_refresh = True
                print(f"[EV_DEBUG] {sym} | _decision_loop: No cached decision, need_refresh=True")

            tsk = self._decision_tasks.get(sym)
            if need_refresh and (tsk is None or tsk.done()):
                # ✅ ctx에 필수 데이터가 있는지 확인
                mu_base_check = ctx.get('mu_base')
                sigma_check = ctx.get('sigma')
                price_check = ctx.get('price')
                print(f"[EV_DEBUG] {sym} | _decision_loop: Creating _compute_decision_task: need_refresh={need_refresh} tsk=None={tsk is None} tsk.done={tsk.done() if tsk else 'N/A'}")
                print(f"[EV_DEBUG] {sym} | _decision_loop: ctx data check - mu_base={mu_base_check} sigma={sigma_check} price={price_check}")
                # ✅ mu_base/sigma가 없어도 기본값으로 task 생성 (모든 심볼 반영 보장)
                if mu_base_check is None or sigma_check is None or sigma_check <= 0:
                    print(f"[EV_DEBUG] {sym} | ⚠️  WARNING: ctx missing required data! mu_base={mu_base_check} sigma={sigma_check} - using defaults")
                    # 기본값으로 ctx 업데이트
                    if mu_base_check is None:
                        ctx['mu_base'] = 0.0
                        ctx['mu_sim'] = 0.0
                    if sigma_check is None or sigma_check <= 0:
                        ctx['sigma'] = 0.01
                        ctx['sigma_sim'] = 0.01
                    self._log(f"[EV_DEBUG] {sym} | Using default values: mu_base={ctx.get('mu_base')} sigma={ctx.get('sigma')}")
                # Debug: log ctx keys and pmaker_entry value before calling _compute_decision_task
                # Check ctx value right before creating task
                pmaker_entry_before_task = ctx.get('pmaker_entry')
                pmaker_entry_delay_sec_before_task = ctx.get('pmaker_entry_delay_sec')
                print(f"[PMAKER_DEBUG] {sym} | _decision_loop: ctx keys before _compute_decision_task: {list(ctx.keys())} pmaker_entry={pmaker_entry_before_task} (type={type(pmaker_entry_before_task)}) pmaker_entry_delay_sec={pmaker_entry_delay_sec_before_task} (type={type(pmaker_entry_delay_sec_before_task)})")
                self._log(f"[PMAKER_DEBUG] {sym} | _decision_loop: ctx keys before _compute_decision_task: {list(ctx.keys())} pmaker_entry={pmaker_entry_before_task} pmaker_entry_delay_sec={pmaker_entry_delay_sec_before_task}")
                # Store ctx id to track if it's the same dict
                ctx_id = id(ctx)
                print(f"[PMAKER_DEBUG] {sym} | _decision_loop: ctx id={ctx_id} before creating task")
                self._decision_tasks[sym] = asyncio.create_task(self._compute_decision_task(sym, ctx, ts))
                # Check ctx value right after creating task (to see if it was modified)
                pmaker_entry_after_task = ctx.get('pmaker_entry')
                pmaker_entry_delay_sec_after_task = ctx.get('pmaker_entry_delay_sec')
                ctx_id_after = id(ctx)
                print(f"[PMAKER_DEBUG] {sym} | _decision_loop: ctx id={ctx_id_after} after creating task (same={ctx_id == ctx_id_after}) pmaker_entry={pmaker_entry_after_task} (changed={pmaker_entry_before_task != pmaker_entry_after_task}) pmaker_entry_delay_sec={pmaker_entry_delay_sec_after_task} (changed={pmaker_entry_delay_sec_before_task != pmaker_entry_delay_sec_after_task})")
                print(f"[EV_DEBUG] {sym} | _decision_loop: Task created successfully")
                # ✅ 캐시가 없으면 task가 완료될 때까지 기다려서 decision을 가져옴
                if cached is None:
                    print(f"[EV_DEBUG] {sym} | _decision_loop: No cache, waiting for task to complete...")
                    try:
                        new_task = self._decision_tasks[sym]
                        await new_task
                        # Task 완료 후 캐시에서 decision 가져오기
                        cached_after_task = self._decision_cache.get(sym)
                        if cached_after_task is not None:
                            decision = dict(cached_after_task[1])
                            print(f"[EV_DEBUG] {sym} | _decision_loop: Task completed, got decision from cache: ev={decision.get('ev') if isinstance(decision, dict) else None}")
                        else:
                            print(f"[EV_DEBUG] {sym} | _decision_loop: Task completed but cache still empty")
                    except Exception as e:
                        print(f"[EV_DEBUG] {sym} | _decision_loop: Error waiting for task: {e}")
                        self._log_err(f"[EV_DEBUG] {sym} | Error waiting for task: {e}")
            elif not need_refresh:
                print(f"[EV_DEBUG] {sym} | _decision_loop: Skipping task creation - need_refresh=False (cache still valid)")
            elif tsk is not None and not tsk.done():
                print(f"[EV_DEBUG] {sym} | _decision_loop: Task already running, waiting for completion...")
                # ✅ task가 실행 중이면 완료될 때까지 기다림
                try:
                    await tsk
                    # Task 완료 후 캐시에서 decision 가져오기
                    cached_after_task = self._decision_cache.get(sym)
                    if cached_after_task is not None:
                        decision = dict(cached_after_task[1])
                        print(f"[EV_DEBUG] {sym} | _decision_loop: Task completed, got decision from cache: ev={decision.get('ev') if isinstance(decision, dict) else None}")
                    else:
                        print(f"[EV_DEBUG] {sym} | _decision_loop: Task completed but cache still empty")
                except Exception as e:
                    print(f"[EV_DEBUG] {sym} | _decision_loop: Error waiting for task: {e}")
                    self._log_err(f"[EV_DEBUG] {sym} | Error waiting for task: {e}")
            
            # ✅ EV_DEBUG: decision 상태 확인
            if decision:
                decision_ev = decision.get('ev')
                print(f"[EV_DEBUG] {sym} | _decision_loop: decision exists: ev={decision_ev} (type={type(decision_ev)})")
            else:
                print(f"[EV_DEBUG] {sym} | _decision_loop: decision is None!")
            
            # ✅ Step 4: mu_bar/sigma_bar를 meta에 명시적으로 주입 (키 mismatch 방지)
            if decision:
                decision_meta = decision.get("meta") or {}
                # mu_bar, sigma_bar를 meta에 추가 (mc_engine이 meta에서 읽을 수 있도록)
                if mu_bar is not None and sigma_bar is not None:
                    decision_meta["mu_sim"] = float(mu_bar)
                    decision_meta["sigma_sim"] = float(sigma_bar)
                    decision["meta"] = decision_meta

            # ---- DEBUG: dump TP/SL-ish keys from decision.meta (옵션)
            if DEBUG_TPSL_META and decision:
                meta = decision.get("meta") or {}

                keys = [
                    "mc_tp", "mc_sl", "tp", "sl",
                    "profit_target", "stop_loss",
                    "tp_pct", "sl_pct",
                    "params",
                ]

                picked = {}
                for k in keys:
                    if k in meta:
                        picked[k] = meta.get(k)

                params = meta.get("params") or {}
                if isinstance(params, dict):
                    for k in ["profit_target", "stop_loss", "tp_pct", "sl_pct", "tp", "sl", "n_paths"]:
                        if k in params:
                            picked[f"params.{k}"] = params.get(k)

                self._log(f"[DBG_META_TPSL] {sym} action={decision.get('action')} picked={json.dumps(picked, ensure_ascii=False, separators=(',',':'))}")

            if decision is None:
                # decision이 None이면 기본값으로 row 추가하고 다음 심볼로
                return self._row(sym, price, ts, None, candles, ctx=ctx)
            
            ctx["ev"] = decision.get("ev", 0.0)
            side = decision.get("action_type") or decision.get("action")

            price_bucket = round(float(price), 3)
            cache_key = (sym, side, regime, price_bucket)
            now_cache = time.time()
            cached = self.mc_cache.get(cache_key)
            if cached and (now_cache - cached[0] <= self.mc_cache_ttl):
                # 메타 재사용
                decision_meta = decision.get("meta") or {}
                decision_meta.update(cached[1])
                decision["meta"] = decision_meta
            else:
                self.mc_cache[cache_key] = (now_cache, decision.get("meta", {}))

            # 러닝 통계 업데이트 (regime/session)
            def _s(val, default=0.0):
                try:
                    if val is None:
                        return float(default)
                    return float(val)
                except Exception:
                    return float(default)

            self.stats.push("ev", (regime, session), _s(decision.get("ev")))
            self.stats.push("cvar", (regime, session), _s(decision.get("cvar")))
            spread_val = _s(decision.get("meta", {}).get("spread_pct", ctx.get("spread_pct")), 0.0)
            self.stats.push("spread", (regime, session), spread_val)
            self.stats.push("liq", (regime, session), _s(self._liquidity_score(sym)))
            self.stats.push("ofi", (regime, session), _s(ofi_score))
            ofi_mean = self.stats.ema_update("ofi_mean", (regime, session), ofi_score, half_life_sec=900)
            ofi_res = ofi_score - ofi_mean
            self.stats.push("ofi_res", (regime, session), ofi_res)

            # 동적 레버리지(리스크 기반)
            dyn_leverage = float(decision.get("leverage") or decision.get("meta", {}).get("lev") or self.leverage)
            ctx["leverage"] = dyn_leverage


            # 레짐별 사이즈 상한 적용
            cap_map = {"bull": 0.25, "bear": 0.25, "chop": 0.10, "volatile": 0.08}
            cap_frac_regime = cap_map.get(regime, 0.10)
            decision = dict(decision)
            decision_meta = dict(decision.get("meta") or {})
            decision_meta["regime_cap_frac"] = cap_frac_regime
            decision["meta"] = decision_meta
            sz = decision.get("size_frac") or decision_meta.get("size_fraction") or self.default_size_frac
            decision["size_frac"] = float(min(max(0.0, sz), cap_frac_regime))

            # 추가 하드 게이트: event_cvar_r, spread_pct 상한
            spread_pct_now = decision_meta.get("spread_pct", ctx.get("spread_pct"))
            ev_cvar_r = decision_meta.get("event_cvar_r")
            # [TEST] spread_cap 비활성화 - EV 회복 테스트
            # 레짐별 스프레드 상한
            # spread_cap_map = {
            #     "bull": 0.0020,
            #     "bear": 0.0020,
            #     "chop": 0.0012,
            #     "volatile": 0.0008,
            # }
            # spread_cap = spread_cap_map.get(regime, SPREAD_PCT_MAX)
            # if spread_pct_now is not None and spread_cap is not None and spread_pct_now > spread_cap:
            #     decision["action"] = "WAIT"
            #     decision["reason"] = f"{decision.get('reason','')} | spread_cap"
            # 레짐별 event_cvar_r 하한
            cvar_floor_map = {
                "bull": -1.2,
                "bear": -1.2,
                "chop": -1.0,
                "volatile": -0.8,
            }
            cvar_floor_regime = cvar_floor_map.get(regime, -1.0)
            if ev_cvar_r is not None and ev_cvar_r < cvar_floor_regime:
                decision["action"] = "WAIT"
                decision["reason"] = f"{decision.get('reason','')} | event_cvar_floor"

            # EV 동적 진입 문턱: 최근 30분 EV의 p80 (레짐별) EMA(half-life 10m)
            ev_net_now = float(decision.get("ev", 0.0) or 0.0)
            hist_sym = self._ev_tune_hist[sym]
            hist_sym.append((ts, ev_net_now))
            cutoff = ts - int(EV_TUNE_WINDOW_SEC * 1000)
            while hist_sym and hist_sym[0][0] < cutoff:
                hist_sym.popleft()

            regime_key = (regime or "chop", session or "OFF")
            hist_reg = self._ev_regime_hist.setdefault(regime_key, deque(maxlen=4000))
            hist_reg.append((ts, ev_net_now))
            while hist_reg and hist_reg[0][0] < cutoff:
                hist_reg.popleft()

            dyn_enter_floor = None
            ev_vals = [x[1] for x in hist_reg]
            if len(ev_vals) >= EV_TUNE_MIN_SAMPLES:
                try:
                    raw_thr = float(np.percentile(ev_vals, 80))
                    # EMA half-life 10m (600s)
                    prev = self._ev_thr_ema.get(regime_key)
                    prev_ts = self._ev_thr_ema_ts.get(regime_key, ts)
                    dt_sec = max(1.0, (ts - prev_ts) / 1000.0)
                    alpha = 1.0 - math.exp(-math.log(2) * dt_sec / 600.0)
                    ema = raw_thr if prev is None else (alpha * raw_thr + (1 - alpha) * prev)
                    ema = float(max(EV_ENTER_FLOOR_MIN, min(EV_ENTER_FLOOR_MAX, ema)))
                    self._ev_thr_ema[regime_key] = ema
                    self._ev_thr_ema_ts[regime_key] = ts
                    dyn_enter_floor = ema
                except Exception:
                    dyn_enter_floor = None

            if dyn_enter_floor is not None:
                decision = dict(decision)
                meta_tmp = dict(decision.get("meta") or {})
                meta_tmp["ev_entry_threshold_dyn"] = dyn_enter_floor
                decision["meta"] = meta_tmp

            # 캐시는 "가공 이후의 decision"으로 업데이트 (UI/정책에 일관성)
            if decision:
                self._decision_cache[sym] = (ts, dict(decision))

            consensus_action, consensus_score = self._consensus_action(decision, ctx)
            if decision.get("action") == "WAIT" and consensus_action in ("LONG", "SHORT") and decision.get("ev",0.0) > 0 and decision.get("confidence",0.0) >= 0.60:
                decision = dict(decision)
                decision["action"] = consensus_action
                decision["reason"] = f"{decision.get('reason', '')} | consensus {consensus_action} score={consensus_score:.2f}"
                decision["details"] = decision.get("details", [])
                decision["details"].append({
                    "_engine": "consensus",
                    "_weight": 0.5,
                    "action": consensus_action,
                    "ev": decision.get("ev", 0.0),
                    "confidence": decision.get("confidence", 0.0),
                    "reason": f"consensus {consensus_score:.2f}",
                    "meta": {"consensus_score": consensus_score, "consensus_used": True},
                })
                decision["size_frac"] = decision.get("size_frac") or decision.get("meta", {}).get("size_fraction") or self.default_size_frac

            # ---- HOLD / EXIT ----
            exited_by_event = False
            pos = self.positions.get(sym)
            exit_unified = bool(getattr(self, "_exit_unified_enabled", False))
            run_event_mc_exits = (not exit_unified) or (not bool(getattr(self, "_exit_unified_disable_event_mc", True)))
            run_ema_exits = (not exit_unified) or (not bool(getattr(self, "_exit_unified_disable_ema_exit", True)))

            # Legacy: event-based MC exits (remaining TP/SL)
            if run_event_mc_exits and pos and decision and ctx.get("mu_base") is not None and ctx.get("sigma", 0.0) > 0:
                mu_evt, sigma_evt = adjust_mu_sigma(
                    float(ctx.get("mu_base", 0.0)),
                    float(ctx.get("sigma", 0.0)),
                    str(ctx.get("regime", "chop")),
                )
                seed_evt = int(time.time()) ^ hash(sym)
                entry = float(pos.get("entry_price", price))
                price_now = float(price)

                meta = decision.get("meta") or {}
                if not meta:
                    for d in decision.get("details", []):
                        if d.get("_engine") in ("mc_barrier", "mc_engine", "mc"):
                            meta = d.get("meta") or {}
                            break
                params_meta = meta.get("params") or {}
                tp_pct = decision.get("mc_tp") or meta.get("mc_tp") or params_meta.get("profit_target") or 0.001
                sl_pct = decision.get("mc_sl") or meta.get("mc_sl") or (tp_pct * 0.8)
                tp_pct = float(max(tp_pct, 1e-6))
                sl_pct = float(max(sl_pct, 1e-6))

                tp_rem = max((entry * (1 + tp_pct) / price_now) - 1.0, 1e-6)
                sl_rem = max(1.0 - (entry * (1 - sl_pct) / price_now), 1e-6)

                m_evt = mc_first_passage_tp_sl_jax(
                    s0=price_now,
                    tp_pct=tp_rem,
                    sl_pct=sl_rem,
                    mu=mu_evt,
                    sigma=sigma_evt,
                    dt=1.0,
                    max_steps=240,
                    n_paths=int(decision.get("n_paths", meta.get("params", {}).get("n_paths", 2048))),
                    seed=seed_evt,
                    dist=ctx.get("tail_model", "student_t"),
                    df=ctx.get("tail_df", 6.0),
                    boot_rets=ctx.get("bootstrap_returns"),
                )

                if m_evt:
                    ev_r_evt = float(m_evt.get("event_ev_r", 0.0) or 0.0)
                    cvar_r_evt = float(m_evt.get("event_cvar_r", 0.0) or 0.0)
                    p_sl_evt = float(m_evt.get("event_p_sl", 0.0) or 0.0)
                    ev_pct_evt = ev_r_evt * sl_rem
                    cvar_pct_evt = cvar_r_evt * sl_rem
                else:
                    ev_pct_evt = 0.0
                    cvar_pct_evt = 0.0
                    p_sl_evt = 0.0

                if m_evt and (
                    (ev_pct_evt < -0.0005)
                    or (cvar_pct_evt < -0.006)
                    or p_sl_evt >= 0.55
                ):
                    self._log(
                        f"[{sym}] EXIT by MC "
                        f"(EV%={ev_pct_evt*100:.2f}%, "
                        f"CVaR%={cvar_pct_evt*100:.2f}%, "
                        f"P_SL={p_sl_evt:.2f})"
                    )
                    self._close_position(sym, price_now, "event_mc_exit")
                    exited_by_event = True

            # Legacy: policy/event exit using decision.meta (mc_risk.py)
            if run_event_mc_exits and (not exited_by_event) and sym in self.positions and decision:
                pos = self.positions.get(sym) or {}
                meta = decision.get("meta") or {}
                age_sec = (ts - int(pos.get("entry_time", ts))) / 1000.0
                do_exit, reason = should_exit_position(pos, meta, age_sec=age_sec, policy=self.exit_policy)
                if do_exit:
                    self._close_position(sym, float(price), f"MC_EXIT:{reason}")
                    exited_by_event = True
                else:
                    # EMA 기반 EV/PSL 악화 감지
                    if run_ema_exits:
                        ev_now = float(decision.get("ev", 0.0) or 0.0)
                        p_sl_now = float(meta.get("event_p_sl", 0.0) or 0.0)
                        ev_ema = self._ema_update(self._ema_ev, sym, ev_now, half_life_sec=30, ts_ms=ts)
                        psl_ema = self._ema_update(self._ema_psl, sym, p_sl_now, half_life_sec=30, ts_ms=ts)
                        d_ev = ev_now - ev_ema
                        d_psl = p_sl_now - psl_ema
                        ev_floor_reg = EV_EXIT_FLOOR.get(regime, -0.0002)
                        ev_drop_reg = EV_DROP.get(regime, 0.0008)
                        psl_rise_reg = PSL_RISE.get(regime, 0.03)
                        persist_need = 1 if regime in ("bull", "bear") else 2
                        if (ev_now < ev_floor_reg) and (d_ev < -ev_drop_reg) and (d_psl > psl_rise_reg):
                            self._exit_bad_ticks[sym] = self._exit_bad_ticks.get(sym, 0) + 1
                        else:
                            self._exit_bad_ticks[sym] = 0
                        if self._exit_bad_ticks[sym] >= persist_need:
                            self._close_position(sym, float(price), "ev_psl_ema_exit", exit_kind="RISK")
                            exited_by_event = True

            self._maybe_exit_position(sym, float(price), decision, ts)

            # 열린 포지션에도 최신 동적 레버리지 반영
            if sym in self.positions:
                self.positions[sym]["leverage"] = dyn_leverage

            if not exited_by_event:
                if decision.get("action") in ("LONG", "SHORT") and sym in self.positions:
                    self._rebalance_position(sym, float(price), decision, leverage_override=dyn_leverage)

                # EV 급락 기반 추가 exit
                run_ev_drop_exit = (not exit_unified) or (not bool(getattr(self, "_exit_unified_disable_ev_drop", True)))
                if run_ev_drop_exit and sym in self.positions:
                    ev_now = float(decision.get("ev", 0.0) or 0.0)
                    meta_now = decision.get("meta") or {}
                    ev_floor = float(meta_now.get("ev_entry_threshold_dyn") or meta_now.get("ev_entry_threshold") or 0.0)
                    prev_ev = self._ev_drop_state[sym].get("prev")
                    delta_ev = None
                    if prev_ev is not None:
                        delta_ev = ev_now - prev_ev
                    # 강신호 판정
                    strong_signal = decision.get("confidence", 0.0) >= 0.65
                    needed_ticks = 1 if strong_signal else 2
                    if ev_now < ev_floor and (delta_ev is not None) and (delta_ev < -EV_DROP_THRESHOLD):
                        self._ev_drop_state[sym]["streak"] += 1
                    else:
                        self._ev_drop_state[sym]["streak"] = 0
                    self._ev_drop_state[sym]["prev"] = ev_now
                    if self._ev_drop_state[sym]["streak"] >= needed_ticks:
                        self._close_position(sym, float(price), "ev_drop_exit", exit_kind="RISK")
                        exited_by_event = True

                # ✅ 7번 비활성화: 이미 포지션이 있는 경우 체크 제거 (중복 포지션 허용)
                if decision.get("action") in ("LONG", "SHORT"):  # and sym not in self.positions 제거
                    permit, deny_reason = self._entry_permit(sym, decision, ts)
                    if permit:
                        self._enter_position(sym, decision["action"], float(price), decision, ts, ctx=ctx, leverage_override=dyn_leverage)
                    else:
                        # ✅ 상세 로그 출력 (EV가 높은데 진입이 안 되는 이유 확인)
                        meta = decision.get("meta") or {}
                        ev = decision.get("ev", 0.0)
                        win = decision.get("confidence", 0.0)
                        ev_thr = meta.get("ev_entry_threshold", 0.0)
                        ev_thr_dyn = meta.get("ev_entry_threshold_dyn")
                        win_thr = meta.get("win_entry_threshold", 0.55)
                        self._log(f"[ENTRY_BLOCK] {sym} skip entry: reason={deny_reason} | ev={ev:.6f} ev_thr={ev_thr:.6f} ev_thr_dyn={ev_thr_dyn} win={win:.3f} win_thr={win_thr:.3f} action={decision.get('action')} reason={decision.get('reason','')}")

            # ✅ DECISION 로그(너무 많으면 성능 병목이 될 수 있어 log_this_cycle에만)
            if decision:
                meta = decision.get("meta") or {}
                
                # ✅ Step B + Step 2: 비용 및 EV 검증 로그 (BTC/ETH/SOL 중 하나만 출력)
                cost_log = ""
                ev_debug_log = ""
                if sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
                    fee_roundtrip_total = meta.get("fee_roundtrip_total", meta.get("execution_cost", 0.0))
                    fee_roundtrip_base = meta.get("fee_roundtrip_base", 0.0)
                    slippage_dyn = meta.get("slippage_dyn", meta.get("slippage_pct", 0.0))
                    expected_spread_cost = meta.get("expected_spread_cost", 0.0)
                    spread_pct_meta = meta.get("spread_pct", 0.0)
                    liq_score_meta = meta.get("liq_score", 0.0)
                    sigma_sim_meta = meta.get("sigma_sim", 0.0)
                    mu_sim_meta = meta.get("mu_sim", meta.get("mu_adj", 0.0))
                    
                    # ✅ Step 2: EV 관련 값들
                    ev_raw_meta = meta.get("ev_raw", decision.get("ev", 0.0))
                    ev_for_gate_meta = meta.get("ev_for_gate", decision.get("ev", 0.0))
                    ev_hold_meta = meta.get("ev_hold")  # 있으면 출력
                    ev_policy_30m_meta = meta.get("ev_policy_30m")  # 있으면 출력
                    value_after_cost_meta = meta.get("policy_value_after_cost", meta.get("value_after_cost"))  # 있으면 출력
                    score_long_meta = meta.get("score_long")  # 있으면 출력
                    score_short_meta = meta.get("score_short")  # 있으면 출력
                    
                    # returns_window_len은 main_engine에서 계산된 값 사용 (ctx에 저장 필요)
                    rets_n_meta = ctx.get("rets_n") if ctx else None
                    vol_src_meta = ctx.get("vol_src") if ctx else None
                    
                    cost_log = (
                        f" | [COST_BREAKDOWN] fee_roundtrip_total={fee_roundtrip_total:.6f} "
                        f"fee_roundtrip_base={fee_roundtrip_base:.6f} slippage_dyn={slippage_dyn:.6f} "
                        f"expected_spread_cost={expected_spread_cost:.6f} spread_pct={spread_pct_meta:.6f} "
                        f"liq_score={liq_score_meta:.4f}"
                    )
                    
                    ev_debug_log = (
                        f" | [EV_DEBUG] mu_sim={mu_sim_meta:.8f} sigma_sim={sigma_sim_meta:.8f} "
                        f"rets_n={rets_n_meta} vol_src={vol_src_meta} "
                        f"ev_raw={ev_raw_meta:.6f} ev_for_gate={ev_for_gate_meta:.6f}"
                    )
                    if ev_hold_meta is not None:
                        ev_debug_log += f" ev_hold={ev_hold_meta:.6f}"
                    if ev_policy_30m_meta is not None:
                        ev_debug_log += f" ev_policy_30m={ev_policy_30m_meta:.6f}"
                    if value_after_cost_meta is not None:
                        ev_debug_log += f" value_after_cost={value_after_cost_meta:.6f}"
                    if score_long_meta is not None:
                        ev_debug_log += f" score_long={score_long_meta:.6f}"
                    if score_short_meta is not None:
                        ev_debug_log += f" score_short={score_short_meta:.6f}"
                
                if log_this_cycle:
                    self._log(
                        f"[DECISION] {sym} action={decision.get('action')} "
                        f"ev={decision.get('ev', 0.0):.4f} "
                        f"win={decision.get('confidence', 0.0):.2f} "
                        f"size={decision.get('size_frac', meta.get('size_fraction', 0.0)):.3f} "
                        f"reason={decision.get('reason', '')}{cost_log}{ev_debug_log}"
                    )

            # ✅ 최신 가격으로 포지션 값 계산: broadcast 직전에 최신 가격 가져오기
            latest_price = self.data.market[sym].get("price", price)
            row_data = self._row(sym, float(latest_price) if latest_price is not None else None, ts, decision, candles, ctx=ctx)
            # ✅ 디버그: _row()가 decision을 제대로 읽는지 확인(옵션)
            if DEBUG_ROW and sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
                self._log(f"[ROW_DEBUG] {sym} | decision_ev={decision.get('ev', 0.0) if decision else None} row_ev={row_data.get('ev', 0.0)} | decision_conf={decision.get('confidence', 0.0) if decision else None} row_conf={row_data.get('conf', 0.0)}")
            return row_data
        except Exception as e:
            import traceback
            # ✅ 각 심볼별로 개별 에러 처리하여 하나 실패해도 나머지 계속 처리
            self._log_err(f"[ERR] analyze_symbol {sym}: {e} {traceback.format_exc()}")
            # 에러가 발생해도 기본 row는 반환
            return self._row(sym, self.data.market.get(sym, {}).get("price"), ts, None, len(list(self.data.ohlcv_buffer.get(sym, []))), ctx=locals().get('ctx'))

    async def _compute_decision_task(self, sym: str, ctx: dict, ts_ms: int):
        try:
            # Debug: log ctx keys and pmaker_entry value before calling hub.decide
            ctx_id_at_start = id(ctx)
            pmaker_entry_at_start = ctx.get('pmaker_entry')
            pmaker_entry_delay_sec_at_start = ctx.get('pmaker_entry_delay_sec')
            print(f"[PMAKER_DEBUG] {sym} | _compute_decision_task: START ctx id={ctx_id_at_start} ctx keys: {list(ctx.keys())} pmaker_entry={pmaker_entry_at_start} (type={type(pmaker_entry_at_start)}) pmaker_entry_delay_sec={pmaker_entry_delay_sec_at_start} (type={type(pmaker_entry_delay_sec_at_start)})")
            self._log(f"[PMAKER_DEBUG] {sym} | _compute_decision_task: START ctx keys: {list(ctx.keys())} pmaker_entry={pmaker_entry_at_start} pmaker_entry_delay_sec={pmaker_entry_delay_sec_at_start}")
            async with self._decision_sem:
                # Check ctx again after acquiring semaphore
                ctx_id_after_sem = id(ctx)
                pmaker_entry_after_sem = ctx.get('pmaker_entry')
                pmaker_entry_delay_sec_after_sem = ctx.get('pmaker_entry_delay_sec')
                print(f"[PMAKER_DEBUG] {sym} | _compute_decision_task: AFTER semaphore ctx id={ctx_id_after_sem} (same={ctx_id_at_start == ctx_id_after_sem}) pmaker_entry={pmaker_entry_after_sem} (changed={pmaker_entry_at_start != pmaker_entry_after_sem}) pmaker_entry_delay_sec={pmaker_entry_delay_sec_after_sem} (changed={pmaker_entry_delay_sec_at_start != pmaker_entry_delay_sec_after_sem})")
                # Make a deep copy of ctx to ensure values are preserved across thread boundary
                # Use copy.deepcopy to handle nested dicts (like ctx["meta"])
                import copy
                # Debug: check ctx pmaker_entry value and type before deepcopy
                pmaker_entry_orig = ctx.get('pmaker_entry')
                pmaker_entry_delay_sec_orig = ctx.get('pmaker_entry_delay_sec')
                print(f"[PMAKER_DEBUG] {sym} | _compute_decision_task: BEFORE deepcopy - pmaker_entry={pmaker_entry_orig} (type={type(pmaker_entry_orig)}) pmaker_entry_delay_sec={pmaker_entry_delay_sec_orig} (type={type(pmaker_entry_delay_sec_orig)})")
                ctx_copy = copy.deepcopy(ctx)
                pmaker_entry_copied = ctx_copy.get('pmaker_entry')
                pmaker_entry_delay_sec_copied = ctx_copy.get('pmaker_entry_delay_sec')
                print(f"[PMAKER_DEBUG] {sym} | _compute_decision_task: AFTER deepcopy - pmaker_entry={pmaker_entry_copied} (type={type(pmaker_entry_copied)}) pmaker_entry_delay_sec={pmaker_entry_delay_sec_copied} (type={type(pmaker_entry_delay_sec_copied)})")
                print(f"[PMAKER_DEBUG] {sym} | _compute_decision_task: ctx_copy keys: {list(ctx_copy.keys())} pmaker_entry={ctx_copy.get('pmaker_entry')} pmaker_entry_delay_sec={ctx_copy.get('pmaker_entry_delay_sec')}")
                self._log(f"[PMAKER_DEBUG] {sym} | _compute_decision_task: ctx_copy keys: {list(ctx_copy.keys())} pmaker_entry={ctx_copy.get('pmaker_entry')} pmaker_entry_delay_sec={ctx_copy.get('pmaker_entry_delay_sec')}")
                d = await asyncio.to_thread(self.hub.decide, ctx_copy)
                # [EV_DEBUG] hub.decide 반환값에서 EV 확인
                if d:
                    ev_from_decision = d.get("ev", 0.0)
                    action_from_decision = d.get("action", "WAIT")
                    print(f"[EV_DEBUG] {sym} | _compute_decision_task: hub.decide returned type={type(d)} d.keys={list(d.keys())[:20]}")
                    print(f"[EV_DEBUG] {sym} | _compute_decision_task: hub.decide returned ev={ev_from_decision} (type={type(ev_from_decision)}) action={action_from_decision} details count={len(d.get('details', []))}")
                    self._log(f"[EV_DEBUG] {sym} | _compute_decision_task: hub.decide returned ev={ev_from_decision} action={action_from_decision}")
                    # ✅ EV가 None이거나 0인 경우 상세 확인
                    if ev_from_decision is None or ev_from_decision == 0.0:
                        meta_ev = d.get('meta', {}).get('ev') if isinstance(d.get('meta'), dict) else None
                        print(f"[EV_DEBUG] {sym} | ⚠️  EV is None or 0! d.get('ev')={d.get('ev')} meta.ev={meta_ev}")
                        if d.get('details'):
                            for idx, detail in enumerate(d.get('details', [])):
                                detail_ev = detail.get('ev')
                                detail_meta = detail.get('meta', {})
                                detail_meta_ev = detail_meta.get('ev') if isinstance(detail_meta, dict) else None
                                print(f"[EV_DEBUG] {sym} | detail[{idx}] ev={detail_ev} meta.ev={detail_meta_ev}")
                else:
                    print(f"[EV_DEBUG] {sym} | ⚠️  hub.decide returned None or empty dict!")
                    # Debug: log hub.decide return value
                    print(f"[PMAKER_DEBUG] {sym} | _compute_decision_task: hub.decide returned, details count={len(d.get('details', []))}")
                    for idx, detail in enumerate(d.get('details', [])):
                        engine_name = detail.get('_engine')
                        detail_meta = detail.get('meta', {})
                        detail_ev = detail.get('ev')
                        print(f"[PMAKER_DEBUG] {sym} | _compute_decision_task: detail[{idx}] engine={engine_name} ev={detail_ev} meta keys={list(detail_meta.keys())[:30] if isinstance(detail_meta, dict) else []} pmaker_entry={detail_meta.get('pmaker_entry') if isinstance(detail_meta, dict) else None}")
                    decision_meta = d.get('meta', {})
                    print(f"[PMAKER_DEBUG] {sym} | _compute_decision_task: decision.meta keys={list(decision_meta.keys())[:30] if isinstance(decision_meta, dict) else []} pmaker_entry={decision_meta.get('pmaker_entry') if isinstance(decision_meta, dict) else None}")
            if d:
                d_ev = d.get('ev')
                print(f"[EV_DEBUG] {sym} | _compute_decision_task: Storing in _decision_cache: ev={d_ev}")
                self._decision_cache[sym] = (ts_ms, dict(d))
                self._mc_ready = True
            else:
                print(f"[EV_DEBUG] {sym} | _compute_decision_task: hub.decide returned None or empty, not caching")
        except Exception as e:
            self._log_err(f"[ERR] decision_task {sym}: {e}")
        finally:
            # task cleanup
            try:
                cur = self._decision_tasks.get(sym)
                if cur is not None and cur.done():
                    self._decision_tasks.pop(sym, None)
            except Exception:
                pass

    def _log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": "INFO", "msg": text})
        if LOG_STDOUT:
            print(text)

    def _log_err(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": "ERROR", "msg": text})
        # 에러는 기본적으로 stdout에 출력(운영 가시성)
        print(text)

    def _pmaker_status(self) -> dict:
        return self.pmaker.status_dict()


    async def _ccxt_call(self, label: str, fn, *args, **kwargs):
        """
        Best-effort CCXT call with:
          - concurrency cap (semaphore)
          - retry with exponential backoff + jitter
          - ExchangeNotAvailable 에러에 대한 상세 로그
        """
        for attempt in range(1, MAX_RETRY + 1):
            try:
                async with self._net_sem:
                    return await fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                exc_type = type(e).__name__
                is_retryable = any(k in msg for k in [
                    "RequestTimeout", "DDoSProtection", "ExchangeNotAvailable",
                    "NetworkError", "ETIMEDOUT", "ECONNRESET", "502", "503", "504"
                ]) or "ExchangeNotAvailable" in exc_type
                
                # ✅ ExchangeNotAvailable 에러에 대한 상세 로그
                if "ExchangeNotAvailable" in msg or "ExchangeNotAvailable" in exc_type:
                    self._log_err(f"[CCXT_DEBUG] {label} | ExchangeNotAvailable detected - attempt={attempt}/{MAX_RETRY} exc_type={exc_type} msg={msg[:200]}")
                    # exchange 객체 상태 확인
                    if hasattr(self, 'exchange') and self.exchange:
                        self._log_err(f"[CCXT_DEBUG] {label} | exchange type={type(self.exchange)} has_fetch_orderbook={hasattr(self.exchange, 'fetch_order_book')}")
                
                if (attempt >= MAX_RETRY) or (not is_retryable):
                    raise
                backoff = (RETRY_BASE_SEC * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                self._log_err(f"[WARN] {label} retry {attempt}/{MAX_RETRY} err={msg} sleep={backoff:.2f}s")
                await asyncio.sleep(backoff)

    # -----------------------------
    # Persistence helpers
    # -----------------------------
    def _load_json(self, path: Path, default):
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self._log_err(f"[ERR] load {path.name}: {e}")
        return default

    def _load_persistent_state(self):
        balance_loaded = False
        # balance
        bal = self._load_json(self.state_files.get("balance"), None)
        if isinstance(bal, (int, float)):
            self.balance = float(bal)
            balance_loaded = True

        # equity history
        eq = self._load_json(self.state_files["equity"], [])
        for item in eq:
            try:
                t = int(item.get("time", 0))
                v = float(item.get("equity", 0.0))
                self._equity_history.append({"time": t, "equity": v})
            except Exception:
                continue

        # trade tape
        trades = self._load_json(self.state_files["trade"], [])
        for t in trades:
            self.trade_tape.append(t)

        # eval history
        evals = self._load_json(self.state_files["eval"], [])
        for e in evals:
            self.eval_history.append(e)

        # positions
        poss = self._load_json(self.state_files.get("positions"), [])
        if isinstance(poss, list):
            for p in poss:
                try:
                    sym = str(p.get("symbol"))
                    if not sym:
                        continue
                    p["entry_price"] = float(p.get("entry_price", 0.0))
                    p["quantity"] = float(p.get("quantity", 0.0))
                    p["notional"] = float(p.get("notional", 0.0))
                    p["leverage"] = float(p.get("leverage", self.leverage))
                    p["cap_frac"] = float(p.get("cap_frac", 0.0))
                    p["entry_time"] = int(p.get("entry_time", now_ms()))
                    p["hold_limit"] = int(p.get("hold_limit", MAX_POSITION_HOLD_SEC * 1000))
                    self.positions[sym] = p
                except Exception:
                    continue

        # fallback balance from equity history if not loaded and no positions
        if (not balance_loaded) and self._equity_history and not self.positions:
            self.balance = float(self._equity_history[-1]["equity"])

    def _persist_state(self, force: bool = False):
        now = now_ms()
        if not force and (now - self._last_state_persist_ms < 10_000):
            return
        self._last_state_persist_ms = now
        try:
            with self.state_files["equity"].open("w", encoding="utf-8") as f:
                json.dump(list(self._equity_history), f, ensure_ascii=False)
            with self.state_files["trade"].open("w", encoding="utf-8") as f:
                json.dump(list(self.trade_tape), f, ensure_ascii=False)
            with self.state_files["eval"].open("w", encoding="utf-8") as f:
                json.dump(list(self.eval_history), f, ensure_ascii=False)
            with self.state_files["positions"].open("w", encoding="utf-8") as f:
                json.dump(list(self.positions.values()), f, ensure_ascii=False)
            with self.state_files["balance"].open("w", encoding="utf-8") as f:
                json.dump(self.balance, f, ensure_ascii=False)
        except Exception as e:
            self._log_err(f"[ERR] persist state: {e}")

    def _compute_returns_and_vol(self, prices):
        if prices is None or len(prices) < 10:
            return None, None

        log_returns = []
        for i in range(1, len(prices)):
            p0, p1 = prices[i - 1], prices[i]
            if p0 and p1 and p0 > 0 and p1 > 0:
                log_returns.append(math.log(p1 / p0))

        if len(log_returns) < 5:
            return None, None

        mu = sum(log_returns) / len(log_returns)
        var = sum((r - mu) ** 2 for r in log_returns) / len(log_returns)
        sigma = math.sqrt(var)
        return mu, sigma

    @staticmethod
    def _annualize_mu_sigma(mu_bar: float, sigma_bar: float, bar_seconds: float) -> tuple[float, float]:
        bars_per_year = (365.0 * 24.0 * 3600.0) / float(bar_seconds)
        mu_base = mu_bar * bars_per_year
        sigma_annual = sigma_bar * math.sqrt(bars_per_year)
        return float(mu_base), float(sigma_annual)

    @staticmethod
    def _safe_float(x, default=0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    @staticmethod
    def _sanitize_for_json(obj, _depth: int = 0):
        """
        payload 전체를 JSON-친화적인 값으로 정규화한다.
        - NaN / Inf -> None (null)
        - numpy scalar/array -> Python 기본형 + 리스트
        - 알 수 없는 타입 -> str(...) 또는 None
        """
        if _depth > 6:
            return None

        # 기본 스칼라
        if obj is None or isinstance(obj, (str, bool, int)):
            return obj

        # float 및 numpy 스칼라
        if isinstance(obj, float):
            if math.isfinite(obj):
                return obj
            return None
        if isinstance(obj, (np.floating, np.integer)):
            v = float(obj)
            if math.isfinite(v):
                return v
            return None

        # 시퀀스
        if isinstance(obj, (list, tuple)):
            return [LiveOrchestrator._sanitize_for_json(x, _depth + 1) for x in obj]

        # dict
        if isinstance(obj, dict):
            return {str(k): LiveOrchestrator._sanitize_for_json(v, _depth + 1) for k, v in obj.items()}

        # numpy 배열
        if isinstance(obj, np.ndarray):
            try:
                return LiveOrchestrator._sanitize_for_json(obj.tolist(), _depth + 1)
            except Exception:
                return None

        # 그 외 숫자 비슷한 것
        try:
            v = float(obj)
            if math.isfinite(v):
                return v
            return None
        except Exception:
            pass

        # 마지막 수단: 문자열
        try:
            return str(obj)
        except Exception:
            return None

    def _row(self, sym, price, ts, decision, candles, ctx=None):
        status = "WAIT"
        ai = "-"
        mc = "-"
        conf = 0.0

        mc_meta = {}
        # ✅ EV_DEBUG: decision 타입 및 값 확인
        decision_type = type(decision).__name__
        decision_repr = str(decision)[:100] if decision is not None else "None"
        print(f"[EV_DEBUG_ROW] {sym} | _row START: decision type={decision_type} decision={decision_repr}")
        
        # Debug: log decision state at start of _row
        print(f"[PMAKER_DEBUG] {sym} | _row: decision={decision is not None} details={len(decision.get('details', [])) if decision and isinstance(decision, dict) else 0} decision.meta={list(decision.get('meta', {}).keys())[:10] if decision and isinstance(decision, dict) and isinstance(decision.get('meta'), dict) else []}")
        
        # ✅ decision이 dict인지 확인 (boolean False가 아닌지)
        if decision is None:
            cached = self._decision_cache.get(sym)
            if cached:
                decision = cached[1]
                print(f"[EV_DEBUG_ROW] {sym} | Falling back to cached decision for display")

        # [PMAKER] mc_meta extracted for display from most relevant engine detail
        mc_meta = {}
        if decision and isinstance(decision, dict):
            status = decision.get("action", "WAIT")  # LONG/SHORT/WAIT
            conf = self._safe_float(decision.get("confidence", 0.0), 0.0)
            mc = decision.get("reason", "") or "-"
            # details에서 mc 메타 뽑기
            details = decision.get("details", [])
            print(f"[PMAKER_DEBUG] {sym} | _row: details count={len(details)}")
            for d in details:
                engine_name = d.get("_engine")
                print(f"[PMAKER_DEBUG] {sym} | _row: checking detail engine={engine_name}")
                if engine_name in ("mc_barrier", "mc_engine", "mc"):
                    mc_meta = d.get("meta", {}) or {}
                    # Debug: log what we found in details
                    print(f"[PMAKER_DEBUG] {sym} | _row: found mc_engine in details, meta keys={list(mc_meta.keys()) if isinstance(mc_meta, dict) else []} pmaker_entry={mc_meta.get('pmaker_entry') if isinstance(mc_meta, dict) else None}")
                    break
            # details에 없으면 decision.meta 사용 (엔진이 meta를 직접 주는 경우)
            if not mc_meta:
                mc_meta = decision.get("meta", {}) or {}
                print(f"[PMAKER_DEBUG] {sym} | _row: using decision.meta, meta keys={list(mc_meta.keys()) if isinstance(mc_meta, dict) else []} pmaker_entry={mc_meta.get('pmaker_entry') if isinstance(mc_meta, dict) else None}")
        else:
            print(f"[PMAKER_DEBUG] {sym} | _row: decision is None (even after cache fallback), mc_meta will be empty")
            mc_meta = {}

        trade_age = ts - self.data.market[sym]["ts"] if self.data.market[sym]["ts"] else None
        kline_age = ts - self.data._last_kline_ok_ms[sym] if self.data._last_kline_ok_ms[sym] else None

        ob_age = ts - self.data.orderbook[sym]["ts"] if self.data.orderbook[sym]["ts"] else None
        ob_ready = bool(self.data.orderbook[sym]["ready"])
        best_bid = None
        best_ask = None
        spread_pct_book = None
        if ob_ready:
            try:
                ob0 = self.data.orderbook.get(sym) or {}
                bids0 = ob0.get("bids") or []
                asks0 = ob0.get("asks") or []
                if bids0 and asks0 and len(bids0[0]) >= 2 and len(asks0[0]) >= 2:
                    best_bid = float(bids0[0][0])
                    best_ask = float(asks0[0][0])
                    mid0 = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0.0
                    if mid0 > 0:
                        spread_pct_book = float((best_ask - best_bid) / mid0)
            except Exception:
                best_bid = None
                best_ask = None
                spread_pct_book = None

        # ✅ decision이 dict인지 확인 (boolean False가 아닌지)
        if decision is not None and not isinstance(decision, dict):
            print(f"[EV_DEBUG_ROW] {sym} | ⚠️  WARNING: decision is not a dict! type={type(decision)} value={decision}")
            decision = None  # dict가 아니면 None으로 처리
        
        meta = decision.get("meta", {}) if decision and isinstance(decision, dict) else {}
        ev_from_decision = decision.get('ev') if decision and isinstance(decision, dict) else None
        ev_from_meta = meta.get('ev') if isinstance(meta, dict) else None
        ev_raw_from_decision = decision.get("ev", meta.get("ev", 0.0)) if decision and isinstance(decision, dict) else 0.0
        
        if abs(ev_raw_from_decision) < 1e-6 and mc_meta:
            policy_ev_long_val = mc_meta.get("policy_ev_mix_long", 0.0) or 0.0
            policy_ev_short_val = mc_meta.get("policy_ev_mix_short", 0.0) or 0.0
            # Use the one with larger absolute value (better direction)
            if abs(policy_ev_long_val) > abs(policy_ev_short_val):
                ev_raw = policy_ev_long_val
            else:
                ev_raw = policy_ev_short_val
            print(f"[EV_DEBUG_ROW] {sym} | Using policy_ev from mc_meta: long={policy_ev_long_val:.6f} short={policy_ev_short_val:.6f} selected={ev_raw:.6f}")
        else:
            ev_raw = ev_raw_from_decision

        # ✅ If EV is still 0, try to use ctx as last resort (raw mu_base)
        if abs(ev_raw) < 1e-9 and ctx:
            mu_base_ctx = ctx.get("mu_base", 0.0) or 0.0
            if abs(mu_base_ctx) > 1e-9:
                ev_raw = mu_base_ctx
                print(f"[EV_DEBUG_ROW] {sym} | Using mu_base from ctx as fallback EV: {ev_raw:.6f}")
        
        ev = self._safe_float(ev_raw, 0.0)
        # ✅ EV 값 디버깅: 실제 값 확인
        print(f"[EV_DEBUG_ROW] {sym} | decision type={type(decision).__name__} decision={decision is not None} decision.keys={list(decision.keys())[:10] if decision and isinstance(decision, dict) else 'N/A'} ev_from_decision={ev_from_decision} ev_from_meta={ev_from_meta} ev_raw={ev_raw} ev={ev}")
        
        # ✅ 모든 심볼의 EV 값 검증 (LINK 제외, 음수인 경우)
        if not sym.startswith("LINK") and ev < 0 and abs(ev) > 1e-6:
            self._log(f"[EV_VALIDATION_NEG_ROW] {sym} | ⚠️  _row에서 음수 EV 발견: ev={ev:.6f} ev_from_decision={ev_from_decision} ev_from_meta={ev_from_meta}")
            print(f"[EV_VALIDATION_NEG_ROW] {sym} | ⚠️  _row에서 음수 EV 발견: ev={ev:.6f} ev_from_decision={ev_from_decision} ev_from_meta={ev_from_meta}")
        kelly = self._safe_float(meta.get("kelly", 0.0), 0.0)
        regime = (ctx or {}).get("regime") or meta.get("regime") or "-"
        mode = "alpha"
        if self.positions.get(sym, {}).get("tag") == "spread":
            mode = "spread"
        elif decision and any(d.get("_engine") == "consensus" for d in decision.get("details", [])):
            mode = "consensus"
        action_type = self.positions.get(sym, {}).get("tag") == "spread" and "SPREAD" or (self.positions.get(sym) and "HOLD") or self._last_actions.get(sym, "-")

        pos = self.positions.get(sym)
        pos_side = pos.get("side") if pos else "-"
        pos_pnl = None
        pos_roe = None
        pos_lev = pos.get("leverage") if pos else self._dyn_leverage.get(sym)
        pos_sf = pos.get("size_frac") if pos else None
        pos_cap_frac = pos.get("cap_frac") if pos else None
        # ✅ 최신 가격 사용: price 파라미터가 None이면 market에서 최신 가격 가져오기
        current_price = price if price is not None else self.data.market[sym].get("price")
        if pos and current_price:
            entry = float(pos.get("entry_price", current_price))
            qty = float(pos.get("quantity", 0.0))
            pnl = ((current_price - entry) * qty) if pos_side == "LONG" else ((entry - current_price) * qty)
            notional = float(pos.get("notional", 0.0))
            lev_safe = float(pos.get("leverage", self.leverage) or 1.0)
            base_notional = notional / max(lev_safe, 1e-6)
            pos_pnl = pnl
            pos_roe = pnl / base_notional if base_notional else 0.0

        def _opt_float(val):
            if val is None:
                return None
            try:
                return float(val)
            except Exception:
                return None

        event_p_tp = _opt_float(meta.get("event_p_tp"))
        event_p_timeout = _opt_float(meta.get("event_p_timeout"))
        event_t_median = _opt_float(meta.get("event_t_median"))
        event_ev_r = _opt_float(meta.get("event_ev_r"))
        event_cvar_r = _opt_float(meta.get("event_cvar_r"))
        event_ev_pct = _opt_float(meta.get("event_ev_pct"))
        event_cvar_pct = _opt_float(meta.get("event_cvar_pct"))
        horizon_weights = meta.get("horizon_weights")
        ev_by_h = meta.get("ev_by_horizon")
        win_by_h = meta.get("win_by_horizon")
        cvar_by_h = meta.get("cvar_by_horizon")
        horizon_seq = meta.get("horizon_seq")

        # policy-rollforward debug (from MC engine meta)
        def _opt_int(val):
            if val is None:
                return None
            try:
                return int(val)
            except Exception:
                return None

        policy_ev_long = _opt_float(mc_meta.get("policy_ev_mix_long"))
        policy_ev_short = _opt_float(mc_meta.get("policy_ev_mix_short"))
        policy_p_pos_long = _opt_float(mc_meta.get("policy_p_pos_mix_long"))
        policy_p_pos_short = _opt_float(mc_meta.get("policy_p_pos_mix_short"))
        hold_best_ev_long = _opt_float(mc_meta.get("hold_best_ev_long"))
        hold_best_ev_short = _opt_float(mc_meta.get("hold_best_ev_short"))
        policy_exit_unrealized_dd_frac = _opt_float(mc_meta.get("policy_exit_unrealized_dd_frac"))
        policy_exit_hold_bad_frac = _opt_float(mc_meta.get("policy_exit_hold_bad_frac"))
        policy_exit_score_flip_frac = _opt_float(mc_meta.get("policy_exit_score_flip_frac"))
        policy_ev_gap = None
        policy_p_pos_gap = None
        try:
            if policy_ev_long is not None and policy_ev_short is not None:
                policy_ev_gap = float(policy_ev_long) - float(policy_ev_short)
        except Exception:
            policy_ev_gap = None
        try:
            if policy_p_pos_long is not None and policy_p_pos_short is not None:
                policy_p_pos_gap = float(policy_p_pos_long) - float(policy_p_pos_short)
        except Exception:
            policy_p_pos_gap = None

        policy_signal_strength = _opt_float(mc_meta.get("policy_signal_strength"))
        policy_weight_peak_h = _opt_int(mc_meta.get("policy_weight_peak_h"))
        policy_half_life_sec = _opt_float(mc_meta.get("policy_half_life_sec"))
        policy_h_eff_sec = _opt_float(mc_meta.get("policy_h_eff_sec"))
        policy_h_eff_sec_prior = _opt_float(mc_meta.get("policy_h_eff_sec_prior"))  # [DIFF 3] Rule-based only
        policy_w_short_sum = _opt_float(mc_meta.get("policy_w_short_sum"))
        policy_exit_time_mean_sec = _opt_float(mc_meta.get("exit_time_mean_sec"))
        policy_horizons = mc_meta.get("policy_horizons")
        policy_w_h = mc_meta.get("policy_w_h")
        # ✅ 검증 포인트: paths_reused 메타 확인
        paths_reused = bool(mc_meta.get("paths_reused")) if isinstance(mc_meta, dict) else False
        # exit reason counts per horizon (long/short 분리)
        policy_exit_reason_counts_per_h = mc_meta.get("policy_exit_reason_counts_per_h")
        policy_exit_reason_counts_per_h_long = mc_meta.get("policy_exit_reason_counts_per_h_long")
        policy_exit_reason_counts_per_h_short = mc_meta.get("policy_exit_reason_counts_per_h_short")

        # EV decomposition (quick sanity)
        ev_decomp_mu_annual = _opt_float(mc_meta.get("ev_decomp_mu_annual"))
        ev_decomp_mu_per_sec = _opt_float(mc_meta.get("ev_decomp_mu_per_sec"))
        ev_decomp_fee_rt = _opt_float(mc_meta.get("ev_decomp_fee_roundtrip_total"))
        ev_decomp_gross_long_600 = _opt_float(mc_meta.get("ev_decomp_gross_long_600"))
        ev_decomp_gross_long_1800 = _opt_float(mc_meta.get("ev_decomp_gross_long_1800"))
        ev_decomp_gross_short_600 = _opt_float(mc_meta.get("ev_decomp_gross_short_600"))
        ev_decomp_gross_short_1800 = _opt_float(mc_meta.get("ev_decomp_gross_short_1800"))
        ev_decomp_net_long_600 = _opt_float(mc_meta.get("ev_decomp_net_long_600"))
        ev_decomp_net_long_1800 = _opt_float(mc_meta.get("ev_decomp_net_long_1800"))
        ev_decomp_net_short_600 = _opt_float(mc_meta.get("ev_decomp_net_short_600"))
        ev_decomp_net_short_1800 = _opt_float(mc_meta.get("ev_decomp_net_short_1800"))
        ev_decomp_mu_req_annual_600 = _opt_float(mc_meta.get("ev_decomp_mu_req_annual_600"))
        ev_decomp_mu_req_annual_1800 = _opt_float(mc_meta.get("ev_decomp_mu_req_annual_1800"))
        ev_decomp_mu_req_annual_exit_mean = _opt_float(mc_meta.get("ev_decomp_mu_req_annual_exit_mean"))
        # μ(alpha) debug (why mu is negative / too large)
        mu_alpha = _opt_float(mc_meta.get("mu_alpha"))
        mu_alpha_raw = _opt_float(mc_meta.get("mu_alpha_raw"))
        mu_alpha_mom = _opt_float(mc_meta.get("mu_alpha_mom"))
        mu_alpha_ofi = _opt_float(mc_meta.get("mu_alpha_ofi"))
        mu_alpha_regime_scale = _opt_float(mc_meta.get("mu_alpha_regime_scale"))
        mu_alpha_mom_15 = _opt_float(mc_meta.get("mu_alpha_mom_15"))
        mu_alpha_mom_30 = _opt_float(mc_meta.get("mu_alpha_mom_30"))
        mu_alpha_mom_60 = _opt_float(mc_meta.get("mu_alpha_mom_60"))
        mu_alpha_mom_120 = _opt_float(mc_meta.get("mu_alpha_mom_120"))
        # ✅ [FIX 2] PMaker mu_alpha boost 정보
        mu_alpha_pmaker_fill_rate = _opt_float(mc_meta.get("mu_alpha_pmaker_fill_rate"))
        mu_alpha_pmaker_boost = _opt_float(mc_meta.get("mu_alpha_pmaker_boost"))
        mu_alpha_before_pmaker = _opt_float(mc_meta.get("mu_alpha_before_pmaker"))
        exec_mode = mc_meta.get("exec_mode") or meta.get("exec_mode") or os.environ.get("EXEC_MODE", EXEC_MODE)
        p_maker = _opt_float(mc_meta.get("p_maker"))
        fee_roundtrip_fee_mix = _opt_float(mc_meta.get("fee_roundtrip_fee_mix"))
        fee_roundtrip_fee_taker = _opt_float(mc_meta.get("fee_roundtrip_fee_taker"))  # [DIFF 5] For validation
        fee_roundtrip_fee_maker = _opt_float(mc_meta.get("fee_roundtrip_fee_maker"))  # [DIFF 5] For validation
        fee_roundtrip_total = _opt_float(mc_meta.get("fee_roundtrip_total"))
        pmaker_entry = _opt_float(mc_meta.get("pmaker_entry"))
        pmaker_entry_delay_sec = _opt_float(mc_meta.get("pmaker_entry_delay_sec"))
        pmaker_exit = _opt_float(mc_meta.get("pmaker_exit"))
        pmaker_exit_delay_sec = _opt_float(mc_meta.get("pmaker_exit_delay_sec"))
        pmaker_entry_delay_penalty_r = _opt_float(mc_meta.get("pmaker_entry_delay_penalty_r"))
        pmaker_exit_delay_penalty_r = _opt_float(mc_meta.get("pmaker_exit_delay_penalty_r"))
        policy_entry_shift_steps = _opt_int(mc_meta.get("policy_entry_shift_steps"))
        policy_horizon_eff_sec = _opt_int(mc_meta.get("policy_horizon_eff_sec"))
        pmaker_override_used = bool(mc_meta.get("pmaker_override_used")) if isinstance(mc_meta, dict) else False
        policy_direction = _opt_int(mc_meta.get("policy_direction"))
        
        # Debug logging (always print to stdout for visibility)
        print(f"[PMAKER_DEBUG] {sym} | _row: mc_meta keys={list(mc_meta.keys()) if isinstance(mc_meta, dict) else []}")
        print(f"[PMAKER_DEBUG] {sym} | _row: pmaker_entry={pmaker_entry} pmaker_exit={pmaker_exit}")
        print(f"[PMAKER_DEBUG] {sym} | _row: entry_penalty={pmaker_entry_delay_penalty_r} exit_penalty={pmaker_exit_delay_penalty_r}")
        print(f"[PMAKER_DEBUG] {sym} | _row: shift_steps={policy_entry_shift_steps} horizon_eff={policy_horizon_eff_sec}")
        if os.getenv("PMAKER_DEBUG", "0") == "1":
            self._log(f"[PMAKER_DEBUG] {sym} | _row: mc_meta keys={list(mc_meta.keys()) if isinstance(mc_meta, dict) else []}")
            self._log(f"[PMAKER_DEBUG] {sym} | _row: pmaker_entry={pmaker_entry} pmaker_exit={pmaker_exit}")
            self._log(f"[PMAKER_DEBUG] {sym} | _row: entry_penalty={pmaker_entry_delay_penalty_r} exit_penalty={pmaker_exit_delay_penalty_r}")
            self._log(f"[PMAKER_DEBUG] {sym} | _row: shift_steps={policy_entry_shift_steps} horizon_eff={policy_horizon_eff_sec}")
        spread_cap = _opt_float(meta.get("spread_cap"))
        spread_entry_max = _opt_float(meta.get("spread_entry_max"))

        # execution KPI snapshot (real fills only when ENABLE_LIVE_ORDERS=1)
        st = self._exec_stats.get(sym) or {}
        maker_path = float(st.get("order_requests_maker_path", 0) or 0)
        maker_filled = float(st.get("order_requests_maker_filled", 0) or 0)
        fallback_market = float(st.get("order_requests_fallback_market", 0) or 0)
        maker_path_fill_rate = (maker_filled / maker_path) if maker_path > 0 else None
        fallback_rate = (fallback_market / maker_path) if maker_path > 0 else None
        maker_orders = float(st.get("maker_limit_orders", 0) or 0)
        cancel_ok = float(st.get("maker_limit_cancel_ok", 0) or 0)
        cancel_rate = (cancel_ok / maker_orders) if maker_orders > 0 else None
        lat_m = list(st.get("lat_ms_maker_filled", []) or [])
        fill_delay_p95 = float(np.quantile(np.asarray(lat_m, dtype=np.float64), 0.95)) if lat_m else None

        return {
            "symbol": sym,
            "price": price,
            "status": status,
            "ai": ai,
            "mc": mc,
            "conf": conf,
            "win": conf,  # ✅ conf와 동일한 값 (대시보드 호환성)
            "ev": ev,
            "ev_raw": self._safe_float(
                mc_meta.get("policy_ev_mix_long") if mc_meta.get("policy_ev_mix_long", 0) > mc_meta.get("policy_ev_mix_short", 0) 
                else mc_meta.get("policy_ev_mix_short") if mc_meta 
                else decision.get("ev_raw", 0.0) if decision and isinstance(decision, dict) 
                else 0.0, 
                0.0
            ),
            "kelly": kelly,
            "regime": regime,
            "mode": mode,
            "action_type": action_type,
            "candles": candles,
            "event_p_tp": event_p_tp,
            "event_p_timeout": event_p_timeout,
            "event_t_median": event_t_median,
            "event_ev_r": event_ev_r,
            "event_cvar_r": event_cvar_r,
            "event_ev_pct": event_ev_pct,
            "event_cvar_pct": event_cvar_pct,
            "horizon_weights": horizon_weights,
            "ev_by_horizon": ev_by_h,
            "win_by_horizon": win_by_h,
            "cvar_by_horizon": cvar_by_h,
            "horizon_seq": horizon_seq,

            # freshness
            "trade_age": trade_age,
            "kline_age": kline_age,
            "orderbook_age": ob_age,
            "orderbook_ready": ob_ready,

            "pos_side": pos_side,
            "pos_pnl": pos_pnl,
            "pos_roe": pos_roe,
            "pos_tag": pos.get("tag") if pos else None,
            "pos_leverage": pos_lev,
            "pos_size_frac": pos_sf,
            "pos_cap_frac": pos_cap_frac,

            # MC diagnostics
            "mc_h_desc": mc_meta.get("best_horizon_desc") or mc_meta.get("best_horizon") or "-",
            "mc_ev": self._safe_float(mc_meta.get("ev", 0.0), 0.0),
            "mc_win_rate": self._safe_float(mc_meta.get("win_rate", 0.0), 0.0),
            "mc_be_win_rate": self._safe_float(mc_meta.get("be_win_rate", 0.0), 0.0),
            "mc_tp": self._safe_float(mc_meta.get("tp", 0.0), 0.0),
            "mc_sl": self._safe_float(mc_meta.get("sl", 0.0), 0.0),
            "mc_hit_rate": self._safe_float(mc_meta.get("hit_rate", 0.0), 0.0),

            # policy-rollforward quick checks (B/C/D)
            # ✅ 검증 포인트: policy_ev_mix_long/short (Multi-Horizon Policy Mix)
            "policy_ev_long": policy_ev_long,
            "policy_ev_short": policy_ev_short,
            "policy_ev_mix_long": policy_ev_long,  # alias for consistency
            "policy_ev_mix_short": policy_ev_short,  # alias for consistency
            "policy_ev_gap": policy_ev_gap,
            "policy_p_pos_long": policy_p_pos_long,
            "policy_p_pos_short": policy_p_pos_short,
            "policy_p_pos_gap": policy_p_pos_gap,
            "hold_best_ev_long": hold_best_ev_long,
            "hold_best_ev_short": hold_best_ev_short,
            "policy_exit_unrealized_dd_frac": policy_exit_unrealized_dd_frac,
            "policy_exit_hold_bad_frac": policy_exit_hold_bad_frac,
            "policy_exit_score_flip_frac": policy_exit_score_flip_frac,
            "policy_signal_strength": policy_signal_strength,
            "policy_weight_peak_h": policy_weight_peak_h,
            "policy_half_life_sec": policy_half_life_sec,
            "policy_h_eff_sec": policy_h_eff_sec,
            "policy_h_eff_sec_prior": policy_h_eff_sec_prior,  # [DIFF 3] Rule-based only (for validation)
            "policy_w_short_sum": policy_w_short_sum,
            "policy_exit_time_mean_sec": policy_exit_time_mean_sec,
            "policy_horizons": policy_horizons,
            "policy_w_h": policy_w_h,
            # ✅ 검증 포인트: paths_reused 메타 확인 가능
            "paths_reused": paths_reused,
            "policy_exit_reason_counts_per_h": policy_exit_reason_counts_per_h,
            "policy_exit_reason_counts_per_h_long": policy_exit_reason_counts_per_h_long,
            "policy_exit_reason_counts_per_h_short": policy_exit_reason_counts_per_h_short,
            "ev_decomp_mu_annual": ev_decomp_mu_annual,
            "ev_decomp_mu_per_sec": ev_decomp_mu_per_sec,
            "ev_decomp_fee_roundtrip_total": ev_decomp_fee_rt,
            "ev_decomp_gross_long_600": ev_decomp_gross_long_600,
            "ev_decomp_gross_long_1800": ev_decomp_gross_long_1800,
            "ev_decomp_gross_short_600": ev_decomp_gross_short_600,
            "ev_decomp_gross_short_1800": ev_decomp_gross_short_1800,
            "ev_decomp_net_long_600": ev_decomp_net_long_600,
            "ev_decomp_net_long_1800": ev_decomp_net_long_1800,
            "ev_decomp_net_short_600": ev_decomp_net_short_600,
            "ev_decomp_net_short_1800": ev_decomp_net_short_1800,
            "ev_decomp_mu_req_annual_600": ev_decomp_mu_req_annual_600,
            "ev_decomp_mu_req_annual_1800": ev_decomp_mu_req_annual_1800,
            "ev_decomp_mu_req_annual_exit_mean": ev_decomp_mu_req_annual_exit_mean,
            "mu_alpha": mu_alpha,
            "mu_alpha_raw": mu_alpha_raw,
            "mu_alpha_mom": mu_alpha_mom,
            "mu_alpha_ofi": mu_alpha_ofi,
            "mu_alpha_regime_scale": mu_alpha_regime_scale,
            "mu_alpha_mom_15": mu_alpha_mom_15,
            "mu_alpha_mom_30": mu_alpha_mom_30,
            "mu_alpha_mom_60": mu_alpha_mom_60,
            "mu_alpha_mom_120": mu_alpha_mom_120,
            # ✅ [FIX 2] PMaker mu_alpha boost 정보
            "mu_alpha_pmaker_fill_rate": mu_alpha_pmaker_fill_rate,
            "mu_alpha_pmaker_boost": mu_alpha_pmaker_boost,
            "mu_alpha_before_pmaker": mu_alpha_before_pmaker,
            # spread diagnostics
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_pct_book": spread_pct_book,
            "spread_cap": spread_cap,
            "spread_entry_max": spread_entry_max,
            # PMaker visibility (ensure EV cost model is actually wired)
            "pmaker_override_used": pmaker_override_used,
            "pmaker_entry": pmaker_entry,
            "pmaker_entry_delay_sec": pmaker_entry_delay_sec,
            "pmaker_exit": pmaker_exit,
            "pmaker_exit_delay_sec": pmaker_exit_delay_sec,
            "pmaker_entry_delay_penalty_r": pmaker_entry_delay_penalty_r,
            "pmaker_exit_delay_penalty_r": pmaker_exit_delay_penalty_r,
            "policy_entry_shift_steps": policy_entry_shift_steps,
            "policy_horizon_eff_sec": policy_horizon_eff_sec,
            "policy_direction": policy_direction,
            "exec_mode": exec_mode,
            "p_maker": p_maker,
            "fee_roundtrip_fee_mix": fee_roundtrip_fee_mix,
            "fee_roundtrip_fee_taker": fee_roundtrip_fee_taker,  # [DIFF 5] For validation
            "fee_roundtrip_fee_maker": fee_roundtrip_fee_maker,  # [DIFF 5] For validation
            "fee_roundtrip_total": fee_roundtrip_total,
            
            # [DIFF 6] PMaker Survival model status (for validation)
            "pmaker": self.pmaker.status_dict(),
            "exec_maker_fill_rate": float(maker_path_fill_rate) if maker_path_fill_rate is not None else None,
            "exec_fallback_rate": float(fallback_rate) if fallback_rate is not None else None,
            "exec_cancel_rate": float(cancel_rate) if cancel_rate is not None else None,
            "exec_fill_delay_p95_ms": fill_delay_p95,

            # ✅ 대시보드 렌더링 성능을 위해 details(대용량)를 기본 제외
            "details": (decision.get("details", []) if (decision and DASHBOARD_INCLUDE_DETAILS) else []),
        }

    def _total_open_notional(self) -> float:
        return sum(float(pos.get("notional", 0.0)) for pos in self.positions.values())

    def _can_enter_position(self, notional: float) -> tuple[bool, str]:
        if self.position_cap_enabled and len(self.positions) >= self.max_positions:
            return False, "max positions reached"
        if self.exposure_cap_enabled and (self._total_open_notional() + notional) > (self.balance * self.max_notional_frac):
            return False, "exposure capped"
        return True, ""

    def _entry_permit(self, sym: str, decision: dict, ts_ms: int) -> tuple[bool, str]:
        # ✅ 비활성화: cooldown 체크 (2번)
        # if int(ts_ms) < int(self._cooldown_until.get(sym, 0) or 0):
        #     self._entry_streak[sym] = 0
        #     return False, "cooldown"
        
        meta = (decision.get("meta") or {}) if decision else {}
        ev = float(decision.get("ev", 0.0) or 0.0)
        win = float(decision.get("confidence", 0.0) or 0.0)
        
        # ✅ 단순 EV 진입 임계값: 환경 변수로 설정된 고정값이 있으면 우선 사용
        if EV_ENTRY_THRESHOLD > 0:
            ev_thr = EV_ENTRY_THRESHOLD
        else:
            # 동적 임계값 사용 (기존 로직)
            ev_thr = float(meta.get("ev_entry_threshold", 0.0) or 0.0)
            ev_thr_dyn = meta.get("ev_entry_threshold_dyn")
            if ev_thr_dyn is not None:
                try:
                    ev_thr = max(ev_thr, float(ev_thr_dyn))
                except Exception:
                    pass
        
        # ✅ 비활성화: 중기 필터 (3번)
        # ev_mid = meta.get("ev_mid")
        # win_mid = meta.get("win_mid")
        # if ev_mid is not None and ev_mid < 0:
        #     self._entry_streak[sym] = 0
        #     return False, "mid_ev_neg"
        # if win_mid is not None and win_mid < 0.50:
        #     self._entry_streak[sym] = 0
        #     return False, "mid_win_low"

        # ✅ 4번 유지: 동적 문턱값 체크 (ev_entry_threshold / ev_entry_threshold_dyn)
        # ✅ 비활성화: win_entry_threshold 체크 (5번)
        # ✅ 비활성화: streak 체크 (6번)
        # EV만 체크하고 win과 streak는 무시
        if ev >= ev_thr:
            # ✅ streak 체크 비활성화 - 항상 통과
            return True, ""
        else:
            # ✅ EV가 높은데 threshold 때문에 차단된 경우 상세 로그
            if ev >= 0.004:  # EV가 0.4% 이상인데 차단된 경우만 로그
                self._log(f"[ENTRY_PERMIT] {sym} threshold: ev={ev:.6f} < ev_thr={ev_thr:.6f}")
            return False, "threshold"

    def _calc_position_size(self, decision: dict, price: float, leverage: float, size_frac_override: float | None = None) -> tuple[float, float, float]:
        meta = (decision or {}).get("meta", {}) or {}
        size_frac = size_frac_override if size_frac_override is not None else decision.get("size_frac") or meta.get("size_fraction") or self.default_size_frac
        cap_frac = meta.get("regime_cap_frac")
        if cap_frac is not None:
            try:
                size_frac = min(size_frac, float(cap_frac))
            except Exception:
                pass
        # 상한 제거: 신호가 강하면 엔진이 제시한 비중을 그대로 사용
        size_frac = float(max(0.0, size_frac))
        notional = float(max(0.0, self.balance * size_frac * leverage))
        qty = float(notional / price) if price and notional > 0 else 0.0
        return size_frac, notional, qty

    #def _dynamic_leverage_risk(self, decision: dict, ctx: dict) -> float:
    def _dynamic_leverage_risk(self, decision: dict, ctx: dict) -> float:
        def _f(x, default=0.0):
            try:
                if x is None:
                    return float(default)
                return float(x)
            except Exception:
                return float(default)

        regime = ctx.get("regime") or "chop"
        sigma = _f(ctx.get("sigma"), 0.0)
        meta = decision.get("meta") or {}
        ev = _f(decision.get("ev"), 0.0)
        cvar = _f(meta.get("cvar05", decision.get("cvar")), 0.0)
        event_p_sl = _f(meta.get("event_p_sl"), 0.0)
        spread_pct = _f(meta.get("spread_pct", ctx.get("spread_pct")), 0.0002)
        execution_cost = _f(meta.get("execution_cost", meta.get("fee_rt")), 0.0)
        slippage_pct = _f(meta.get("slippage_pct", 0.0), 0.0)
        event_cvar_pct = _f(meta.get("event_cvar_pct"), 0.0)

        # risk = max(|CVaR|, |event_cvar_pct|) + 0.7*spread + 0.5*slippage + 0.5*sigma (+ p_sl 가중)
        risk_score = max(abs(cvar), abs(event_cvar_pct)) + 0.7 * spread_pct + 0.5 * slippage_pct + 0.5 * sigma + 0.2 * event_p_sl
        if risk_score <= 1e-6:
            risk_score = 1e-6

        lev_max_map = {"bull": self.max_leverage, "bear": self.max_leverage, "chop": min(self.max_leverage, 30.0), "volatile": min(self.max_leverage, 20.0)}
        lev_max = lev_max_map.get(regime, self.max_leverage)

        # EV를 위험 대비 비례 반영 (음수 EV면 최소 1배로 수렴)
        lev_raw = (max(ev - execution_cost, 0.0) / risk_score) * K_LEV
        lev = float(max(1.0, min(lev_max, lev_raw)))
        sym = ctx.get("symbol")
        if sym:
            self._dyn_leverage[sym] = lev
        return lev

    def _direction_bias(self, closes) -> int:
        if not closes:
            return 1
        price_now = closes[-1]
        fast_period = min(10, len(closes))
        slow_period = min(40, len(closes))
        fast = sum(closes[-fast_period:]) / fast_period
        slow = sum(closes[-slow_period:]) / slow_period
        bias = 1 if fast >= slow else -1
        # 레짐 힌트: 상승장에서는 롱 우선, 하락장에서는 숏 우선 (더 민감하게)
        if len(closes) >= 20:
            slope_short = closes[-1] - closes[-20]
            pct_short = slope_short / max(price_now, 1e-6)
            if pct_short > 0.002:
                bias = 1
            elif pct_short < -0.002:
                bias = -1
        if len(closes) >= 60:
            slope_long = closes[-1] - closes[-60]
            pct_long = slope_long / max(price_now, 1e-6)
            if pct_long > 0.004:
                bias = 1
            elif pct_long < -0.004:
                bias = -1
        # 최근 방향 비율로 편향 교정
        if len(closes) >= 15:
            rets = [1 if closes[i] >= closes[i - 1] else -1 for i in range(len(closes) - 14, len(closes))]
            up_ratio = sum(1 for r in rets if r > 0) / len(rets)
            if up_ratio >= 0.6:
                bias = 1
            elif up_ratio <= 0.4:
                bias = -1
        return bias

    def _infer_regime(self, closes) -> str:
        if not closes or len(closes) < 30:
            return "chop"
        fast_period = min(80, len(closes))
        slow_period = min(200, len(closes))
        fast = sum(closes[-fast_period:]) / fast_period
        slow = sum(closes[-slow_period:]) / slow_period
        slope_short = closes[-1] - closes[max(0, len(closes) - 6)]
        slope_long = closes[-1] - closes[max(0, len(closes) - 40)]
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, min(len(closes), 180))]
        vol = float(np.std(rets)) if rets else 0.0
        # 변동성 높고 추세 약하면 volatile
        if vol > 0.01 and abs(slope_short) < closes[-1] * 0.0015:
            return "volatile"
        # 강한 상승/하락을 더 민감하게
        if fast > slow and slope_long > 0 and slope_short > 0:
            return "bull"
        if fast < slow and slope_long < 0 and slope_short < 0:
            return "bear"
        return "chop"

    def _compute_ofi_score(self, sym: str) -> float:
        ob = self.data.orderbook.get(sym)
        if not ob:
            return 0.0
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        # 심도별 가중(상위호가 가중치 ↑)
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        bid_vol = sum(float(b[1]) * weights[i] for i, b in enumerate(bids[: len(weights)]) if len(b) >= 2)
        ask_vol = sum(float(a[1]) * weights[i] for i, a in enumerate(asks[: len(weights)]) if len(a) >= 2)
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return float((bid_vol - ask_vol) / denom)

    def _liquidity_score(self, sym: str) -> float:
        ob = self.data.orderbook.get(sym)
        if not ob:
            return 1.0
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        vol = sum(float(b[1]) for b in bids[:5] if len(b) >= 2) + sum(float(a[1]) for a in asks[:5] if len(a) >= 2)
        return float(max(vol, 1.0))

    def _enter_position(
        self,
        sym: str,
        side: str,
        price: float,
        decision: dict | None,
        ts: int,
        *,
        ctx: dict | None = None,
        leverage_override: float | None = None,
        size_frac_override: float | None = None,
        hold_limit_override: int | None = None,
        tag: str | None = None,
    ) -> None:
        if price is None:
            return
        decision = decision or {}
        side_raw = str(side).upper()
        if side_raw in ("BUY", "LONG"):
            side_norm = "LONG"
        elif side_raw in ("SELL", "SHORT"):
            side_norm = "SHORT"
        else:
            return

        lev = leverage_override
        if lev is None:
            lev = float(decision.get("leverage") or self._dyn_leverage.get(sym, self.leverage) or self.leverage)
        size_frac, notional, qty = self._calc_position_size(decision, price, lev, size_frac_override)
        if notional <= 0 or qty <= 0:
            return

        meta = decision.get("meta") or {}
        mc_meta = meta
        for d in decision.get("details", []) or []:
            if d.get("_engine") in ("mc_barrier", "mc_engine", "mc"):
                mc_meta = d.get("meta") or meta
                break

        exec_mode = str(mc_meta.get("exec_mode") or meta.get("exec_mode") or os.environ.get("EXEC_MODE", EXEC_MODE)).lower()
        hold_limit_ms = int(hold_limit_override if hold_limit_override is not None else MAX_POSITION_HOLD_SEC * 1000)

        pos = {
            "symbol": sym,
            "side": side_norm,
            "entry_price": float(price),
            "quantity": float(qty),
            "notional": float(notional),
            "leverage": float(lev),
            "size_frac": float(size_frac),
            "cap_frac": float(notional / self.balance) if self.balance else 0.0,
            "entry_time": int(ts),
            "hold_limit": hold_limit_ms,
            "tag": tag,
            "exec_mode": exec_mode,
            "pred_win": decision.get("confidence"),
            "pred_ev": decision.get("ev"),
            "pred_event_ev_r": mc_meta.get("event_ev_r"),
            "pred_event_p_tp": mc_meta.get("event_p_tp"),
            "pred_event_p_sl": mc_meta.get("event_p_sl"),
            "consensus_used": self._consensus_used_flag(decision) if decision else False,
        }

        policy_ev_long = mc_meta.get("policy_ev_mix_long")
        policy_ev_short = mc_meta.get("policy_ev_mix_short")
        policy_win_long = mc_meta.get("policy_p_pos_mix_long")
        policy_win_short = mc_meta.get("policy_p_pos_mix_short")
        if side_norm == "LONG":
            pos["entry_policy_ev_side"] = policy_ev_long
            pos["entry_policy_win_side"] = policy_win_long
        else:
            pos["entry_policy_ev_side"] = policy_ev_short
            pos["entry_policy_win_side"] = policy_win_short

        decision_snap = dict(decision)
        meta_snap = dict(meta)
        if mc_meta is not meta:
            meta_snap.update(mc_meta)
        decision_snap["meta"] = meta_snap
        self._update_pos_pred_snapshot_from_decision(pos, decision_snap)

        self.positions[sym] = pos
        self._last_actions[sym] = "ENTER"

        fee_rate = self.fee_maker if exec_mode.startswith("maker") else self.fee_taker
        fee = float(notional) * float(fee_rate)
        self.balance -= fee
        self._equity_history.append({"time": int(ts), "equity": float(self.balance)})

        reason = decision.get("reason")
        self._log(f"[{sym}] ENTER {side_norm} qty={qty:.4f} notional={notional:.2f} fee={fee:.4f} reason={reason}")
        self._record_trade("ENTER", sym, side_norm, price, qty, pos, fee=fee, reason=reason)

        if self.enable_orders:
            order_side = "buy" if side_norm == "LONG" else "sell"
            self._maybe_place_order(sym, order_side, qty, reduce_only=False, decision=decision_snap)

    def _close_position(
        self,
        sym: str,
        price: float,
        reason: str,
        *,
        exit_kind: str = "MANUAL",
        decision: dict | None = None,
    ) -> None:
        pos = self.positions.get(sym)
        if not pos or price is None:
            return
        qty = float(pos.get("quantity", 0.0))
        if qty <= 0:
            return
        entry = float(pos.get("entry_price", price))
        side = pos.get("side")
        notional = float(pos.get("notional", 0.0))
        pnl = (price - entry) * qty if side == "LONG" else (entry - price) * qty
        fee = float(notional) * float(self.fee_taker)
        pnl_net = float(pnl) - fee

        self.balance += pnl_net
        ts = now_ms()
        self._equity_history.append({"time": int(ts), "equity": float(self.balance)})

        self._log(f"[{sym}] EXIT {side} qty={qty:.4f} pnl={pnl_net:.4f} fee={fee:.4f} reason={reason}")
        self._record_trade("EXIT", sym, side, price, qty, pos, pnl=pnl_net, fee=fee, reason=reason)
        self.positions.pop(sym, None)

        self._last_actions[sym] = f"EXIT_{exit_kind}"
        self._mark_exit_and_cooldown(sym, exit_kind, ts)

        if self.enable_orders:
            close_side = "sell" if side == "LONG" else "buy"
            asyncio.create_task(
                self._execute_close_order(
                    symbol=sym,
                    side=close_side,
                    qty=qty,
                    price=price,
                    decision=decision,
                )
            )

    def _pmaker_update_attempt(
        self,
        *,
        symbol: str,
        side: str,
        maker_price: float,
        attempt_idx: int,
        bid: float,
        ask: float,
        timeout_ms: int,
        first_fill_delay_ms: int | None,
        qty_attempt: float,
        qty_filled: float,
        decision: dict | None,
        sigma: float | None,
    ) -> None:
        if not self.pmaker.enabled or self.pmaker.surv is None:
            return
        try:
            mid = (float(bid) + float(ask)) / 2.0 if (bid and ask) else float(maker_price)
            spread_pct = float((float(ask) - float(bid)) / mid) if mid > 0 and bid and ask else 0.0
            rel_px = float((float(maker_price) - mid) / mid) if mid > 0 else 0.0
            feats = {
                "spread_pct": spread_pct,
                "sigma": float(sigma) if sigma is not None else 0.0,
                "ofi_z": float(self._compute_ofi_score(symbol)),
                "momentum_z": 0.0,
                "liq_score": float(self._liquidity_score(symbol)),
                "attempt_idx": float(attempt_idx),
                "rel_px": rel_px,
            }
            x = self.pmaker.surv.featurize(feats)
            self.pmaker.surv.update_one_attempt(
                symbol,
                x,
                timeout_ms=timeout_ms,
                first_fill_delay_ms=first_fill_delay_ms,
                qty_attempt=qty_attempt,
                qty_filled=qty_filled,
            )
            self._pmaker_dirty = True
        except Exception as e:
            self._log_err(f"[PMAKER_UPDATE] {symbol} attempt={attempt_idx} err={e}")

    def _maybe_place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        decision: dict | None = None,
    ):
        if not self.enable_orders or quantity <= 0:
            return
        asyncio.create_task(self._execute_order(symbol, side, quantity, reduce_only, decision=decision))

    async def _execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        decision: dict | None = None,
    ):
        if not self.enable_orders:
            return None
        st = self._exec_stats_for(symbol)
        st["order_requests"] += 1

        side_raw = str(side).lower()
        order_side = "buy" if side_raw in ("buy", "long") else "sell"
        decision = decision or {}
        meta = decision.get("meta") or {}
        exec_mode = str(meta.get("exec_mode") or os.environ.get("EXEC_MODE", EXEC_MODE)).lower().strip()

        if exec_mode != "maker_then_market":
            st["order_requests_market"] += 1
            params = {"reduceOnly": True} if reduce_only else {}
            try:
                return await self._ccxt_call(
                    "create_order_market",
                    self.exchange.create_order,
                    symbol,
                    "market",
                    order_side,
                    quantity,
                    None,
                    params,
                )
            except Exception as e:
                self._log_err(f"[EXEC] market order failed {symbol} {order_side} qty={quantity}: {e}")
                return None

        st["order_requests_maker_path"] += 1
        bid, ask = self._best_bid_ask(symbol)
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            st["order_requests_fallback_market"] += 1
            st["order_requests_market"] += 1
            params = {"reduceOnly": True} if reduce_only else {}
            try:
                return await self._ccxt_call(
                    "create_order_market_fallback",
                    self.exchange.create_order,
                    symbol,
                    "market",
                    order_side,
                    quantity,
                    None,
                    params,
                )
            except Exception as e:
                self._log_err(f"[EXEC] fallback market failed {symbol} {order_side} qty={quantity}: {e}")
                return None

        ob = self.data.orderbook.get(symbol) or {}
        tick = self._infer_tick_size_from_ladders(symbol, bid, ask, ob.get("bids"), ob.get("asks"))
        maker_timeout_ms = int(os.environ.get("MAKER_TIMEOUT_MS", str(MAKER_TIMEOUT_MS)))
        maker_poll_ms = int(os.environ.get("MAKER_POLL_MS", str(MAKER_POLL_MS)))
        maker_retries = int(os.environ.get("MAKER_RETRIES", str(MAKER_RETRIES)))
        behind_ticks = int(os.environ.get("MAKER_BEHIND_TICKS", "0"))

        sigma = None
        try:
            sigma = float(meta.get("sigma_sim") or meta.get("sigma") or meta.get("sigma_annual"))
        except Exception:
            sigma = None

        remaining = float(quantity)
        maker_filled_all = False
        last_order = None

        max_attempts = max(1, maker_retries + 1)
        for attempt in range(max_attempts):
            if remaining <= 0:
                maker_filled_all = True
                break

            maker_price = self._pmaker_probe_price(order_side, bid, ask, tick, attempt, behind_ticks=behind_ticks)
            params = {"postOnly": True}
            if reduce_only:
                params["reduceOnly"] = True

            st["maker_limit_orders"] += 1
            start_ms = now_ms()
            order = None
            order_id = None
            filled_qty = 0.0
            first_fill_delay_ms = None
            status = "open"

            try:
                order = await self._ccxt_call(
                    "create_order_maker",
                    self.exchange.create_order,
                    symbol,
                    "limit",
                    order_side,
                    remaining,
                    maker_price,
                    params,
                )
                order_id = order.get("id") if isinstance(order, dict) else None
            except Exception as e:
                msg = str(e).lower()
                if "post only" in msg or "postonly" in msg:
                    st["maker_reject"] += 1
                else:
                    st["maker_other_err"] += 1
                self._log_err(f"[EXEC] maker create failed {symbol} {order_side} qty={remaining}: {e}")
                self._pmaker_update_attempt(
                    symbol=symbol,
                    side=order_side,
                    maker_price=maker_price,
                    attempt_idx=attempt,
                    bid=bid,
                    ask=ask,
                    timeout_ms=maker_timeout_ms,
                    first_fill_delay_ms=None,
                    qty_attempt=remaining,
                    qty_filled=0.0,
                    decision=decision,
                    sigma=sigma,
                )
                continue

            if order_id is None:
                st["maker_other_err"] += 1
                self._pmaker_update_attempt(
                    symbol=symbol,
                    side=order_side,
                    maker_price=maker_price,
                    attempt_idx=attempt,
                    bid=bid,
                    ask=ask,
                    timeout_ms=maker_timeout_ms,
                    first_fill_delay_ms=None,
                    qty_attempt=remaining,
                    qty_filled=0.0,
                    decision=decision,
                    sigma=sigma,
                )
                continue

            end_ms = start_ms + maker_timeout_ms
            while now_ms() < end_ms:
                await asyncio.sleep(max(0.05, maker_poll_ms / 1000.0))
                try:
                    ord_now = await self._ccxt_call("fetch_order", self.exchange.fetch_order, order_id, symbol)
                except Exception:
                    continue
                status, amount, filled, remaining_now, price_f = self._extract_order_stats(ord_now)
                if filled > filled_qty:
                    filled_qty = filled
                    if first_fill_delay_ms is None:
                        first_fill_delay_ms = now_ms() - start_ms
                if status in ("closed", "canceled"):
                    break

            try:
                trades = await self._fetch_order_trades_any(order_id, symbol, since_ms=start_ms)
                if trades:
                    t_first = None
                    t_qty = 0.0
                    for tr in trades:
                        t_ts = tr.get("timestamp")
                        if t_ts is not None:
                            t_first = t_ts if t_first is None else min(t_first, t_ts)
                        amt = tr.get("amount")
                        if amt is None:
                            amt = (tr.get("info") or {}).get("execQty")
                        try:
                            if amt is not None:
                                t_qty += float(amt)
                        except Exception:
                            pass
                    if t_first is not None:
                        first_fill_delay_ms = max(0, int(t_first) - int(start_ms))
                    if t_qty > 0:
                        filled_qty = max(filled_qty, t_qty)
            except Exception:
                pass

            timed_out = False
            if status not in ("closed", "canceled") and (now_ms() - start_ms >= maker_timeout_ms):
                timed_out = True
                try:
                    await self._ccxt_call("cancel_order", self.exchange.cancel_order, order_id, symbol)
                    st["maker_limit_cancel_ok"] += 1
                    status = "canceled"
                except Exception as e:
                    st["maker_limit_cancel_err"] += 1
                    self._log_err(f"[EXEC] maker cancel failed {symbol} order_id={order_id}: {e}")

            if filled_qty > 0:
                st["maker_limit_filled"] += 1
                if first_fill_delay_ms is not None:
                    st["lat_ms_maker_first_fill"].append(first_fill_delay_ms)
            if timed_out and filled_qty <= 0:
                st["maker_limit_timeout"] += 1

            self._pmaker_update_attempt(
                symbol=symbol,
                side=order_side,
                maker_price=maker_price,
                attempt_idx=attempt,
                bid=bid,
                ask=ask,
                timeout_ms=maker_timeout_ms,
                first_fill_delay_ms=first_fill_delay_ms,
                qty_attempt=remaining,
                qty_filled=filled_qty,
                decision=decision,
                sigma=sigma,
            )

            remaining = max(0.0, float(remaining) - float(filled_qty))
            last_order = order
            if remaining <= 0:
                maker_filled_all = True
                if first_fill_delay_ms is not None:
                    st["lat_ms_maker_filled"].append(first_fill_delay_ms)
                break

        if maker_filled_all:
            st["order_requests_maker_filled"] += 1
            return last_order

        if remaining > 0:
            st["order_requests_fallback_market"] += 1
            st["order_requests_market"] += 1
            start_ms = now_ms()
            params = {"reduceOnly": True} if reduce_only else {}
            try:
                order = await self._ccxt_call(
                    "create_order_market_fallback",
                    self.exchange.create_order,
                    symbol,
                    "market",
                    order_side,
                    remaining,
                    None,
                    params,
                )
                st["lat_ms_fallback_market"].append(now_ms() - start_ms)
                return order
            except Exception as e:
                self._log_err(f"[EXEC] fallback market failed {symbol} {order_side} qty={remaining}: {e}")
                return None

    async def _execute_close_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        기존 close/exit 주문 함수가 있으면 거기서 meta merge만 하면 됨.
        없으면 네 코드 구조에 맞춰 close 주문을 호출하는 지점(청산 주문 직전)에 이 블록을 붙여.
        """
        meta = meta or {}
        if self.PMAKER_PREDICT_EXIT and self.PMAKER_USE_SURVIVAL:
            # close side 기준으로 exit prediction
            try:
                bid, ask = self._best_bid_ask(symbol)
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    spread_pct = (float(ask) - float(bid)) / float(bid)
                else:
                    bid = ask = 0.0
                    spread_pct = 0.0
            except Exception:
                bid = ask = 0.0
                spread_pct = 0.0
            
            # ✅ Use async background prediction (non-blocking)
            pred_exit = await self.pmaker.request_prediction(
                symbol=symbol,
                side=side,
                price=price,
                best_bid=float(bid) if bid is not None else 0.0,
                best_ask=float(ask) if ask is not None else 0.0,
                spread_pct=spread_pct,
                qty=qty,
                maker_timeout_ms=int(os.environ.get("MAKER_TIMEOUT_MS", str(MAKER_TIMEOUT_MS))),
                prefix="pmaker_exit",
                use_cache=True,
            )
            if pred_exit:
                meta.update(pred_exit)
        
        # 이후 실제 close 주문 호출(기존 로직 유지)
        decision_exit = kwargs.get("decision")
        if decision_exit is None:
            decision_exit = {"meta": meta}
        else:
            decision_exit.setdefault("meta", {}).update(meta)
        kwargs["decision"] = decision_exit
        return await self._execute_order(symbol=symbol, side=side, quantity=qty, reduce_only=True, **kwargs)

    def _best_bid_ask(self, sym: str) -> tuple[float | None, float | None]:
        ob = self.data.orderbook.get(sym) or {}
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        try:
            bid = float(bids[0][0]) if bids and len(bids[0]) >= 2 else None
        except Exception:
            bid = None
        try:
            ask = float(asks[0][0]) if asks and len(asks[0]) >= 2 else None
        except Exception:
            ask = None
        if bid is not None and bid <= 0:
            bid = None
        if ask is not None and ask <= 0:
            ask = None
        if bid is None or ask is None:
            ticker = self.data.market.get(sym, {})
            t_bid = ticker.get("bid")
            t_ask = ticker.get("ask")
            if t_bid is not None and t_ask is not None and t_bid > 0 and t_ask > 0:
                bid = float(t_bid)
                ask = float(t_ask)
        return bid, ask

    def _infer_tick_size(self, sym: str, bid: float, ask: float) -> float:
        # 1) exchange precision if available
        try:
            m = self.exchange.market(sym)  # ccxt
            p = (m or {}).get("precision", {}).get("price")
            if p is not None:
                # CCXT precision can be either:
                #   - number of decimals (e.g., 2 -> tick 0.01)
                #   - step size itself (e.g., 0.01 -> tick 0.01)  [Bybit v5 often returns this]
                fp = float(p)
                if fp > 0.0 and fp < 1.0:
                    return float(fp)
                pp = int(fp)
                if pp >= 0 and abs(fp - float(pp)) < 1e-12:
                    return float(10 ** (-pp))
                if fp > 0.0:
                    return float(fp)
        except Exception:
            pass

        # 2) infer from top-of-book ladders
        diffs: list[float] = []
        ob = self.data.orderbook.get(sym) or {}
        for side in (ob.get("bids") or [], ob.get("asks") or []):
            for i in range(min(len(side) - 1, 4)):
                try:
                    d = abs(float(side[i][0]) - float(side[i + 1][0]))
                    if d > 0:
                        diffs.append(d)
                except Exception:
                    continue
        if diffs:
            return float(max(1e-12, min(diffs)))

        # 3) fallback: relative tick
        mid = 0.5 * (float(bid) + float(ask))
        return float(max(1e-12, mid * 1e-6))

    def _infer_tick_size_from_ladders(
        self,
        sym: str,
        bid: float,
        ask: float,
        bids: list | None,
        asks: list | None,
    ) -> float:
        tick_prec = self._infer_tick_size(sym, bid, ask)
        diffs: list[float] = []
        for side in ((bids or []), (asks or [])):
            for i in range(min(len(side) - 1, 6)):
                try:
                    d = abs(float(side[i][0]) - float(side[i + 1][0]))
                    if d > 0:
                        diffs.append(d)
                except Exception:
                    continue
        spread = abs(float(ask) - float(bid))
        tick = float(tick_prec)
        if diffs:
            tick = float(min(tick, float(min(diffs))))
        # Guardrail: never let "tick" exceed observed spread when we have a tight market;
        # this prevents rounding that moves a touch price far away (e.g., tick=1 with 0.01 spread).
        if spread > 0:
            tick = float(min(tick, spread))
        return float(max(1e-12, tick))

    def _round_price_to_tick(self, price: float, tick: float) -> float:
        t = float(max(1e-12, tick))
        return float(round(price / t) * t)


    def _extract_order_stats(self, order: dict | None) -> tuple[str, float, float, float, float | None]:
        """
        Returns: (status, amount, filled, remaining, price)
        Tries both CCXT normalized fields and Bybit raw 'info' fallbacks.
        """
        o = order or {}
        info = o.get("info") if isinstance(o, dict) else None
        info = info if isinstance(info, dict) else {}

        def _pick(*keys: str):
            for k in keys:
                v = info.get(k)
                if v is not None and v != "":
                    return v
            return None

        status = str((o.get("status") or _pick("orderStatus", "status") or "")).lower()
        if status in ("filled",):
            status = "closed"
        if status in ("cancelled",):
            status = "canceled"

        def _f(x) -> float:
            try:
                if x is None or x == "":
                    return 0.0
                return float(x)
            except Exception:
                return 0.0

        amount = _f(o.get("amount") or _pick("qty", "orderQty", "origQty"))
        filled = _f(o.get("filled") or _pick("cumExecQty", "cum_exec_qty", "executedQty", "execQty", "cumQty"))

        remaining = o.get("remaining")
        if remaining is None:
            remaining = _pick("leavesQty", "leaves_qty", "remainingQty")
        rem_f = None
        if remaining is not None and remaining != "":
            try:
                rem_f = float(remaining)
            except Exception:
                rem_f = None
        if rem_f is None:
            rem_f = max(0.0, amount - filled) if amount > 0 else 0.0

        price = o.get("price")
        if price is None:
            price = _pick("price", "orderPrice", "avgPrice", "avg_price")
        price_f = None
        if price is not None and price != "":
            try:
                price_f = float(price)
            except Exception:
                price_f = None

        return status, float(amount), float(filled), float(rem_f), price_f

    async def _fetch_order_trades_any(self, order_id: str, symbol: str, *, since_ms: int) -> list[dict]:
        """
        Best-effort trade fetch for an order:
          - prefer fetch_order_trades
          - fallback to fetch_my_trades filtered by order id
        """
        try:
            has = getattr(self.exchange, "has", {}) or {}
            if has.get("fetchOrderTrades"):
                trades = await self.exchange.fetch_order_trades(order_id, symbol)
                if trades:
                    return list(trades)
        except Exception:
            pass

        try:
            has = getattr(self.exchange, "has", {}) or {}
            if has.get("fetchMyTrades"):
                # Try Bybit-compatible orderId filter first.
                try:
                    trades = await self.exchange.fetch_my_trades(symbol, since=since_ms, limit=50, params={"orderId": order_id})
                except Exception:
                    trades = await self.exchange.fetch_my_trades(symbol, since=since_ms, limit=50)
                out = []
                for tr in trades or []:
                    try:
                        if str(tr.get("order") or "") == str(order_id):
                            out.append(tr)
                    except Exception:
                        pass
                return out
        except Exception:
            pass

        return []

    def _pmaker_probe_price(self, side: str, bid: float, ask: float, tick: float, k: int, *, behind_ticks: int) -> float:
        """
        Probe quote builder (post-only safe):
          - behind_ticks > 0 places the initial quote *behind* the touch to induce non-fill samples.
          - k then improves the quote by k ticks toward the opposite side (still capped to avoid taking).
        """
        bt = max(0, int(behind_ticks))
        kk = max(0, int(k))
        if str(side).lower() == "buy":
            base = float(bid) - float(bt) * float(tick)
            px = min(float(ask) - float(tick), float(base) + float(kk) * float(tick))
        else:
            base = float(ask) + float(bt) * float(tick)
            px = max(float(bid) + float(tick), float(base) - float(kk) * float(tick))
        return self._round_price_to_tick(float(px), float(tick))


    def _rebalance_position(self, sym: str, price: float, decision: dict, leverage_override: float | None = None):
        """
        기존 포지션이 있을 때 목표 비중과 현 포지션이 크게 다르면 수량/노출을 조정한다.
        실제 주문은 ENABLE_LIVE_ORDERS에 따라 별도 처리.
        """
        pos = self.positions.get(sym)
        if not pos or price is None:
            return
        lev = leverage_override if leverage_override is not None else pos.get("leverage", self.leverage)
        target_size_frac, target_notional, target_qty = self._calc_position_size(decision, price, lev)
        if target_notional <= 0:
            # 목표 노출이 0이면 전량 청산
            self._close_position(sym, price, "rebalance to zero")
            return
        curr_notional = float(pos.get("notional", 0.0))
        delta = abs(target_notional - curr_notional) / curr_notional if curr_notional else 1.0
        # 항상 레버리지/메타 업데이트
        if leverage_override is not None:
            pos["leverage"] = leverage_override
        if delta < REBALANCE_THRESHOLD_FRAC:
            return
        entry = float(pos.get("entry_price", price))
        side = pos.get("side")
        curr_qty = float(pos.get("quantity", 0.0))

        if target_notional < curr_notional and curr_notional > 0:
            # 부분 청산: 줄이는 비율만큼 실현 손익을 balance에 반영
            reduce_ratio = 1.0 - (target_notional / curr_notional)
            close_qty = curr_qty * reduce_ratio
            close_notional = curr_notional * reduce_ratio
            pnl_realized = (price - entry) * close_qty if side == "LONG" else (entry - price) * close_qty
            fee_partial = close_notional * (self.fee_taker if self.fee_mode == "taker" else self.fee_maker)
            pnl_realized_net = pnl_realized - fee_partial
            self.balance += pnl_realized_net

            # 남은 포지션 업데이트
            pos["quantity"] = max(curr_qty - close_qty, 0.0)
            pos["notional"] = max(target_notional, 0.0)
            pos["size_frac"] = target_size_frac
            pos["cap_frac"] = float(pos["notional"] / self.balance) if self.balance else 0.0

            # 부분 청산 기록
            close_pos = dict(pos)
            close_pos["notional"] = close_notional
            close_pos["quantity"] = close_qty
            close_pos["leverage"] = lev
            self._log(f"[{sym}] PARTIAL EXIT by REBAL qty={close_qty:.4f} pnl={pnl_realized_net:.2f} fee={fee_partial:.4f}")
            self._last_actions[sym] = "REBAL_EXIT"
            self._record_trade("REBAL_EXIT", sym, side, price, close_qty, close_pos, pnl=pnl_realized_net, fee=fee_partial, reason="rebalance partial")
            self._cooldown_until[sym] = now_ms() + int(COOLDOWN_SEC * 1000)
        else:
            # 노출 확대/동일: 포지션만 갱신
            pos["notional"] = target_notional
            pos["quantity"] = target_qty
            pos["size_frac"] = target_size_frac
            pos["cap_frac"] = float(target_notional / self.balance) if self.balance else 0.0
            self._log(f"[{sym}] REBALANCE qty={target_qty:.4f} notional={target_notional:.2f} size={target_size_frac:.2%}")
            self._last_actions[sym] = "REBAL"
            # 기록용 스냅샷
            pnl_now = (price - entry) * target_qty if side == "LONG" else (entry - price) * target_qty
            self._record_trade("REBAL", sym, side, price, target_qty, pos, pnl=pnl_now, reason="rebalance")

        # 실제 리밸런싱 주문 경로 (옵션)
        adj_qty = max(0.0, target_qty - float(pos.get("quantity", 0.0)))
        if adj_qty > 0:
            self._maybe_place_order(sym, pos["side"], adj_qty, reduce_only=False)
        self._persist_state(force=True)

    def _update_pos_pred_snapshot_from_decision(self, pos: dict, decision: dict | None) -> None:
        # exit 시점 스냅샷(ENTER 시점 의존 제거): 최신 decision.meta의 policy_*를 pos에 덮어쓴다.
        try:
            meta = (decision or {}).get("meta") or {}
            if not isinstance(meta, dict) or not meta:
                return
            for k_meta, k_pos in (
                ("policy_ev_mix_long", "pred_policy_ev_mix_long"),
                ("policy_ev_mix_short", "pred_policy_ev_mix_short"),
                ("policy_p_pos_mix_long", "pred_policy_p_pos_mix_long"),
                ("policy_p_pos_mix_short", "pred_policy_p_pos_mix_short"),
                ("policy_direction", "pred_policy_direction"),
                ("policy_signal_strength", "pred_policy_signal_strength"),
                ("policy_weight_peak_h", "pred_policy_weight_peak_h"),
                ("policy_half_life_sec", "pred_policy_half_life_sec"),
                ("policy_horizons", "pred_policy_horizons"),
                ("policy_w_h", "pred_policy_w_h"),
                ("exit_time_mean_sec", "pred_policy_exit_time_mean_sec"),
                ("policy_exit_t_mean_per_h", "pred_policy_exit_t_mean_per_h"),
                ("policy_exit_t_p50_per_h", "pred_policy_exit_t_p50_per_h"),
                ("policy_exit_reason_counts_per_h", "pred_policy_exit_reason_counts_per_h"),
            ):
                v = meta.get(k_meta)
                if v is not None:
                    pos[k_pos] = v
            # gaps (for diagnostics)
            ev_l = pos.get("pred_policy_ev_mix_long")
            ev_s = pos.get("pred_policy_ev_mix_short")
            pp_l = pos.get("pred_policy_p_pos_mix_long")
            pp_s = pos.get("pred_policy_p_pos_mix_short")
            try:
                if ev_l is not None and ev_s is not None:
                    pos["pred_policy_ev_gap"] = float(ev_l) - float(ev_s)
            except Exception:
                pass
            try:
                if pp_l is not None and pp_s is not None:
                    pos["pred_policy_p_pos_gap"] = float(pp_l) - float(pp_s)
            except Exception:
                pass
        except Exception:
            return

    def _maybe_exit_position_unified(self, sym: str, price: float, decision: dict, ts: int) -> None:
        pos = self.positions.get(sym)
        if not pos or price is None:
            return

        def _f(x, default=None):
            try:
                if x is None:
                    return default
                return float(x)
            except Exception:
                return default

        side = pos.get("side")
        action = (decision or {}).get("action") or "WAIT"
        hold_limit_ms = int(pos.get("hold_limit", POSITION_HOLD_HARD_CAP_SEC * 1000) or 0)
        age_ms = int(ts - int(pos.get("entry_time", ts)))
        age_sec = max(0.0, float(age_ms) / 1000.0)

        entry = float(pos.get("entry_price", price))
        qty = float(pos.get("quantity", 0.0))
        notional = float(pos.get("notional", 0.0))
        pnl_unreal = (price - entry) * qty if side == "LONG" else (entry - price) * qty
        lev_safe = float(pos.get("leverage", self.leverage) or 1.0)
        base_notional = notional / max(lev_safe, 1e-6) if notional else 0.0
        roe_unreal = pnl_unreal / base_notional if base_notional else 0.0

        # 0) hard stop (with grace) — keep as safety even in unified mode
        if age_sec >= float(getattr(self, "_exit_stop_grace_sec", 0.0) or 0.0):
            thr = float(getattr(self, "_exit_stop_roe", -0.02) or -0.02)
            if roe_unreal <= thr:
                self._update_pos_pred_snapshot_from_decision(pos, decision)
                self._close_position(sym, price, f"unrealized_dd roe={roe_unreal:.3f}<= {thr:.3f}", exit_kind="SL")
                return

        # 1) hard max hold (ms) — last resort
        if hold_limit_ms > 0 and age_ms >= hold_limit_ms:
            self._update_pos_pred_snapshot_from_decision(pos, decision)
            self._close_position(sym, price, "hold timeout", exit_kind="TIMEOUT")
            return

        # Below: "exit like entry" — requires policy-rollforward meta present
        meta = (decision or {}).get("meta") or {}
        if not isinstance(meta, dict) or not meta:
            return

        ev_floor = _f(meta.get("ev_entry_threshold"), 0.0) or 0.0
        win_floor = _f(meta.get("win_entry_threshold"), 0.55) or 0.55
        if str(getattr(self, "_exit_unified_hold_ev_floor_raw", "")).strip():
            ev_floor = _f(getattr(self, "_exit_unified_hold_ev_floor_raw", ""), ev_floor) or ev_floor
        if str(getattr(self, "_exit_unified_hold_win_floor_raw", "")).strip():
            win_floor = _f(getattr(self, "_exit_unified_hold_win_floor_raw", ""), win_floor) or win_floor

        ev_long = _f(meta.get("policy_ev_mix_long"))
        ev_short = _f(meta.get("policy_ev_mix_short"))
        win_long = _f(meta.get("policy_p_pos_mix_long"))
        win_short = _f(meta.get("policy_p_pos_mix_short"))

        if side == "LONG":
            ev_side, win_side = ev_long, win_long
            ev_other, win_other = ev_short, win_short
            other_side = "SHORT"
        else:
            ev_side, win_side = ev_short, win_short
            ev_other, win_other = ev_long, win_long
            other_side = "LONG"

        # 2) min hold: prevent instant churn after entry
        min_hold = float(getattr(self, "_exit_unified_min_hold_sec", 0.0) or 0.0)
        if age_sec < min_hold:
            return

        entry_ev_side = _f(pos.get("entry_policy_ev_side"))
        entry_win_side = _f(pos.get("entry_policy_win_side"))
        ev_drop = False
        try:
            if (entry_ev_side is not None) and (ev_side is not None):
                ev_drop = ev_side < float(entry_ev_side) - float(getattr(self, "_exit_unified_ev_drop", 0.0) or 0.0)
        except Exception:
            ev_drop = False

        hold_bad = False
        if (ev_side is not None) and (ev_side < ev_floor):
            hold_bad = True
        if (win_side is not None) and (win_side < win_floor):
            hold_bad = True
        if ev_drop:
            hold_bad = True

        # Track WAIT streak separately (WAIT itself should not auto-exit unless sustained + weak hold)
        if action == "WAIT" and hold_bad:
            pos["_exit_unified_wait_streak"] = int(pos.get("_exit_unified_wait_streak", 0) or 0) + 1
        else:
            pos["_exit_unified_wait_streak"] = 0

        if hold_bad:
            pos["_exit_unified_bad_streak"] = int(pos.get("_exit_unified_bad_streak", 0) or 0) + 1
        else:
            pos["_exit_unified_bad_streak"] = 0

        # Flip streak (exit + allow immediate re-entry on opposite side)
        flip_signal = (action in ("LONG", "SHORT")) and (action != side)
        flip_ok = False
        try:
            if (ev_other is not None) and (win_other is not None):
                flip_ok = (ev_other >= ev_floor) and (win_other >= win_floor)
                if (ev_side is not None):
                    flip_ok = flip_ok and ((ev_other - ev_side) >= float(getattr(self, "_exit_unified_flip_margin", 0.0) or 0.0))
        except Exception:
            flip_ok = False

        if flip_signal and flip_ok:
            pos["_exit_unified_flip_streak"] = int(pos.get("_exit_unified_flip_streak", 0) or 0) + 1
        else:
            pos["_exit_unified_flip_streak"] = 0

        # Act on streaks
        if int(pos.get("_exit_unified_flip_streak", 0) or 0) >= int(getattr(self, "_exit_unified_flip_streak_need", 1) or 1):
            self._update_pos_pred_snapshot_from_decision(pos, decision)
            self._close_position(
                sym,
                price,
                f"flip_to={other_side} ev_side={ev_side} ev_other={ev_other} win_side={win_side} win_other={win_other}",
                exit_kind="FLIP",
            )
            return

        # Persistent WAIT+bad => exit
        if int(pos.get("_exit_unified_wait_streak", 0) or 0) >= int(getattr(self, "_exit_unified_wait_streak_need", 1) or 1):
            self._update_pos_pred_snapshot_from_decision(pos, decision)
            self._close_position(
                sym,
                price,
                f"wait_exit ev_side={ev_side} win_side={win_side} ev_floor={ev_floor} win_floor={win_floor}",
                exit_kind="RISK",
            )
            return

        # Persistent bad-hold => exit
        if int(pos.get("_exit_unified_bad_streak", 0) or 0) >= int(getattr(self, "_exit_unified_bad_streak_need", 1) or 1):
            self._update_pos_pred_snapshot_from_decision(pos, decision)
            self._close_position(
                sym,
                price,
                f"hold_bad ev_side={ev_side} win_side={win_side} ev_floor={ev_floor} win_floor={win_floor} ev_drop={int(ev_drop)} entry_ev={entry_ev_side} entry_win={entry_win_side}",
                exit_kind="RISK",
            )
            return

    def _maybe_exit_position(self, sym: str, price: float, decision: dict, ts: int):
        pos = self.positions.get(sym)
        if not pos or price is None:
            return
        if bool(getattr(self, "_exit_unified_enabled", False)):
            self._maybe_exit_position_unified(sym, price, decision, ts)
            return
        # ✅ 스프레드 포지션은 _manage_spreads에서 별도 관리하므로 여기서는 WAIT로 인한 exit 하지 않음
        is_spread = pos.get("tag") == "spread"
        action = (decision or {}).get("action") or "WAIT"
        hold_limit_ms = pos.get("hold_limit", MAX_POSITION_HOLD_SEC * 1000)
        age_ms = ts - pos.get("entry_time", ts)
        entry = float(pos.get("entry_price", price))
        qty = float(pos.get("quantity", 0.0))
        side = pos.get("side")
        notional = float(pos.get("notional", 0.0))
        pnl_unreal = (price - entry) * qty if side == "LONG" else (entry - price) * qty
        lev_safe = float(pos.get("leverage", self.leverage) or 1.0)
        base_notional = notional / max(lev_safe, 1e-6) if notional else 0.0
        roe_unreal = pnl_unreal / base_notional if base_notional else 0.0
        exit_reasons = []
        # 스프레드 포지션이 아니고 action이 WAIT일 때만 exit
        if not is_spread and action == "WAIT":
            exit_reasons.append("engine WAIT")
        elif action in ("LONG", "SHORT") and action != pos["side"]:
            exit_reasons.append("signal flip")
        if age_ms >= hold_limit_ms:
            exit_reasons.append("hold timeout")
        # 공격적 손실 컷 (미실현 ROE 기준) - 스프레드도 적용
        if roe_unreal <= -0.02:
            exit_reasons.append("unrealized_dd")
        if exit_reasons:
            self._update_pos_pred_snapshot_from_decision(pos, decision)
            self._close_position(sym, price, ", ".join(exit_reasons))

    def _record_trade(
        self,
        ttype: str,
        sym: str,
        side: str,
        price: float,
        qty: float,
        pos: dict,
        pnl: float | None = None,
        fee: float | None = None,
        reason: str | None = None,
        realized_r: float | None = None,
        hit: int | None = None,
        **_ignored,
    ):
        ts = time.strftime("%H:%M:%S")
        # normalize numeric fields
        pnl_val = None if pnl is None else float(pnl)
        notional = pos.get("notional")
        lev = pos.get("leverage")
        base_notional = None
        if notional is not None and lev not in (None, 0):
            try:
                base_notional = float(notional) / float(max(lev, 1e-6))
            except Exception:
                base_notional = None
        roe_val = None
        if pnl_val is not None and base_notional:
            try:
                roe_val = pnl_val / base_notional
            except Exception:
                roe_val = None

        entry = {
            "time": ts,
            "type": ttype,
            "ttype": ttype,
            "symbol": sym,
            "side": side,
            "price": float(price),
            "qty": float(qty),
            "pnl": pnl_val,
            "roe": roe_val,
            "notional": notional,
            "leverage": lev,
            "fee": None if fee is None else float(fee),
            "tag": pos.get("tag"),
            "reason": reason,
            "pred_win": pos.get("pred_win"),
            "pred_ev": pos.get("pred_ev"),
            "pred_event_ev_r": pos.get("pred_event_ev_r"),
            "pred_event_p_tp": pos.get("pred_event_p_tp"),
            "pred_event_p_sl": pos.get("pred_event_p_sl"),
            "pred_policy_ev_mix_long": pos.get("pred_policy_ev_mix_long"),
            "pred_policy_ev_mix_short": pos.get("pred_policy_ev_mix_short"),
            "pred_policy_p_pos_mix_long": pos.get("pred_policy_p_pos_mix_long"),
            "pred_policy_p_pos_mix_short": pos.get("pred_policy_p_pos_mix_short"),
            "pred_policy_ev_gap": pos.get("pred_policy_ev_gap"),
            "pred_policy_p_pos_gap": pos.get("pred_policy_p_pos_gap"),
            "pred_policy_direction": pos.get("pred_policy_direction"),
            "pred_policy_signal_strength": pos.get("pred_policy_signal_strength"),
            "pred_policy_weight_peak_h": pos.get("pred_policy_weight_peak_h"),
            "pred_policy_half_life_sec": pos.get("pred_policy_half_life_sec"),
            "pred_policy_horizons": pos.get("pred_policy_horizons"),
            "pred_policy_w_h": pos.get("pred_policy_w_h"),
            "pred_policy_exit_time_mean_sec": pos.get("pred_policy_exit_time_mean_sec"),
            "pred_policy_exit_t_mean_per_h": pos.get("pred_policy_exit_t_mean_per_h"),
            "pred_policy_exit_t_p50_per_h": pos.get("pred_policy_exit_t_p50_per_h"),
            "pred_policy_exit_reason_counts_per_h": pos.get("pred_policy_exit_reason_counts_per_h"),
            "consensus_used": pos.get("consensus_used"),
            "realized_r": float(realized_r)
            if realized_r is not None
            else (pnl_val / base_notional if (pnl_val is not None and base_notional) else None),
            "hit": int(hit) if hit is not None else None,
        }
        self.trade_tape.append(entry)
        self._persist_state(force=True)

    def _consensus_used_flag(self, decision: dict) -> bool:
        meta = decision.get("meta") or {}
        if meta.get("consensus_used"):
            return True
        for d in decision.get("details", []) or []:
            m = d.get("meta") or {}
            if m.get("consensus_used"):
                return True
        return False

    def _ema_update(self, store: dict, key, x: float, half_life_sec: float, ts_ms: int) -> float:
        prev = store.get(key)
        if prev is None:
            store[key] = (float(x), ts_ms)
            return float(x)
        prev_val, prev_ts = prev
        dt_sec = max(1.0, (ts_ms - prev_ts) / 1000.0)
        alpha = 1.0 - math.exp(-math.log(2) * dt_sec / max(half_life_sec, 1e-6))
        new_val = alpha * float(x) + (1.0 - alpha) * float(prev_val)
        store[key] = (new_val, ts_ms)
        return new_val

    @staticmethod
    def _calc_rsi(closes, period: int = RSI_PERIOD) -> float | None:
        if closes is None or len(closes) <= period:
            return None
        gains = []
        losses = []
        for i in range(1, period + 1):
            delta = closes[-i] - closes[-i - 1]
            if delta >= 0:
                gains.append(delta)
            else:
                losses.append(abs(delta))
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _consensus_action(self, decision: dict, ctx: dict) -> tuple[str, float]:
        """
        여러 지표를 투표/가중합하여 방향 합의 점수와 액션을 결정.
        score > CONSENSUS_THRESHOLD => LONG, score < -CONSENSUS_THRESHOLD => SHORT, else WAIT
        """
        score = 0.0
        base_action = decision.get("action") if decision else "WAIT"
        ev = float(decision.get("ev", 0.0)) if decision else 0.0
        meta = (decision.get("meta") or {}) if decision else {}
        cvar = float(meta.get("cvar05", decision.get("cvar", 0.0)) if decision else 0.0)
        win = float(decision.get("confidence", 0.0)) if decision else 0.0
        direction = int(ctx.get("direction", 1))
        regime = str(ctx.get("regime", "chop"))
        session = str(ctx.get("session", "OFF"))
        ofi = float(ctx.get("ofi_score", 0.0))
        rsi = self._calc_rsi(ctx.get("closes"), period=RSI_PERIOD)
        closes = ctx.get("closes") or []
        sym = ctx.get("symbol")
        spread_pct = float(meta.get("spread_pct", ctx.get("spread_pct", 0.0) or 0.0))
        liq = self._liquidity_score(sym) if sym else 1.0
        p_sl = float(meta.get("event_p_sl", 0.0) or 0.0)

        if ev <= 0:
            return "WAIT", score

        # --- history update for z-scores (regime/session residual 기반) ---
        def _rz(val, hist_deque: deque | None, default_scale: float = 0.001):
            try:
                if hist_deque is None or len(hist_deque) < 5:
                    return float(val / max(default_scale, 1e-6))
                arr = np.asarray(hist_deque, dtype=np.float64)
                mean = float(arr.mean())
                std = float(arr.std())
                std = std if std > 1e-6 else default_scale
                return float((val - mean) / std)
            except Exception:
                return float(val / max(default_scale, 1e-6))

        key = (regime, session)
        ev_hist_rs = self._ev_regime_hist_rs.setdefault(key, deque(maxlen=500))
        cvar_hist_rs = self._cvar_regime_hist_rs.setdefault(key, deque(maxlen=500))
        ev_hist_rs.append(ev)
        cvar_hist_rs.append(cvar)

        ofi_hist_base = self._ofi_regime_hist.setdefault(key, deque(maxlen=500))
        ofi_hist_base.append(ofi)
        ofi_mean = float(np.mean(ofi_hist_base)) if ofi_hist_base else 0.0
        ofi_residual = ofi - ofi_mean
        ofi_resid_hist = self._ofi_resid_hist.setdefault(key, deque(maxlen=500))
        ofi_resid_hist.append(ofi_residual)

        spread_hist = self._spread_regime_hist.setdefault(key, deque(maxlen=500))
        spread_hist.append(spread_pct)
        liq_hist = self._liq_regime_hist.setdefault(key, deque(maxlen=500))
        liq_hist.append(liq)

        z_ev = _rz(ev, ev_hist_rs)
        z_cvar = _rz(-cvar, cvar_hist_rs, default_scale=0.0005)
        z_ofi = _rz(ofi_residual, ofi_resid_hist, default_scale=0.0005)
        z_spread = _rz(spread_pct, spread_hist, default_scale=0.0002)
        z_liq = _rz(liq, liq_hist, default_scale=1.0)

        bias = float(direction)

        score = (
            1.2 * z_ev
            + 0.9 * z_cvar  # -CVaR를 넣어 양수일수록 좋게
            - 0.7 * p_sl
            + 0.5 * z_ofi
            + 0.5 * bias
            + 0.3 * z_liq
            - 0.6 * z_spread
        )

        # RSI as mild tie-breaker
        if rsi is not None:
            if rsi >= RSI_LONG:
                score += 0.2
            elif rsi <= RSI_SHORT:
                score -= 0.2

        action = "WAIT"
        if score >= CONSENSUS_THRESHOLD:
            action = "LONG"
        elif score <= -CONSENSUS_THRESHOLD:
            action = "SHORT"
        else:
            action = base_action if base_action in ("LONG", "SHORT") else "WAIT"
        return action, score

    def _spread_signal(self, base: str, quote: str) -> tuple[str, str, float] | None:
        """
        단순 비율 mean-reversion 스프레드.
        ratio = base/quote. z > entry: short base / long quote, z < -entry: long base / short quote.
        """
        base_closes = list(self.data.ohlcv_buffer.get(base) or [])
        quote_closes = list(self.data.ohlcv_buffer.get(quote) or [])
        if len(base_closes) < SPREAD_LOOKBACK + 1 or len(quote_closes) < SPREAD_LOOKBACK + 1:
            return None
        ratio = [b / q for b, q in zip(base_closes[-SPREAD_LOOKBACK:], quote_closes[-SPREAD_LOOKBACK:]) if q]
        if len(ratio) < SPREAD_LOOKBACK:
            return None
        mean = sum(ratio) / len(ratio)
        var = sum((x - mean) ** 2 for x in ratio) / len(ratio)
        std = math.sqrt(max(var, 1e-9))
        latest = ratio[-1]
        z = (latest - mean) / std if std > 0 else 0.0
        if abs(z) < SPREAD_Z_ENTRY:
            return None
        if z > 0:
            return ("SHORT", "LONG", z)
        return ("LONG", "SHORT", z)

    def _manage_spreads(self, ts: int):
        """
        스프레드 진입/청산. 페어 양쪽 포지션이 모두 없을 때만 진입.
        """
        if not self.spread_enabled:
            return
        for base, quote in self.spread_pairs:
            base_px = self.data.market.get(base, {}).get("price")
            quote_px = self.data.market.get(quote, {}).get("price")
            if base_px is None or quote_px is None:
                continue
            signal = self._spread_signal(base, quote)
            base_pos = self.positions.get(base)
            quote_pos = self.positions.get(quote)

            # 청산 조건: 태그가 spread이고 z가 수렴하거나 보유 초과
            if (base_pos and base_pos.get("tag") == "spread") or (quote_pos and quote_pos.get("tag") == "spread"):
                # 재계산 z
                zinfo = self._spread_signal(base, quote)
                z_now = zinfo[2] if zinfo else 0.0
                should_exit = abs(z_now) <= SPREAD_Z_EXIT
                age_ok = False
                if base_pos:
                    age_ok = age_ok or (ts - base_pos.get("entry_time", ts) >= SPREAD_HOLD_SEC * 1000)
                if quote_pos:
                    age_ok = age_ok or (ts - quote_pos.get("entry_time", ts) >= SPREAD_HOLD_SEC * 1000)
                if should_exit or age_ok:
                    if base_pos:
                        self._close_position(base, float(base_px), f"spread exit z={z_now:.2f}")
                    if quote_pos:
                        self._close_position(quote, float(quote_px), f"spread exit z={z_now:.2f}")
                    continue

            # 진입: 두 심볼 모두 포지션 없고 스프레드 신호가 명확할 때
            if signal and not base_pos and not quote_pos:
                base_side, quote_side, z = signal
                hold_override = min(SPREAD_HOLD_SEC, MAX_POSITION_HOLD_SEC)
                ctx_spread = {"regime": "spread", "session": time_regime()}
                meta_spread = {"regime": "spread", "session": time_regime()}
                self._enter_position(
                    base, base_side, float(base_px), {"meta": meta_spread}, ts,
                    ctx=ctx_spread, size_frac_override=SPREAD_SIZE_FRAC,
                    hold_limit_override=hold_override, tag="spread"
                )
                self._enter_position(
                    quote, quote_side, float(quote_px), {"meta": meta_spread}, ts,
                    ctx=ctx_spread, size_frac_override=SPREAD_SIZE_FRAC,
                    hold_limit_override=hold_override, tag="spread"
                )
                self._log(f"[SPREAD] {base}/{quote} z={z:.2f} -> {base_side}/{quote_side}")
                self._last_actions[base] = "SPREAD_ENTER"
                self._last_actions[quote] = "SPREAD_ENTER"
    def _decide_v3(self, ctx: dict) -> dict:
        """
        v3 flow:
          1) Execution gate (spread)
          2) Alpha candidate (direction + OFI) -> choose side or WAIT
          3) MC validate at leverage=1 using EngineHub.decide(ctx)
          4) Risk-based leverage compute
          5) (optional) Re-run hub.decide with final leverage for reporting/meta
        Returns a decision dict compatible with existing downstream code.
        """

        sym = ctx.get("symbol")
        regime = str(ctx.get("regime", "chop"))
        session = str(ctx.get("session", "OFF"))

        # -------------------------
        # Stage-1: Execution gate
        # -------------------------
        spread_pct = ctx.get("spread_pct")
        if spread_pct is None:
            spread_pct = 0.0002  # fallback (2bp)
        spread_pct = float(spread_pct)

        # 추가 하드 스프레드 필터(옵션): 비용 줄이기 목적
        if SPREAD_ENTRY_MAX and spread_pct > float(SPREAD_ENTRY_MAX):
            return {
                "action": "WAIT",
                "reason": "v3_exec_gate_spread_hard",
                "ev": 0.0,
                "confidence": 0.0,
                "cvar": 0.0,
                "meta": {"regime": regime, "session": session, "spread_pct": spread_pct, "spread_entry_max": float(SPREAD_ENTRY_MAX)},
                "details": [],
            }

        # [TEST] spread_cap 비활성화 - EV 회복 테스트
        # spread_cap_map = {
        #     "bull": 0.0020,
        #     "bear": 0.0020,
        #     "chop": 0.0012,
        #     "volatile": 0.0008,
        # }
        # spread_cap = spread_cap_map.get(regime, 0.0012)
        # if spread_pct > spread_cap:
        #     return {
        #         "action": "WAIT",
        #         "reason": "v3_exec_gate_spread",
        #         "ev": 0.0,
        #         "confidence": 0.0,
        #         "cvar": 0.0,
        #         "meta": {"regime": regime, "session": session, "spread_pct": spread_pct, "spread_cap": spread_cap},
        #         "details": [],
        #     }

        # -------------------------
        # Stage-2: Alpha candidate (direction + OFI residual-ish)
        # -------------------------
        direction = float(ctx.get("direction", 1))  # +1 / -1
        ofi = float(ctx.get("ofi_score", 0.0))
        ofi_mean = self.stats.ema_update("ofi_mean_v3", (regime, session), ofi, half_life_sec=900)
        ofi_res = ofi - float(ofi_mean)

        # simple scale proxy using recent abs-mean from stats buffer if available
        # (if not enough samples, it'll behave like "raw")
        try:
            z_ofi = self.stats.robust_z("ofi_res", (regime, session), ofi_res, fallback=ofi_res / 0.001)
        except Exception:
            z_ofi = ofi_res / 0.001

        alpha_long = 0.55 * direction + 0.35 * z_ofi
        alpha_short = -0.55 * direction - 0.35 * z_ofi
        alpha_max = max(alpha_long, alpha_short)

        alpha_min_map = {"bull": 0.25, "bear": 0.25, "chop": 0.40, "volatile": 0.55}
        alpha_min = alpha_min_map.get(regime, 0.40)
        if alpha_max < alpha_min:
            return {
                "action": "WAIT",
                "reason": "v3_alpha_gate",
                "ev": 0.0,
                "confidence": 0.0,
                "cvar": 0.0,
                "meta": {
                    "regime": regime, "session": session,
                    "alpha_long": alpha_long, "alpha_short": alpha_short,
                    "alpha_min": alpha_min, "z_ofi": z_ofi,
                    "spread_pct": spread_pct,
                },
                "details": [],
            }

        alpha_side = "LONG" if alpha_long >= alpha_short else "SHORT"

        # -------------------------
        # Stage-3: MC validate at leverage=1 (using existing EngineHub)
        # -------------------------
        ctx1 = dict(ctx)
        ctx1["leverage"] = 1.0
        decision1 = self.hub.decide(ctx1)

        ev1 = float(decision1.get("ev", 0.0) or 0.0)
        win1 = float(decision1.get("confidence", 0.0) or 0.0)
        cvar1 = float(decision1.get("cvar", 0.0) or 0.0)
        meta1 = decision1.get("meta") or {}

        p_sl = float(meta1.get("event_p_sl", 0.0) or 0.0)
        event_cvar_r = meta1.get("event_cvar_r")
        event_cvar_r = float(event_cvar_r) if event_cvar_r is not None else -999.0

        # v3 lev=1 gates (완화된 버전)
        ev1_floor = {"bull": 0.0002, "bear": 0.0002, "chop": 0.0005, "volatile": 0.0008}.get(regime, 0.0005)
        win1_floor = {"bull": 0.50, "bear": 0.50, "chop": 0.52, "volatile": 0.53}.get(regime, 0.52)
        cvar1_floor = {"bull": -0.010, "bear": -0.011, "chop": -0.008, "volatile": -0.007}.get(regime, -0.010)
        psl_max = {"bull": 0.42, "bear": 0.40, "chop": 0.35, "volatile": 0.32}.get(regime, 0.40)
        event_cvar_r_floor = {"bull": -1.20, "bear": -1.15, "chop": -1.05, "volatile": -0.95}.get(regime, -1.10)

        # ✅ EV가 높은데 진입이 안 되는 이유 확인을 위한 상세 로그
        gate_failed = []
        if ev1 < ev1_floor:
            gate_failed.append(f"ev1({ev1:.6f})<ev1_floor({ev1_floor:.6f})")
        if win1 < win1_floor:
            gate_failed.append(f"win1({win1:.3f})<win1_floor({win1_floor:.3f})")
        if cvar1 < cvar1_floor:
            gate_failed.append(f"cvar1({cvar1:.6f})<cvar1_floor({cvar1_floor:.6f})")
        if p_sl > psl_max:
            gate_failed.append(f"p_sl({p_sl:.3f})>psl_max({psl_max:.3f})")
        if event_cvar_r < event_cvar_r_floor:
            gate_failed.append(f"event_cvar_r({event_cvar_r:.3f})<event_cvar_r_floor({event_cvar_r_floor:.3f})")
        
        if ev1 < ev1_floor or win1 < win1_floor or cvar1 < cvar1_floor or p_sl > psl_max or event_cvar_r < event_cvar_r_floor:
            # ✅ 상세 로그 출력
            if ev1 >= 0.004:  # EV가 0.4% 이상인데 차단된 경우만 로그
                self._log(f"[V3_MC1_GATE] {sym} BLOCKED: {', '.join(gate_failed)} | ev1={ev1:.6f} win1={win1:.3f} cvar1={cvar1:.6f} p_sl={p_sl:.3f} event_cvar_r={event_cvar_r:.3f}")
            
            # decision1 포맷 유지 + reason만 v3로 덮어쓰기
            d = dict(decision1)
            d["action"] = "WAIT"
            d["reason"] = "v3_mc1_gate"
            m = dict(meta1)
            m.update({
                "v3_alpha_side": alpha_side,
                "EV1": ev1, "Win1": win1, "CVaR1": cvar1,
                "ev1_floor": ev1_floor, "win1_floor": win1_floor, "cvar1_floor": cvar1_floor,
                "psl_max": psl_max, "event_cvar_r_floor": event_cvar_r_floor,
                "spread_pct": spread_pct,
            })
            d["meta"] = m
            return d

        # -------------------------
        # Stage-4: Risk-based leverage (use lev=1 metrics)
        # -------------------------
        sl_pct = float(meta1.get("mc_sl") or meta1.get("sl_pct") or 0.002)
        event_cvar_pct = event_cvar_r * sl_pct

        # best-effort slippage from meta if present
        slippage_pct = float(meta1.get("slippage_pct", 0.0) or 0.0)
        sigma = float(ctx.get("sigma", 0.0) or 0.0)

        risk = max(abs(cvar1), abs(event_cvar_pct)) + 0.7 * spread_pct + 0.5 * slippage_pct + 0.5 * sigma
        if risk < 1e-9:
            risk = 1e-9

        # leverage bounds per regime
        lev_max = {"bull": 20.0, "bear": 18.0, "chop": 10.0, "volatile": 8.0}.get(regime, 10.0)
        lev_raw = (max(ev1, 0.0) / risk) * K_LEV
        lev = float(max(1.0, min(lev_max, lev_raw)))

        # -------------------------
        # Stage-5: Re-run hub.decide with final leverage (reporting & size_frac)
        # -------------------------
        ctxF = dict(ctx)
        ctxF["leverage"] = lev
        decisionF = self.hub.decide(ctxF)

        # enforce alpha side: if hub returns opposite, prefer alpha side unless you want flip logic
        decisionF = dict(decisionF)
        decisionF["action"] = alpha_side if decisionF.get("action") in ("LONG", "SHORT") else "WAIT"

        metaF = dict(decisionF.get("meta") or {})
        metaF.update({
            "v3_used": True,
            "v3_alpha_side": alpha_side,
            "EV1": ev1, "Win1": win1, "CVaR1": cvar1,
            "risk": risk, "lev_raw": lev_raw, "lev": lev,
            "event_cvar_pct": event_cvar_pct,
            "spread_pct": spread_pct,
        })
        decisionF["meta"] = metaF

        # expose leverage/size_frac for your existing sizing code
        decisionF["leverage"] = lev

        # size cap by regime (your code already applies regime_cap_frac later, but we can prefill)
        cap_map = {"bull": 0.25, "bear": 0.22, "chop": 0.10, "volatile": 0.08}
        metaF.setdefault("regime_cap_frac", cap_map.get(regime, 0.10))

        return decisionF

    def _mark_exit_and_cooldown(self, sym: str, exit_kind: str, ts_ms: int):
        """
        exit_kind: "TP" | "TIMEOUT" | "SL" | "KILL" | "MANUAL"
        """
        k = (exit_kind or "MANUAL").upper()
        self._last_exit_kind[sym] = k
        if k in ("FLIP",):
            # allow immediate re-entry on the opposite side
            self._cooldown_until[sym] = ts_ms
        elif k in ("SL", "KILL"):
            self._cooldown_until[sym] = ts_ms + int(COOLDOWN_RISK_SEC * 1000)
        else:
            self._cooldown_until[sym] = ts_ms + int(COOLDOWN_TP_SEC * 1000)
        self._streak[sym] = 0




    async def decision_loop(self):
        # ✅ 마지막 실행 시간 추적 (과도한 실행 방지)
        last_decision_run_ms = 0
        # ✅ 더 빠른 반응을 위해 최소 간격을 줄임 (100ms 또는 DECISION_REFRESH_SEC의 10%)
        min_decision_interval_ms = int(max(100, float(DECISION_REFRESH_SEC) * 1000 * 0.1))  # 최소 간격: DECISION_REFRESH_SEC의 10% (기존 25%에서 감소)
        max_wait_sec = max(1.0, float(DECISION_REFRESH_SEC))  # 최대 대기 시간: DECISION_REFRESH_SEC (기존 2.0에서 감소)
        
        print(f"[EV_DEBUG] decision_loop: Started! min_decision_interval_ms={min_decision_interval_ms} max_wait_sec={max_wait_sec}")
        self._log(f"[EV_DEBUG] decision_loop: Started!")
        
        while True:
            print(f"[EV_DEBUG] decision_loop: Loop iteration start")
            # Probe-only mode: keep the event loop responsive so the probe task can run.
            self.pmaker.start_probe()
            # ✅ train_pmaker_only 모드에서도 decision은 실행되어야 함 (EV 값 표시를 위해)
            # if self.pmaker.train_only and self.pmaker.train_only_strict:
            #     print(f"[EV_DEBUG] decision_loop: train_pmaker_only mode, sleeping...")
            #     await asyncio.sleep(max(0.1, float(self.pmaker.probe_interval_sec)))
            #     continue

            # ✅ Producer-Consumer 패턴: 데이터 갱신을 기다림 (최대 대기 시간 설정)
            if LOG_STDOUT:
                print(f"[EV_DEBUG] decision_loop: Waiting for data_updated_event (timeout={max_wait_sec}s)...")
            try:
                await asyncio.wait_for(self.data.data_updated_event.wait(), timeout=max_wait_sec)
                event_received = True
                if LOG_STDOUT:
                    print(f"[EV_DEBUG] decision_loop: Event received!")
            except asyncio.TimeoutError:
                # 타임아웃 발생 시 강제로 실행 (데이터가 없어도 주기적으로 체크)
                event_received = False
                if LOG_STDOUT:
                    print(f"[EV_DEBUG] decision_loop: Timeout waiting for event, forcing execution...")
            
            # ✅ 최소 간격 체크: 너무 자주 실행되지 않도록
            now = now_ms()
            if now - last_decision_run_ms < min_decision_interval_ms:
                # 이벤트는 clear하지 않고 잠시 대기 (다음 갱신을 기다림)
                if event_received:
                    await asyncio.sleep(0.1)
                continue
            
            last_decision_run_ms = now
            if event_received:
                self.data.data_updated_event.clear()
            if LOG_STDOUT:
                print(f"[EV_DEBUG] decision_loop: Processing decision for all symbols... (event_received={event_received})")

            # [D] Train alpha a bit each loop (GPU)
            if self.alpha_trainer:
                alpha_train_stats = self.alpha_trainer.train_tick()
                if alpha_train_stats.get("loss") is not None:
                    self._log(f"[ALPHA_HIT] loss={alpha_train_stats['loss']:.4f} buf={alpha_train_stats['buffer_n']}")
            
            # [PMAKER] 백그라운드에서 지속적으로 pmaker 모델 학습 (S 데이터 누적 → 알파값 개선)
            if self.pmaker.surv is not None and self.pmaker.enabled and self.pmaker.train_steps > 0:
                try:
                    replay_size = len(self.pmaker.surv.replay)
                    if replay_size > 0:
                        self.pmaker.surv.train_from_replay(steps=self.pmaker.train_steps, batch_size=self.pmaker.batch)
                        # 학습된 결과가 sym_n, sym_wins에 누적되어 S로 계산되고, 다음 예측에 반영됨
                        self._log(f"[PMAKER_BG_TRAIN] replay={replay_size} steps={self.pmaker.train_steps} batch={self.pmaker.batch}")
                except Exception as e:
                    self._log_err(f"[PMAKER_BG_TRAIN] error: {e}")

            if self.pmaker.surv is not None and self.pmaker.enabled and self._pmaker_save_interval_ms > 0:
                if self._pmaker_dirty and (now - self._pmaker_last_save_ms >= self._pmaker_save_interval_ms):
                    if self.pmaker.save_model():
                        self._pmaker_last_save_ms = now
                        self._pmaker_dirty = False
                        self._log(f"[PMAKER_SAVE] saved to {self.pmaker.model_path}")

            loop_t0 = time.time()
            rows = []
            ts = now_ms()
            self._decision_cycle = (self._decision_cycle + 1) % self._decision_log_every
            log_this_cycle = (self._decision_cycle == 0)
            
            # ✅ 모든 심볼의 분석을 병렬로 처리
            # Task 리스트 생성: 각 심볼에 대해 analyze_symbol 코루틴 생성
            tasks = [self.analyze_symbol(sym, ts, log_this_cycle) for sym in SYMBOLS]
            
            # Asyncio Gather 사용: 모든 심볼의 MC 시뮬레이션을 동시에 실행
            # return_exceptions=True로 설정하여 한 심볼에서 에러가 나도 나머지는 멈추지 않게 함
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리: 반환된 results를 순회하며 각 심볼의 결과를 rows에 추가
            for idx, result in enumerate(results):
                sym = SYMBOLS[idx]
                if isinstance(result, Exception):
                    # 에러가 발생한 경우 로그를 남기고 기본 row 추가
                    import traceback
                    self._log_err(f"[ERR] analyze_symbol {sym}: {result} {traceback.format_exc()}")
                    rows.append(self._row(sym, self.data.market[sym].get("price"), ts, None, len(list(self.data.ohlcv_buffer[sym]))))
                else:
                    # 정상적으로 완료된 경우 결과를 rows에 추가
                    rows.append(result)
            
            # 기존 순차 처리 루프는 analyze_symbol 함수로 이동하여 병렬 처리됨

            try:
                if self.spread_enabled:
                    self._manage_spreads(ts)
                # ✅ rows가 비어있어도 broadcast 호출 (최소한 빈 데이터라도 전송)
                if not rows:
                    # rows가 비어있으면 모든 심볼에 대해 기본 row 생성
                    for sym in SYMBOLS:
                        price = self.data.market[sym].get("price")
                        closes = list(self.data.ohlcv_buffer[sym])
                        candles = len(closes)
                        rows.append(self._row(sym, price, ts, None, candles))
                # ✅ rows 상태 확인 로그 (너무 잦으면 stdout이 병목이 될 수 있어 throttling)
                if (ts - int(self._last_decision_loop_log_ms)) >= 5000:
                    self._last_decision_loop_log_ms = ts
                    self._log(f"[DECISION_LOOP] completed, rows_count={len(rows)}, broadcasting...")
                # 마지막 계산 결과를 저장해 신규 WS 연결 시 즉시 스냅샷 제공
                self._last_rows = rows
            except Exception as e2:
                import traceback
                self._log_err(f"[ERR] broadcast: {e2} {traceback.format_exc()}")

            # loop timing (dashboard engine.loop_ms)
            loop_t1 = time.time()
            loop_ms = int((loop_t1 - loop_t0) * 1000.0)
            try:
                self._loop_ms = loop_ms
                logger.info(f"[PERF] decision_loop: total_ms={loop_ms} (analyze_gather={int((loop_t1 - loop_t0)*1000)}ms)")
            except Exception:
                pass
            except Exception:
                pass

            # ✅ 이벤트 기반으로 동작하므로 sleep 제거 (orderbook 갱신 시에만 실행됨)




async def main():
    # Optional .env loader (no deps). Does not override existing env vars.
    env_file = str(os.environ.get("BYBIT_ENV_FILE", "")).strip()
    loaded_env_paths: list[str] = []
    if env_file:
        if _load_env_file(env_file):
            loaded_env_paths.append(env_file)
    else:
        # fallbacks (checked in order)
        loaded = _load_env_file(str((BASE_DIR / "state" / "bybit.env")))
        if loaded:
            loaded_env_paths.append(str((BASE_DIR / "state" / "bybit.env")))
        # Always load example defaults after bybit.env (fills missing vars like BYBIT_TESTNET without overriding).
        if _load_env_file(str((BASE_DIR / "state" / "bybit.env.example"))):
            loaded_env_paths.append(str((BASE_DIR / "state" / "bybit.env.example")))
        if _load_env_file(str((BASE_DIR / ".env"))):
            loaded_env_paths.append(str((BASE_DIR / ".env")))

    if loaded_env_paths:
        print("[ENV] loaded:", ", ".join(loaded_env_paths))

    cfg = {
        "enableRateLimit": True,
        "timeout": CCXT_TIMEOUT_MS,
    }
    api_key = str(os.environ.get("BYBIT_API_KEY", "")).strip()
    api_secret = str(os.environ.get("BYBIT_API_SECRET", "")).strip()
    if api_key and api_secret:
        cfg.update({"apiKey": api_key, "secret": api_secret})
        masked = api_key[:4] + "…" + api_key[-4:] if len(api_key) >= 8 else (api_key[:2] + "…")
        print(f"[BYBIT] apiKey loaded: {masked}")
        if not str(os.environ.get("BYBIT_TESTNET", "")).strip() and not str(os.environ.get("CCXT_SANDBOX", "")).strip():
            print("[BYBIT] NOTE: set BYBIT_TESTNET=1 for testnet keys (add it to your env file to avoid API errors).")
    exchange = ccxt.bybit(cfg)
    # If you use testnet keys, you must enable sandbox mode.
    if _env_bool("BYBIT_TESTNET", False) or _env_bool("CCXT_SANDBOX", False):
        try:
            exchange.set_sandbox_mode(True)
            print("[BYBIT] sandbox_mode enabled (testnet)")
        except Exception as e:
            print(f"[BYBIT] sandbox_mode enable failed: {e}")

    # Load markets once and drop symbols that don't exist (common on testnet).
    try:
        await exchange.load_markets()
        global SYMBOLS
        missing = [s for s in SYMBOLS if s not in exchange.markets]
        if missing:
            print(f"[BYBIT] dropping unsupported symbols: {missing}")
        SYMBOLS = [s for s in SYMBOLS if s in exchange.markets]
        if not SYMBOLS:
            raise RuntimeError("No supported symbols after filtering; set SYMBOLS_CSV to valid Bybit markets.")
    except Exception as e:
        print(f"[BYBIT] load_markets/filter failed: {e}")
    orchestrator = LiveOrchestrator(exchange, SYMBOLS)

    runner = None
    site = None
    try:
        dashboard = DashboardServer(orchestrator)
        orchestrator.dashboard = dashboard
        await dashboard.start()

        # ✅ OHLCV preload를 먼저 완료한 후 decision_loop 시작 (데이터 없이 실행 방지)
        if PRELOAD_ON_START:
            try:
                print(f"⏳ Preloading OHLCV (limit={OHLCV_PRELOAD_LIMIT})...")
                await orchestrator.data.preload_all_ohlcv(limit=OHLCV_PRELOAD_LIMIT)
                orchestrator._persist_state(force=True)
                print("✅ OHLCV preload done")
            except Exception as e:
                print(f"[ERR] preload task failed: {e}")

        # ✅ 초기 이벤트 설정: decision_loop가 실행될 수 있도록 (preload 완료 후)
        orchestrator.data.data_updated_event.set()

        asyncio.create_task(orchestrator.data.fetch_prices_loop())
        asyncio.create_task(orchestrator.data.fetch_ohlcv_loop())
        asyncio.create_task(orchestrator.data.fetch_orderbook_loop())
        asyncio.create_task(orchestrator.decision_loop())
        asyncio.create_task(orchestrator.broadcast_loop())
        


        print(f"🚀 Dashboard: http://localhost:{PORT}")
        print(f"📄 Serving: {DASHBOARD_FILE.name}")
        await asyncio.Future()
    except OSError as e:
        print(f"[ERR] Failed to bind on port {PORT}: {e}")
    finally:
        # ✅ 엔진 종료 시 PMaker 모델 저장 (finally 블록에서도 저장)
        if 'orchestrator' in locals() and orchestrator.pmaker.surv is not None and orchestrator.pmaker.enabled:
            try:
                orchestrator.pmaker.surv.save(orchestrator.pmaker.model_path)
                print(f"[SHUTDOWN] PMaker model saved to {orchestrator.pmaker.model_path}")
            except Exception as e:
                print(f"[SHUTDOWN] Failed to save PMaker model: {e}")
        
        try:
            await exchange.close()
        except Exception:
            pass
        if site is not None:
            try:
                await site.stop()
            except Exception:
                pass
        if runner is not None:
            try:
                await runner.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
1
