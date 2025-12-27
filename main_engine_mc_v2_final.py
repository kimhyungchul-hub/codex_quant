import asyncio
import os
import json
import numpy as np
import ccxt.async_support as ccxt
from collections import deque, defaultdict
from typing import Optional, Tuple, List, Dict, Any
from core.dashboard_server import DashboardServer
from core.data_manager import DataManager
from engines.engine_hub import EngineHub
from engines.mc_engine import MonteCarloEngine
from engines.mc_risk import kelly_fraction
from engines.pmaker_manager import PMakerManager
from engines.running_stats import RunningStats
from utils.alpha_features import build_alpha_features
from utils.helpers import _env_bool, _env_float, _env_int, now_ms

class LiveOrchestrator:
    def __init__(self, exchange, symbols):
        self.exchange = exchange
        self.symbols = symbols
        self.data = DataManager(exchange, symbols)
        self.engine_hub = EngineHub()
        self.mc_engine = MonteCarloEngine()
        self.pmaker = PMakerManager()
        self.running_stats = RunningStats()
        self.state_dir = pathlib.Path('state')
        self.state_dir.mkdir(exist_ok=True)
        self._load_persistent_state()

    async def _compute_atr_proxy(self, symbol: str, period: int = 14) -> float:
        # Implementation of _compute_atr_proxy
        pass

    async def _exec_stats_for(self, symbol: str) -> Dict[str, Any]:
        # Implementation of _exec_stats_for
        pass

    async def _rows_snapshot(self) -> List[Dict[str, Any]]:
        # Implementation of _rows_snapshot
        pass

    async def broadcast_loop(self):
        # Implementation of broadcast_loop
        pass

    async def analyze_symbol(self, symbol: str):
        # Implementation of analyze_symbol
        pass

    async def _compute_decision_task(self, symbol: str):
        # Implementation of _compute_decision_task
        pass

    async def _log(self, message: str):
        # Implementation of _log
        pass

    async def _log_err(self, message: str):
        # Implementation of _log_err
        pass

    async def _pmaker_status(self):
        # Implementation of _pmaker_status
        pass

    async def _ccxt_call(self, method: str, *args, **kwargs):
        # Implementation of _ccxt_call
        pass

    async def _load_json(self, filename: str) -> Dict[str, Any]:
        # Implementation of _load_json
        pass

    async def _load_persistent_state(self):
        # Implementation of _load_persistent_state
        pass

    async def _persist_state(self, force: bool = False):
        # Implementation of _persist_state
        pass

    async def _compute_returns_and_vol(self, symbol: str) -> Tuple[float, float]:
        # Implementation of _compute_returns_and_vol
        pass

    async def _annualize_mu_sigma(self, mu: float, sigma: float, period: int = 252) -> Tuple[float, float]:
        # Implementation of _annualize_mu_sigma
        pass

    async def _safe_float(self, value: Any) -> float:
        # Implementation of _safe_float
        pass

    async def _sanitize_for_json(self, value: Any) -> Any:
        # Implementation of _sanitize_for_json
        pass

    async def _row(self, symbol: str) -> Dict[str, Any]:
        # Implementation of _row
        pass

    async def _total_open_notional(self) -> float:
        # Implementation of _total_open_notional
        pass

    async def _can_enter_position(self, symbol: str) -> bool:
        # Implementation of _can_enter_position
        pass

    async def _entry_permit(self, symbol: str) -> bool:
        # Implementation of _entry_permit
        pass

    async def _calc_position_size(self, symbol: str) -> float:
        # Implementation of _calc_position_size
        pass

    async def _dynamic_leverage_risk(self, symbol: str) -> float:
        # Implementation of _dynamic_leverage_risk
        pass

    async def _direction_bias(self, symbol: str) -> float:
        # Implementation of _direction_bias
        pass

    async def _infer_regime(self, symbol: str) -> str:
        # Implementation of _infer_regime
        pass

    async def _compute_ofi_score(self, symbol: str) -> float:
        # Implementation of _compute_ofi_score
        pass

    async def _liquidity_score(self, symbol: str) -> float:
        # Implementation of _liquidity_score
        pass

    async def _enter_position(self, symbol: str):
        # Implementation of _enter_position
        pass

    async def _close_position(self, symbol: str):
        # Implementation of _close_position
        pass

    async def _pmaker_update_attempt(self, symbol: str):
        # Implementation of _pmaker_update_attempt
        pass

    async def _maybe_place_order(self, symbol: str):
        # Implementation of _maybe_place_order
        pass

    async def _execute_order(self, symbol: str):
        # Implementation of _execute_order
        pass

    async def _execute_close_order(self, symbol: str):
        # Implementation of _execute_close_order
        pass

    async def _best_bid_ask(self, symbol: str) -> Tuple[float, float]:
        # Implementation of _best_bid_ask
        pass

    async def _infer_tick_size(self, symbol: str) -> float:
        # Implementation of _infer_tick_size
        pass

    async def _infer_tick_size_from_ladders(self, symbol: str) -> float:
        # Implementation of _infer_tick_size_from_ladders
        pass

    async def _round_price_to_tick(self, price: float, tick_size: float) -> float:
        # Implementation of _round_price_to_tick
        pass

    async def _extract_order_stats(self, order: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation of _extract_order_stats
        pass

    async def _fetch_order_trades_any(self, order_id: str) -> List[Dict[str, Any]]:
        # Implementation of _fetch_order_trades_any
        pass

    async def _pmaker_probe_price(self, symbol: str) -> float:
        # Implementation of _pmaker_probe_price
        pass

    async def _rebalance_position(self, symbol: str):
        # Implementation of _rebalance_position
        pass

    async def _update_pos_pred_snapshot_from_decision(self, symbol: str, decision: Dict[str, Any]):
        # Implementation of _update_pos_pred_snapshot_from_decision
        pass

    async def _maybe_exit_position_unified(self, symbol: str):
        # Implementation of _maybe_exit_position_unified
        pass

    async def _maybe_exit_position(self, symbol: str):
        # Implementation of _maybe_exit_position
        pass

    async def _record_trade(self, trade: Dict[str, Any]):
        # Implementation of _record_trade
        pass

    async def _consensus_used_flag(self, symbol: str) -> bool:
        # Implementation of _consensus_used_flag
        pass

    async def _ema_update(self, symbol: str, value: float):
        # Implementation of _ema_update
        pass

    async def _calc_rsi(self, symbol: str) -> float:
        # Implementation of _calc_rsi
        pass

    async def _consensus_action(self, symbol: str) -> str:
        # Implementation of _consensus_action
        pass

    async def _spread_signal(self, symbol: str) -> float:
        # Implementation of _spread_signal
        pass

    async def _manage_spreads(self, symbol: str):
        # Implementation of _manage_spreads
        pass

    async def _decide_v3(self, symbol: str) -> Dict[str, Any]:
        # Implementation of _decide_v3
        pass

    async def _mark_exit_and_cooldown(self, symbol: str):
        # Implementation of _mark_exit_and_cooldown
        pass

    async def decision_loop(self):
        # Implementation of decision_loop
        pass

async def main():
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
