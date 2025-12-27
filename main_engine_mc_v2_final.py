import asyncio
import os
import traceback
from collections import deque
from typing import Any, Dict, List, Optional

from aiohttp.tcp_helpers import _tcp_keepalive_safe
from ccxt.async_support import Exchange
from config import SYMBOLS, PRELOAD_ON_START, OHLCV_PRELOAD_LIMIT, PORT, DASHBOARD_FILE
from core.dashboard_server import DashboardServer
from core.data_manager import DataManager
from engines.engine_hub import EngineHub
from engines.mc_engine import MonteCarloEngine
from engines.pmaker_manager import PMakerManager
from engines.running_stats import RunningStats
from utils.alpha_features import build_alpha_features
from utils.helpers import _sanitize_for_json

class LiveOrchestrator:
    def __init__(self, exchange: Exchange, symbols: Optional[List[str]] = None):
        self.hub = EngineHub()
        self.mc_engine_by_symbol = {}
        self.alpha_enable = _env_bool("ALPHA_ENABLE", False)
        if self.alpha_enable:
            from trainers.online_alpha_trainer import OnlineAlphaTrainer, AlphaTrainerConfig
            self.alpha_trainer = OnlineAlphaTrainer(AlphaTrainerConfig())
        self.data = DataManager(exchange, symbols)
        self.pmaker = PMakerManager()
        self.state_dir = pathlib.Path("state")
        self.state_dir.mkdir(exist_ok=True)
        self._load_persistent_state()
        self._exec_stats = defaultdict(lambda: deque(maxlen=1000))
        self._decision_cache = {}
        self._load_json("config.json")

    def _load_json(self, filename: str) -> Dict[str, Any]:
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _load_persistent_state(self) -> None:
        try:
            with open(self.state_dir / "state.json", "r") as f:
                state = json.load(f)
                self.__dict__.update(state)
        except FileNotFoundError:
            pass

    def _persist_state(self, force: bool = False) -> None:
        if force or len(self._exec_stats) > 100:
            with open(self.state_dir / "state.json", "w") as f:
                json.dump(self.__dict__, f)

    async def _compute_atr_proxy(self, symbol: str) -> float:
        # Implementation of _compute_atr_proxy
        pass

    async def _exec_stats_for(self, symbol: str) -> Dict[str, Any]:
        # Implementation of _exec_stats_for
        pass

    async def _rows_snapshot(self) -> List[Dict[str, Any]]:
        # Implementation of _rows_snapshot
        pass

    async def broadcast_loop(self) -> None:
        # Implementation of broadcast_loop
        pass

    async def analyze_symbol(self, symbol: str) -> None:
        # Implementation of analyze_symbol
        pass

    async def _compute_decision_task(self, symbol: str) -> None:
        # Implementation of _compute_decision_task
        pass

    async def _log(self, message: str) -> None:
        # Implementation of _log
        pass

    async def _log_err(self, message: str) -> None:
        # Implementation of _log_err
        pass

    async def _pmaker_status(self) -> None:
        # Implementation of _pmaker_status
        pass

    async def _ccxt_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        # Implementation of _ccxt_call
        pass

    async def _load_json(self, filename: str) -> Dict[str, Any]:
        # Implementation of _load_json
        pass

    async def _load_persistent_state(self) -> None:
        # Implementation of _load_persistent_state
        pass

    async def _persist_state(self, force: bool = False) -> None:
        # Implementation of _persist_state
        pass

    async def _compute_returns_and_vol(self, symbol: str) -> None:
        # Implementation of _compute_returns_and_vol
        pass

    async def _annualize_mu_sigma(self, mu: float, sigma: float) -> None:
        # Implementation of _annualize_mu_sigma
        pass

    async def _safe_float(self, value: Any) -> float:
        # Implementation of _safe_float
        pass

    async def _sanitize_for_json(self, obj: Any, _depth: int = 0) -> Any:
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

    async def _enter_position(self, symbol: str) -> None:
        # Implementation of _enter_position
        pass

    async def _close_position(self, symbol: str) -> None:
        # Implementation of _close_position
        pass

    async def _pmaker_update_attempt(self, symbol: str) -> None:
        # Implementation of _pmaker_update_attempt
        pass

    async def _maybe_place_order(self, symbol: str) -> None:
        # Implementation of _maybe_place_order
        pass

    async def _execute_order(self, symbol: str) -> None:
        # Implementation of _execute_order
        pass

    async def _execute_close_order(self, symbol: str) -> None:
        # Implementation of _execute_close_order
        pass

    async def _best_bid_ask(self, symbol: str) -> Dict[str, float]:
        # Implementation of _best_bid_ask
        pass

    async def _infer_tick_size(self, symbol: str) -> float:
        # Implementation of _infer_tick_size
        pass

    async def _infer_tick_size_from_ladders(self, symbol: str) -> float:
        # Implementation of _infer_tick_size_from_ladders
        pass

    async def _round_price_to_tick(self, symbol: str, price: float) -> float:
        # Implementation of _round_price_to_tick
        pass

    async def _extract_order_stats(self, symbol: str) -> Dict[str, Any]:
        # Implementation of _extract_order_stats
        pass

    async def _fetch_order_trades_any(self, symbol: str) -> List[Dict[str, Any]]:
        # Implementation of _fetch_order_trades_any
        pass

    async def _pmaker_probe_price(self, symbol: str) -> float:
        # Implementation of _pmaker_probe_price
        pass

    async def _rebalance_position(self, symbol: str) -> None:
        # Implementation of _rebalance_position
        pass

    async def _update_pos_pred_snapshot_from_decision(self, symbol: str) -> None:
        # Implementation of _update_pos_pred_snapshot_from_decision
        pass

    async def _maybe_exit_position_unified(self, symbol: str) -> None:
        # Implementation of _maybe_exit_position_unified
        pass

    async def _maybe_exit_position(self, symbol: str) -> None:
        # Implementation of _maybe_exit_position
        pass

    async def _record_trade(self, symbol: str) -> None:
        # Implementation of _record_trade
        pass

    async def _consensus_used_flag(self, symbol: str) -> bool:
        # Implementation of _consensus_used_flag
        pass

    async def _ema_update(self, symbol: str) -> None:
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

    async def _manage_spreads(self, symbol: str) -> None:
        # Implementation of _manage_spreads
        pass

    async def _decide_v3(self, symbol: str) -> str:
        # Implementation of _decide_v3
        pass

    async def _mark_exit_and_cooldown(self, symbol: str) -> None:
        # Implementation of _mark_exit_and_cooldown
        pass

    async def decision_loop(self) -> None:
        # Implementation of decision_loop
        pass

async def main() -> None:
    exchange = ccxt.async_support.bybit()
    if _env_bool("BYBIT_TESTNET", False) or _env_bool("CCXT_SANDBOX", False):
        try:
            exchange.set_sandbox_mode(True)
            print("[BYBIT] sandbox_mode enabled (testnet)")
        except Exception as e:
            print(f"[BYBIT] sandbox_mode enable failed: {e}")

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

        if PRELOAD_ON_START:
            try:
                print(f"⏳ Preloading OHLCV (limit={OHLCV_PRELOAD_LIMIT})...")
                await orchestrator.data.preload_all_ohlcv(limit=OHLCV_PRELOAD_LIMIT)
                orchestrator._persist_state(force=True)
                print("✅ OHLCV preload done")
            except Exception as e:
                print(f"[ERR] preload task failed: {e}")

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
