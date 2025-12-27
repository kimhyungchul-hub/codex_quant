from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from core.bus import AsyncEventBus, EventSink
from core.events import BarEvent, MarketEvent
from data.candles import CandleBuilder
from data.feeds.bybit_api import BybitCredentials, BybitRESTClient, BybitWebSocketClient
from data.feeds.bybit_ws import SimulatedFeedConfig, simulated_feed
from execution.live import LiveExecutor
from execution.paper import FillParams, PaperExecutor
from features.feature_engine import FeatureEngine
from ops.logging import setup_logging
from portfolio.positions import Portfolio
from regime.regime_engine import RegimeEngine
from risk.risk_manager import RiskLimits, check_order
from strategies.base import MomentumBreakoutStrategy, StrategyContext


async def handle_events(
    bus: AsyncEventBus,
    executor,
    portfolio: Portfolio,
    strategy_cfg: Dict[str, float],
    limits: RiskLimits,
    logger,
    candle_builder: CandleBuilder,
    feature_engine: FeatureEngine,
    regime_engine: RegimeEngine,
) -> None:
    queue = bus.subscribe()
    strategy = MomentumBreakoutStrategy(
        lookback=int(strategy_cfg.get("lookback", 5)),
        threshold=float(strategy_cfg.get("threshold", 0.001)),
    )
    async for event in bus.iterate(queue):
        bars = candle_builder.update(event)
        if not bars and isinstance(event, (BarEvent, MarketEvent)):
            bars = [event] if isinstance(event, BarEvent) else []
        if not bars:
            logger.debug("Unhandled event %s", event)
            continue

        for bar in bars:
            features = feature_engine.update(bar)
            regime = regime_engine.evaluate(features)
            ctx = StrategyContext(
                symbol=bar.symbol,
                position=portfolio.get_position(bar.symbol).quantity,
                avg_price=portfolio.get_position(bar.symbol).avg_price,
                config=strategy_cfg,
                features=features,
                regime=regime,
            )
            order = strategy.on_bar(bar, ctx)
            if order:
                order.payload.setdefault("last_price", bar.close)
                risk_mult = regime_engine.risk_multiplier(regime)
                if check_order(order, bar.close, limits, risk_multiplier=risk_mult):
                    result = await executor.submit(order)
                    if result.fill:
                        logger.info(
                            "fill %s qty=%s price=%s fee=%s pos=%s regime=%s",
                            result.fill.symbol,
                            result.fill.quantity,
                            result.fill.price,
                            result.fill.fee,
                            portfolio.get_position(bar.symbol).quantity,
                            regime,
                        )
                else:
                    logger.warning(
                        "risk rejected order %s regime=%s mult=%.2f",
                        order.client_order_id,
                        regime,
                        risk_mult,
                    )


async def run_paper(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text())
    logger = setup_logging(cfg.get("log_level", "INFO"))
    logger.info(
        "CLI paper/live 파이프라인은 웹소켓 대시보드를 띄우지 않습니다. 실시간 뷰가 필요하면 main_engine_mc_v2_final.py를 실행해 /ws를 제공하세요."
    )
    bus = AsyncEventBus()
    sink = EventSink(bus, symbols=cfg["symbols"])
    portfolio = Portfolio()
    executor = PaperExecutor(
        portfolio,
        params=FillParams(
            fee_rate=float(cfg["execution"].get("fee_rate", 0.0006)),
            slippage_bps=float(cfg["execution"].get("slippage_bps", 5.0)),
            latency_ms=int(cfg["execution"].get("latency_ms", 50)),
        ),
    )
    feed_cfg = SimulatedFeedConfig(
        symbols=list(cfg["symbols"]),
        interval_seconds=int(cfg["feed"].get("interval_seconds", 1)),
        seed=int(cfg["feed"].get("seed", 42)),
        drift=float(cfg["feed"].get("drift", 0.0)),
        volatility=float(cfg["feed"].get("volatility", 0.001)),
    )
    limits = RiskLimits(
        max_position=float(cfg["risk"].get("max_position", 0.01)),
        max_order_notional=float(cfg["risk"].get("max_order_notional", 1000.0)),
    )
    candle_builder = CandleBuilder(
        interval_seconds=int(cfg["feed"].get("interval_seconds", 1))
    )
    feature_engine = FeatureEngine(
        window=int(cfg.get("feature_engine", {}).get("window", 20))
    )
    regime_engine = RegimeEngine(
        vol_threshold=float(cfg.get("regime", {}).get("vol_threshold", 0.003)),
        trend_threshold=float(cfg.get("regime", {}).get("trend_threshold", 0.001)),
    )

    consumer = asyncio.create_task(
        handle_events(
            bus,
            executor,
            portfolio,
            cfg["strategy"],
            limits,
            logger,
            candle_builder,
            feature_engine,
            regime_engine,
        )
    )
    feed = asyncio.create_task(simulated_feed(sink, feed_cfg))
    try:
        await asyncio.sleep(int(cfg.get("run_seconds", 10)))
    finally:
        await bus.close()
        feed.cancel()
        consumer.cancel()
        await executor.close()
        logger.info(
            "portfolio state: %s",
            json.dumps({k: v.__dict__ for k, v in portfolio.positions.items()}),
        )


async def live_ws_to_bus(
    ws_client: BybitWebSocketClient, sink: EventSink, symbols: list[str], logger
) -> None:
    async for trade in ws_client.stream_public_trades(symbols):
        symbol = trade.get("symbol")
        price = float(trade.get("price", 0.0))
        size = float(trade.get("size", 0.0))
        ts_ms = int(trade.get("ts", 0))
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        event = MarketEvent(
            symbol=symbol, price=price, size=size, payload={"source": "bybit_ws"}
        )
        event.ts = ts
        if sink.accepts(symbol):
            await sink.emit(event)
        else:
            logger.debug("Dropping symbol %s not in sink", symbol)


async def run_live(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text())
    logger = setup_logging(cfg.get("log_level", "INFO"))
    bus = AsyncEventBus()
    sink = EventSink(bus, symbols=cfg["symbols"])
    portfolio = Portfolio()
    creds = BybitCredentials(
        api_key=cfg["execution"]["api_key"],
        api_secret=cfg["execution"]["api_secret"],
        testnet=cfg["execution"].get("testnet", True),
    )
    rest_client = BybitRESTClient(creds)
    executor = LiveExecutor(portfolio, rest_client)
    candle_builder = CandleBuilder(
        interval_seconds=int(cfg["feed"].get("interval_seconds", 60))
    )
    feature_engine = FeatureEngine(
        window=int(cfg.get("feature_engine", {}).get("window", 20))
    )
    regime_engine = RegimeEngine(
        vol_threshold=float(cfg.get("regime", {}).get("vol_threshold", 0.003)),
        trend_threshold=float(cfg.get("regime", {}).get("trend_threshold", 0.001)),
    )

    consumer = asyncio.create_task(
        handle_events(
            bus,
            executor,
            portfolio,
            cfg["strategy"],
            RiskLimits(),
            logger,
            candle_builder,
            feature_engine,
            regime_engine,
        )
    )
    ws_client = BybitWebSocketClient(creds, testnet=creds.testnet)
    feed = asyncio.create_task(live_ws_to_bus(ws_client, sink, list(cfg["symbols"]), logger))
    try:
        await asyncio.sleep(int(cfg.get("run_seconds", 10)))
    finally:
        await bus.close()
        feed.cancel()
        consumer.cancel()
        await executor.close()
        logger.info(
            "live portfolio state: %s",
            json.dumps({k: v.__dict__ for k, v in portfolio.positions.items()}),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="codex quant async trading framework")
    sub = parser.add_subparsers(dest="command", required=True)

    paper = sub.add_parser("paper", help="run paper trading")
    paper.add_argument("--config", type=Path, default=Path("configs/default.yaml"))

    live = sub.add_parser("live", help="run live trading (Bybit)")
    live.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "paper":
        asyncio.run(run_paper(args.config))
    else:
        asyncio.run(run_live(args.config))


if __name__ == "__main__":
    main()
