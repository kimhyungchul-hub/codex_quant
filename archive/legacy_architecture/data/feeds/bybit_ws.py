from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Dict, Iterable, List

from core.bus import EventSink
from core.events import BarEvent


@dataclass
class SimulatedFeedConfig:
    symbols: List[str]
    interval_seconds: int = 60
    seed: int = 42
    drift: float = 0.0
    volatility: float = 0.001


def _generate_bar(now: datetime, last_price: float, cfg: SimulatedFeedConfig) -> tuple[float, float, float, float]:
    rnd = random.Random(cfg.seed + int(now.timestamp()))
    shock = rnd.gauss(cfg.drift, cfg.volatility)
    new_price = max(1e-6, last_price * math.exp(shock))
    high = max(new_price, last_price)
    low = min(new_price, last_price)
    vol = abs(shock) * 100
    return new_price, high, low, vol


async def simulated_feed(sink: EventSink, cfg: SimulatedFeedConfig) -> None:
    """A minimal async feed that emits synthetic bars for each symbol."""

    prices: Dict[str, float] = {symbol: 100.0 for symbol in cfg.symbols}
    ts = datetime.now(timezone.utc)
    while True:
        ts += timedelta(seconds=cfg.interval_seconds)
        for symbol in cfg.symbols:
            price, high, low, vol = _generate_bar(ts, prices[symbol], cfg)
            prices[symbol] = price
            event = BarEvent(
                symbol=symbol,
                open=prices[symbol],
                high=high,
                low=low,
                close=price,
                volume=vol,
                payload={"source": "simulated"},
            )
            event.ts = ts
            if sink.accepts(symbol):
                await sink.emit(event)
        await asyncio.sleep(cfg.interval_seconds)
