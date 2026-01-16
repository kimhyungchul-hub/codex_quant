from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Optional

from core.events import BarEvent, OrderEvent, OrderSide, OrderType


@dataclass
class StrategyContext:
    symbol: str
    position: float
    avg_price: float
    config: Dict[str, float]
    features: Optional[Dict[str, float]] = None
    regime: str = "neutral"


class Strategy(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def on_bar(self, bar: BarEvent, ctx: StrategyContext) -> Optional[OrderEvent]:
        raise NotImplementedError


class MomentumBreakoutStrategy(Strategy):
    name = "momentum_breakout"

    def __init__(self, lookback: int = 3, threshold: float = 0.002) -> None:
        self.lookback = lookback
        self.threshold = threshold
        self._history: Dict[str, list[float]] = {}

    def on_bar(self, bar: BarEvent, ctx: StrategyContext) -> Optional[OrderEvent]:
        hist = self._history.setdefault(bar.symbol, [])
        hist.append(bar.close)
        if len(hist) > self.lookback:
            hist.pop(0)
        if len(hist) < self.lookback:
            return None

        momentum = hist[-1] / hist[0] - 1.0
        if momentum > self.threshold:
            return OrderEvent(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                quantity=ctx.config.get("order_size", 0.001),
                order_type=OrderType.MARKET,
                payload={"reason": "breakout", "momentum": momentum, "regime": ctx.regime},
            )
        if momentum < -self.threshold:
            return OrderEvent(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                quantity=ctx.config.get("order_size", 0.001),
                order_type=OrderType.MARKET,
                payload={"reason": "breakdown", "momentum": momentum, "regime": ctx.regime},
            )
        return None
