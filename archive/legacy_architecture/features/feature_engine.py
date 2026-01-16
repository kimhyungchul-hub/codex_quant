from __future__ import annotations

from collections import deque
from typing import Deque, Dict

from core.events import BarEvent


class FeatureEngine:
    """Lightweight feature calculator over rolling bar windows."""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.history: Dict[str, Deque[float]] = {}

    def update(self, bar: BarEvent) -> Dict[str, float]:
        buf = self.history.setdefault(bar.symbol, deque(maxlen=self.window))
        buf.append(bar.close)
        if len(buf) < 2:
            return {"return": 0.0, "volatility": 0.0, "sma": bar.close}

        last = buf[-1]
        prev = buf[-2]
        momentum = last / prev - 1.0 if prev else 0.0
        mean_price = sum(buf) / len(buf)
        variance = sum((p - mean_price) ** 2 for p in buf) / len(buf)
        volatility = variance**0.5
        return {
            "return": momentum,
            "volatility": volatility,
            "sma": mean_price,
        }
