from __future__ import annotations

from typing import Dict


class RegimeEngine:
    """Simple rule-based regime classifier for dynamic risk sizing."""

    def __init__(self, vol_threshold: float = 0.003, trend_threshold: float = 0.001) -> None:
        self.vol_threshold = vol_threshold
        self.trend_threshold = trend_threshold

    def evaluate(self, features: Dict[str, float]) -> str:
        vol = features.get("volatility", 0.0)
        ret = features.get("return", 0.0)
        if vol > self.vol_threshold:
            return "risk_off"
        if ret > self.trend_threshold:
            return "bull"
        if ret < -self.trend_threshold:
            return "bear"
        return "neutral"

    def risk_multiplier(self, regime: str) -> float:
        mapping = {
            "risk_off": 0.5,
            "bear": 0.7,
            "neutral": 1.0,
            "bull": 1.2,
        }
        return mapping.get(regime, 1.0)
