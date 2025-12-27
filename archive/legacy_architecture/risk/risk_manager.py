from __future__ import annotations

from dataclasses import dataclass

from core.events import OrderEvent


def _abs(val: float) -> float:
    return val if val >= 0 else -val


@dataclass
class RiskLimits:
    max_position: float = 0.01
    max_order_notional: float = 1000.0
    max_drawdown: float = 0.2


def check_order(
    order: OrderEvent, price: float, limits: RiskLimits, risk_multiplier: float = 1.0
) -> bool:
    factor = max(risk_multiplier, 0.0)
    notional = _abs(order.quantity * price)
    if notional > limits.max_order_notional * factor:
        return False
    if _abs(order.quantity) > limits.max_position * factor:
        return False
    return True
