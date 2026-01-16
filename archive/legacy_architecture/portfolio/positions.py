from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from core.events import FillEvent, OrderSide


@dataclass
class PositionState:
    quantity: float = 0.0
    avg_price: float = 0.0

    def update_with_fill(self, fill: FillEvent) -> None:
        signed_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        new_qty = self.quantity + signed_qty
        if new_qty == 0:
            self.quantity = 0.0
            self.avg_price = 0.0
            return
        if self.quantity == 0:
            self.avg_price = fill.price
        else:
            self.avg_price = (self.avg_price * self.quantity + fill.price * signed_qty) / new_qty
        self.quantity = new_qty


@dataclass
class Portfolio:
    positions: Dict[str, PositionState] = field(default_factory=dict)
    fees_paid: float = 0.0

    def get_position(self, symbol: str) -> PositionState:
        return self.positions.setdefault(symbol, PositionState())

    def apply_fill(self, fill: FillEvent) -> None:
        position = self.get_position(fill.symbol)
        position.update_with_fill(fill)
        self.fees_paid += fill.fee
