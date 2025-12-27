from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

from core.events import FillEvent, OrderEvent, OrderSide
from execution.interfaces import ExecutionResult, Executor
from portfolio.positions import Portfolio


@dataclass
class FillParams:
    fee_rate: float = 0.0006
    slippage_bps: float = 5.0
    latency_ms: int = 50


class PaperExecutor(Executor):
    def __init__(self, portfolio: Portfolio, params: Optional[FillParams] = None) -> None:
        self.portfolio = portfolio
        self.params = params or FillParams()

    async def submit(self, order: OrderEvent) -> ExecutionResult:
        await asyncio.sleep(self.params.latency_ms / 1000.0)
        price = order.price or order.payload.get("last_price") or 0.0
        if price <= 0:
            return ExecutionResult(accepted=False, reason="missing price")
        slip = price * self.params.slippage_bps / 10000.0
        effective_price = price + slip if order.side == OrderSide.BUY else price - slip
        fee = abs(order.quantity * effective_price) * self.params.fee_rate
        fill = FillEvent(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=effective_price,
            fee=fee,
            maker=False,
            client_order_id=order.client_order_id,
            payload={"slippage": slip},
        )
        self.portfolio.apply_fill(fill)
        return ExecutionResult(accepted=True, fill=fill)

    async def close(self) -> None:
        return None
