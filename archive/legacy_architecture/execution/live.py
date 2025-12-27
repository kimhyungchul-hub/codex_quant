from __future__ import annotations

import logging
from typing import Optional

from core.events import FillEvent, OrderEvent, OrderSide
from data.feeds.bybit_api import BybitRESTClient
from execution.interfaces import ExecutionResult, Executor
from portfolio.positions import Portfolio


class LiveExecutor(Executor):
    """Live trading executor that routes orders to Bybit REST."""

    def __init__(self, portfolio: Portfolio, rest_client: BybitRESTClient) -> None:
        self.portfolio = portfolio
        self.rest_client = rest_client
        self.logger = logging.getLogger("live_executor")

    async def submit(self, order: OrderEvent) -> ExecutionResult:
        try:
            response = await self.rest_client.create_order(
                symbol=order.symbol,
                side=order.side.value,
                qty=order.quantity,
                order_type=order.order_type.value,
                price=order.price,
            )
        except Exception as exc:  # pragma: no cover - network errors
            self.logger.exception("order submit failed: %s", exc)
            return ExecutionResult(accepted=False, reason=str(exc))

        if response.get("retCode", -1) != 0:
            return ExecutionResult(
                accepted=False, reason=str(response.get("retMsg", "unknown error"))
            )

        result = response.get("result", {})
        price = order.price or float(result.get("avgPrice") or result.get("price") or 0.0)
        fill: Optional[FillEvent] = None
        if price and order.quantity:
            fill = FillEvent(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=price,
                fee=0.0,
                maker=False,
                client_order_id=result.get("orderId", order.client_order_id),
                payload={"exchange_response": response},
            )
            self.portfolio.apply_fill(fill)
        return ExecutionResult(accepted=True, fill=fill)

    async def close(self) -> None:
        return None
