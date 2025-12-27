from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class EventType(str, enum.Enum):
    TICK = "tick"
    BAR = "bar"
    ORDER = "order"
    FILL = "fill"
    POSITION = "position"
    CONTROL = "control"


@dataclass
class Event:
    event_type: EventType = field(init=False)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketEvent(Event):
    symbol: str = ""
    price: float = 0.0
    size: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = EventType.TICK


@dataclass
class BarEvent(Event):
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = EventType.BAR


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class OrderEvent(Event):
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    client_order_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    reduce_only: bool = False

    def __post_init__(self) -> None:
        self.event_type = EventType.ORDER


@dataclass
class FillEvent(Event):
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    fee: float = 0.0
    maker: bool = False
    client_order_id: str = ""

    def __post_init__(self) -> None:
        self.event_type = EventType.FILL


@dataclass
class PositionEvent(Event):
    symbol: str = ""
    position: float = 0.0
    avg_price: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = EventType.POSITION


@dataclass
class ControlEvent(Event):
    command: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_type = EventType.CONTROL
