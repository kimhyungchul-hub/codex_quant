from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from core.events import BarEvent, Event, MarketEvent


@dataclass
class _CandleBucket:
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleBuilder:
    """Aggregates tick-level market events into time-bucketed bars."""

    def __init__(self, interval_seconds: int = 60) -> None:
        self.interval = timedelta(seconds=interval_seconds)
        self._buckets: Dict[str, _CandleBucket] = {}

    def _bucket_start(self, ts: datetime) -> datetime:
        seconds = int(ts.timestamp())
        bucket = seconds - (seconds % int(self.interval.total_seconds()))
        return datetime.fromtimestamp(bucket, tz=timezone.utc)

    def update(self, event: Event) -> List[BarEvent]:
        if isinstance(event, BarEvent):
            return [event]
        if not isinstance(event, MarketEvent):
            return []

        ts = event.ts
        bucket_start = self._bucket_start(ts)
        bucket_end = bucket_start + self.interval
        bucket = self._buckets.get(event.symbol)
        emitted: List[BarEvent] = []

        if bucket is None or bucket.start != bucket_start:
            if bucket is not None:
                emitted.append(self._finalize(event.symbol, bucket))
            bucket = _CandleBucket(
                start=bucket_start,
                end=bucket_end,
                open=event.price,
                high=event.price,
                low=event.price,
                close=event.price,
                volume=abs(event.size),
            )
            self._buckets[event.symbol] = bucket
        else:
            bucket.high = max(bucket.high, event.price)
            bucket.low = min(bucket.low, event.price)
            bucket.close = event.price
            bucket.volume += abs(event.size)

        if ts >= bucket.end:
            emitted.append(self._finalize(event.symbol, bucket))
            self._buckets.pop(event.symbol, None)
        return emitted

    def _finalize(self, symbol: str, bucket: _CandleBucket) -> BarEvent:
        bar = BarEvent(
            symbol=symbol,
            open=bucket.open,
            high=bucket.high,
            low=bucket.low,
            close=bucket.close,
            volume=bucket.volume,
            payload={"source": "candle_builder"},
        )
        bar.ts = bucket.end
        return bar
