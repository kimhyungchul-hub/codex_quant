from __future__ import annotations

import asyncio
from asyncio import Queue
from typing import AsyncIterator, List, Optional

from core.events import Event


class AsyncEventBus:
    """A lightweight async pub/sub bus for event-driven pipelines."""

    def __init__(self) -> None:
        self._subscribers: List[Queue[Event]] = []
        self._closed = False

    def subscribe(self, max_queue: int = 1000) -> Queue[Event]:
        if self._closed:
            raise RuntimeError("Bus is closed")
        queue: Queue[Event] = Queue(max_queue)
        self._subscribers.append(queue)
        return queue

    async def publish(self, event: Event) -> None:
        if self._closed:
            raise RuntimeError("Bus is closed")
        for queue in self._subscribers:
            await queue.put(event)

    async def iterate(self, queue: Queue[Event]) -> AsyncIterator[Event]:
        try:
            while not self._closed:
                event = await queue.get()
                queue.task_done()
                yield event
        finally:
            await self.close_queue(queue)

    async def close_queue(self, queue: Queue[Event]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for queue in list(self._subscribers):
            await self.close_queue(queue)


class EventSink:
    """Simple sink to fan-out events into the bus with optional filtering."""

    def __init__(self, bus: AsyncEventBus, symbols: Optional[List[str]] = None) -> None:
        self.bus = bus
        self.symbols = set(symbols) if symbols else None

    def accepts(self, symbol: str) -> bool:
        return self.symbols is None or symbol in self.symbols

    async def emit(self, event: Event) -> None:
        await self.bus.publish(event)
