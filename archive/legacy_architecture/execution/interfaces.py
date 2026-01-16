from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from core.events import FillEvent, OrderEvent


@dataclass
class ExecutionResult:
    accepted: bool
    fill: Optional[FillEvent] = None
    reason: Optional[str] = None


class Executor(abc.ABC):
    @abc.abstractmethod
    async def submit(self, order: OrderEvent) -> ExecutionResult:
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
