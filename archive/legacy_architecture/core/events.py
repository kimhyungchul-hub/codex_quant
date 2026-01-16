from dataclasses import dataclass
import enum

class Event:
    pass

@dataclass
class MarketEvent(Event):
    symbol: str = ""

    def __post_init__(self) -> None:
        self.greeting = "Hey"

@dataclass
class BarEvent(Event):
    symbol: str = ""

    def __post_init__(self) -> None:
        self.greeting = "Hey"

# Rest of the code...
