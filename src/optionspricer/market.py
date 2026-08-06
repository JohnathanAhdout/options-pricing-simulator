"""Immutable value objects shared by every pricing engine.

The split mirrors how a real desk thinks about an option: the *contract*
(strike, maturity, call/put, exercise style) never changes once it's written,
while the *market* (spot, rate, vol, dividend yield) moves every tick. Keeping
them as two separate frozen dataclasses -- instead of one big bag of
parameters -- means an engine's signature `price(option, market)` documents
exactly which half of the world it's allowed to read, and nothing in this
codebase ever mutates an option or a market snapshot in place. That
immutability is what makes it safe to reuse the same `OptionSpec` across a
sweep of a hundred market scenarios without worrying that some engine quietly
changed it out from under you.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """The terms of the contract. Nothing here depends on today's market."""

    strike: float
    maturity: float  # years to expiry, T
    option_type: OptionType
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    @property
    def is_american(self) -> bool:
        return self.exercise == ExerciseStyle.AMERICAN


@dataclass(frozen=True, slots=True)
class MarketData:
    """A snapshot of the world an option gets priced against."""

    spot: float
    rate: float
    vol: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.vol < 0:
            raise ValueError(f"vol must be non-negative, got {self.vol}")


@dataclass(frozen=True, slots=True)
class Greeks:
    """First- and second-order sensitivities. Sign/scale conventions:
    vega and rho are per 1% (0.01) move, theta is per calendar day."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass(frozen=True, slots=True)
class PriceResult:
    """A price plus provenance. `stderr` is populated by simulation-based
    engines (Monte Carlo) and left as None by closed-form/lattice engines,
    since those have no sampling error to report."""

    price: float
    engine: str
    stderr: float | None = None
