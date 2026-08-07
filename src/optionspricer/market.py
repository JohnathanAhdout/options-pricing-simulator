
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

from __future__ import annotations  # postpones evaluation of type hints (e.g. `float | None`) so they're just strings until something actually inspects them

from dataclasses import dataclass  # decorator: auto-generates __init__/__repr__/__eq__ from the class's field annotations
from enum import Enum  # base class for a fixed, named set of values


class OptionType(str, Enum):  # inherits str AND Enum: members compare equal to plain strings ("call" == OptionType.CALL) and serialize cleanly
    CALL = "call"  # OptionType.CALL; its .value and str() are both "call"
    PUT = "put"  # OptionType.PUT; its .value and str() are both "put"


class ExerciseStyle(str, Enum):  # same str+Enum trick, for the two exercise conventions this package supports
    EUROPEAN = "european"  # exercise only possible at maturity T -- what the closed form and Monte Carlo engine assume
    AMERICAN = "american"  # exercise possible any time up to and including T -- only the binomial tree can price this


@dataclass(frozen=True, slots=True)  # frozen: mutating a field after construction raises; slots: no per-instance __dict__, less memory, no stray attributes
class OptionSpec:
    """The terms of the contract. Nothing here depends on today's market."""

    strike: float  # K: the price at which the option can be exercised
    maturity: float  # years to expiry, T
    option_type: OptionType  # CALL or PUT
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN  # defaults to European unless explicitly constructed as American

    def __post_init__(self) -> None:  # dataclass hook, runs immediately after the generated __init__ -- the natural place to enforce invariants
        if self.strike <= 0:  # a zero or negative strike isn't a real contract
            raise ValueError(f"strike must be positive, got {self.strike}")  # fail immediately at construction, not three formulas later as a mystery NaN
        if self.maturity <= 0:  # a non-positive maturity breaks every formula that divides by sqrt(T)
            raise ValueError(f"maturity must be positive, got {self.maturity}")  # same fail-fast rationale as the strike check

    @property  # exposes this as `spec.is_call` (no parens) even though it's computed, not stored
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL  # True only for calls -- avoids repeating this comparison at every call site in the codebase

    @property
    def is_american(self) -> bool:
        return self.exercise == ExerciseStyle.AMERICAN  # True only for American exercise -- engines that can't handle early exercise check this and raise


@dataclass(frozen=True, slots=True)
class MarketData:
    """A snapshot of the world an option gets priced against."""

    spot: float  # S: today's price of the underlying
    rate: float  # r: the continuously-compounded risk-free rate
    vol: float  # sigma: the volatility used to price/hedge with (not necessarily what actually realizes -- see hedging.py)
    dividend_yield: float = 0.0  # q: continuous dividend yield; defaults to 0 for non-dividend-paying underlyings

    def __post_init__(self) -> None:  # same fail-fast validation pattern as OptionSpec
        if self.spot <= 0:  # a non-positive stock price isn't physically meaningful
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.vol < 0:  # volatility is a standard deviation: can be zero (a deterministic world) but never negative
            raise ValueError(f"vol must be non-negative, got {self.vol}")


@dataclass(frozen=True, slots=True)
class Greeks:
    """First- and second-order sensitivities. Sign/scale conventions:
    vega and rho are per 1% (0.01) move, theta is per calendar day."""

    delta: float  # dV/dS: dollars of option value per $1 move in the underlying
    gamma: float  # d^2V/dS^2: how fast delta itself changes per $1 move in the underlying
    theta: float  # dV/dt: dollars of option value gained/lost per calendar day of time passing
    vega: float  # dV/dsigma: dollars of option value per 1-percentage-point move in volatility
    rho: float  # dV/dr: dollars of option value per 1-percentage-point move in the risk-free rate


@dataclass(frozen=True, slots=True)
class PriceResult:
    """A price plus provenance. `stderr` is populated by simulation-based
    engines (Monte Carlo) and left as None by closed-form/lattice engines,
    since those have no sampling error to report."""

    price: float  # the fair value produced by whichever engine ran
    engine: str  # name of that engine (e.g. "black_scholes"), so a caller can tell which algorithm produced this number
    stderr: float | None = None  # standard error of the estimate; None for exact methods, a real number for Monte Carlo's sampling noise
