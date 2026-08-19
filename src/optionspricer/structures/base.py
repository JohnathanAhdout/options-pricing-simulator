"""Multi-leg option structures: a `Structure` is just an immutable tuple of
`OptionLeg`s and `StockLeg`s, each carrying a signed quantity and the price
it was entered at. Every strategy in this package (a covered call, a
straddle, a bull spread) is the *same* `Structure` type; what varies is
only which legs are in it.

Sign convention: `quantity` is positive for a long position and negative
for a short one; `entry_price` is always the *positive* price per unit the
leg traded at (a premium paid if long, a premium received if short). P&L
per leg is then always `quantity * (current_value - entry_price)`, which
falls out correctly for both directions without an if-statement. Selling
a call at premium p and having it expire worthless is
`(-1) * (0 - p) = +p`, exactly the credit you kept.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

from dataclasses import dataclass, replace  # replace: builds a time-shifted copy of an OptionSpec without mutating the original

import numpy as np  # vectorized payoff math in payoff_at_expiry

from optionspricer.market import Greeks, MarketData, OptionSpec  # the shared value objects every leg and Greek aggregation is built on
from optionspricer.pricing.base import PricingEngine  # the interface used to price/differentiate each leg


@dataclass(frozen=True, slots=True)  # same immutability rationale as OptionSpec/MarketData in market.py
class OptionLeg:
    option: OptionSpec  # the contract terms: strike, maturity, call/put
    quantity: float  # signed: positive long, negative short
    entry_price: float  # always positive: the premium paid (long) or received (short) per unit


@dataclass(frozen=True, slots=True)
class StockLeg:
    quantity: float  # signed shares: positive long, negative short
    entry_price: float  # always positive: the price per share paid or received


@dataclass(frozen=True, slots=True)
class Structure:
    name: str  # human-readable label, e.g. "Short Straddle"; not used for any logic, only display
    option_legs: tuple[OptionLeg, ...] = ()  # empty by default, so a pure-stock Structure is representable
    stock_legs: tuple[StockLeg, ...] = ()  # empty by default, so a pure-options Structure is representable

    @property  # exposes this as `structure.entry_cost` (no parens) even though it's computed, not stored
    def entry_cost(self) -> float:
        """Net premium paid to put the structure on (negative = net credit)."""
        cost = sum(leg.quantity * leg.entry_price for leg in self.option_legs)  # sum over every option leg: long legs add cost, short legs subtract it (negative quantity)
        cost += sum(leg.quantity * leg.entry_price for leg in self.stock_legs)  # same sign convention applied to any stock legs
        return cost


def _intrinsic(option: OptionSpec, S_T: np.ndarray) -> np.ndarray:  # the payoff if this option were exercised right now at spot S_T
    if option.is_call:
        return np.maximum(S_T - option.strike, 0.0)  # a call is worth S_T - K if that's positive, otherwise 0
    return np.maximum(option.strike - S_T, 0.0)  # a put is worth K - S_T if that's positive, otherwise 0


def payoff_at_expiry(structure: Structure, S_T: np.ndarray | float) -> np.ndarray:  # accepts a scalar OR an array of terminal spots, for plotting a whole payoff diagram at once
    """Net P&L at expiry as a function of terminal spot S_T (scalar or
    array). Sums each leg's `quantity * (payoff - entry_price)`; see module
    docstring for why that one expression covers long and short alike."""
    S_T = np.asarray(S_T, dtype=float)  # normalizes a scalar input into a 0-dimensional array, so the arithmetic below is uniform either way
    total = np.zeros_like(S_T)  # accumulator, same shape as S_T, starts at zero before any leg is added
    for leg in structure.option_legs:
        # quantity=+1 (long): payoff - premium_paid, the standard "what you get minus what you paid" P&L
        # quantity=-1 (short): -(payoff - premium_received) = premium_received - payoff, i.e. you keep the
        #   credit and owe the payoff if it finishes ITM. Same formula, sign of quantity does the rest
        total = total + leg.quantity * (_intrinsic(leg.option, S_T) - leg.entry_price)
    for leg in structure.stock_legs:
        total = total + leg.quantity * (S_T - leg.entry_price)  # identical logic: quantity*(exit price - entry price)
    return total


def mark_to_market(structure: Structure, market: MarketData, elapsed: float, engine: PricingEngine) -> float:  # current P&L, not P&L AT expiry: the difference is that legs here still have time value
    """Current P&L if the structure were closed out right now, `elapsed`
    years after it was put on. Each option leg is re-priced at its
    *remaining* time to expiry (`leg.option.maturity - elapsed`) under
    today's spot/vol. Once that remaining time hits zero, the leg is
    valued at intrinsic instead of asking the pricing engine to price a
    zero-maturity option."""
    total = 0.0  # plain float accumulator, unlike payoff_at_expiry's array (this function only ever evaluates at one point in time)
    for leg in structure.option_legs:
        tau = leg.option.maturity - elapsed  # remaining time to expiry, NOT the original maturity: this is what "current" means here
        if tau <= 1e-6:
            current_value = float(_intrinsic(leg.option, np.array(market.spot)))  # essentially expired: value at intrinsic rather than asking the engine to price a near-zero-maturity option
        else:
            live_option = replace(leg.option, maturity=tau)  # a new OptionSpec with the shrunk maturity; the original leg.option is never mutated
            current_value = engine.price(live_option, market).price  # re-price the shrunk-maturity option under today's market
        total += leg.quantity * (current_value - leg.entry_price)  # same sign convention as payoff_at_expiry, just with a mark-to-market value instead of an expiry payoff
    for leg in structure.stock_legs:
        total += leg.quantity * (market.spot - leg.entry_price)  # stock has no time value to worry about; today's spot is all that matters
    return total


def portfolio_greeks(structure: Structure, market: MarketData, engine: PricingEngine) -> Greeks:  # position-level Greeks, used by hedging.py to size the delta hedge
    """Position-level Greeks: the quantity-weighted sum of each leg's
    Greeks. This is what actually gets used to make decisions elsewhere in
    the package. `hedging.py` reads `portfolio_greeks(...).delta` off a
    structure to know how many shares of stock offset it, and the
    regime-driven strategy in `experiments/run_regime_gamma_scalping.py`
    reads `.gamma` and `.theta` to know what it's actually exposed to. A
    stock leg contributes 1.0 to delta and exactly nothing to gamma, theta,
    vega, or rho, since a share's value is linear in itself."""
    delta = gamma = theta = vega = rho = 0.0  # five accumulators, all starting at zero, chained-assigned in one line
    for leg in structure.option_legs:
        g = engine.greeks(leg.option, market)  # each leg's OWN Greeks, unscaled by quantity yet
        delta += leg.quantity * g.delta  # quantity-weighted sum: linearity of differentiation is what makes this valid
        gamma += leg.quantity * g.gamma
        theta += leg.quantity * g.theta
        vega += leg.quantity * g.vega
        rho += leg.quantity * g.rho
    for leg in structure.stock_legs:
        delta += leg.quantity  # a share's delta is exactly 1 (dV/dS = 1 for V = S); gamma/theta/vega/rho are all zero, so nothing else to add
    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)  # bundle all five into the shared output type
