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

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from optionspricer.market import Greeks, MarketData, OptionSpec
from optionspricer.pricing.base import PricingEngine


@dataclass(frozen=True, slots=True)
class OptionLeg:
    option: OptionSpec
    quantity: float
    entry_price: float


@dataclass(frozen=True, slots=True)
class StockLeg:
    quantity: float
    entry_price: float


@dataclass(frozen=True, slots=True)
class Structure:
    name: str
    option_legs: tuple[OptionLeg, ...] = ()
    stock_legs: tuple[StockLeg, ...] = ()

    @property
    def entry_cost(self) -> float:
        """Net premium paid to put the structure on (negative = net credit)."""
        cost = sum(leg.quantity * leg.entry_price for leg in self.option_legs)
        cost += sum(leg.quantity * leg.entry_price for leg in self.stock_legs)
        return cost


def _intrinsic(option: OptionSpec, S_T: np.ndarray) -> np.ndarray:
    if option.is_call:
        return np.maximum(S_T - option.strike, 0.0)
    return np.maximum(option.strike - S_T, 0.0)


def payoff_at_expiry(structure: Structure, S_T: np.ndarray | float) -> np.ndarray:
    """Net P&L at expiry as a function of terminal spot S_T (scalar or
    array). Sums each leg's `quantity * (payoff - entry_price)`; see module
    docstring for why that one expression covers long and short alike."""
    S_T = np.asarray(S_T, dtype=float)
    total = np.zeros_like(S_T)
    for leg in structure.option_legs:
        # quantity=+1 (long): payoff - premium_paid, the standard "what you get minus what you paid" P&L
        # quantity=-1 (short): -(payoff - premium_received) = premium_received - payoff, i.e. you keep the
        #   credit and owe the payoff if it finishes ITM. Same formula, sign of quantity does the rest
        total = total + leg.quantity * (_intrinsic(leg.option, S_T) - leg.entry_price)
    for leg in structure.stock_legs:
        total = total + leg.quantity * (S_T - leg.entry_price)  # identical logic: quantity*(exit price - entry price)
    return total


def mark_to_market(structure: Structure, market: MarketData, elapsed: float, engine: PricingEngine) -> float:
    """Current P&L if the structure were closed out right now, `elapsed`
    years after it was put on. Each option leg is re-priced at its
    *remaining* time to expiry (`leg.option.maturity - elapsed`) under
    today's spot/vol. Once that remaining time hits zero, the leg is
    valued at intrinsic instead of asking the pricing engine to price a
    zero-maturity option."""
    total = 0.0
    for leg in structure.option_legs:
        tau = leg.option.maturity - elapsed
        if tau <= 1e-6:
            current_value = float(_intrinsic(leg.option, np.array(market.spot)))
        else:
            live_option = replace(leg.option, maturity=tau)
            current_value = engine.price(live_option, market).price
        total += leg.quantity * (current_value - leg.entry_price)
    for leg in structure.stock_legs:
        total += leg.quantity * (market.spot - leg.entry_price)
    return total


def portfolio_greeks(structure: Structure, market: MarketData, engine: PricingEngine) -> Greeks:
    """Position-level Greeks: the quantity-weighted sum of each leg's
    Greeks. This is what actually gets used to make decisions elsewhere in
    the package. `hedging.py` reads `portfolio_greeks(...).delta` off a
    structure to know how many shares of stock offset it, and the
    regime-driven strategy in `experiments/run_regime_gamma_scalping.py`
    reads `.gamma` and `.theta` to know what it's actually exposed to. A
    stock leg contributes 1.0 to delta and exactly nothing to gamma, theta,
    vega, or rho, since a share's value is linear in itself."""
    delta = gamma = theta = vega = rho = 0.0
    for leg in structure.option_legs:
        g = engine.greeks(leg.option, market)
        delta += leg.quantity * g.delta
        gamma += leg.quantity * g.gamma
        theta += leg.quantity * g.theta
        vega += leg.quantity * g.vega
        rho += leg.quantity * g.rho
    for leg in structure.stock_legs:
        delta += leg.quantity
    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
