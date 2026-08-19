"""The Strategy pattern for implied-volatility root-finders.

Given a market price, an `IVSolver` inverts Black-Scholes for the sigma
that reproduces it: find sigma such that `bs_price(S, K, T, r, sigma, q)
== market_price`. Three solvers (`NewtonSolver`, `BrentSolver`,
`JaeckelSolver`) implement the same interface with very different
mechanics and very different failure modes. See
`experiments/run_iv_solver_benchmark.py` for a head-to-head under normal
and adversarial (deep OTM, near-expiry) conditions, and BACKGROUND.md for
why the root is guaranteed to exist and be unique in the first place.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

from abc import ABC, abstractmethod  # same ABC/abstractmethod mechanism as PricingEngine in pricing/base.py

from optionspricer.market import OptionType  # the only value object a solver needs; no MarketData/OptionSpec, since inputs are passed as plain floats here


class IVSolver(ABC):  # every concrete solver (NewtonSolver, BrentSolver, JaeckelSolver) inherits from this
    name: str  # class-level type annotation, not an assignment: each subclass must set its own string (e.g. "newton")

    @abstractmethod  # every solver MUST implement solve(); there's no shared default the way PricingEngine.greeks() has one
    def solve(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """Return sigma such that the Black-Scholes price matches
        `market_price`, or the solver's best estimate if it didn't fully
        converge within its iteration budget."""
