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

from __future__ import annotations

from abc import ABC, abstractmethod

from optionspricer.market import OptionType


class IVSolver(ABC):
    name: str

    @abstractmethod
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
