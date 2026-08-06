"""Monte Carlo pricing: price an option by simulating the risk-neutral
terminal distribution of the underlying and averaging discounted payoffs.
Converges to the Black-Scholes price as n_paths -> infinity by the Law of
Large Numbers -- it exists here as an independent cross-check of the
closed form (see experiments/run_mc_convergence.py) and as a template for
pricing payoffs that don't have one, which is most of them.

The one design choice worth calling out: `MonteCarloEngine` reseeds its RNG
from `self.seed` on *every* call to `price()`, rather than advancing a
single stream. Two consequences fall out of that:

1. `engine.price(option, market)` is a pure function of its arguments -- call
   it twice with the same inputs and you get bit-identical output. That's
   what "reproducible" means, and it's what lets a test suite assert exact
   equality on a Monte Carlo price instead of "close to within some
   tolerance I had to guess."
2. The default finite-difference `greeks()` from `PricingEngine` calls
   `price()` five times with slightly bumped market data. Because every one
   of those calls draws the *same* underlying Z's (same seed), the bumped
   and unbumped payoffs are correlated draws of the same random experiment,
   not independent ones -- so the differences that go into delta/gamma/vega
   are (mostly) differences in the deterministic bump, not differences in
   sampling noise. This is the standard variance-reduction trick of
   common random numbers, and here it's a free side effect of reseeding
   deterministically rather than something bolted on separately.
"""

from __future__ import annotations

import numpy as np

from optionspricer.market import MarketData, OptionSpec, OptionType, PriceResult
from optionspricer.pricing.base import PricingEngine
from optionspricer.simulation import gbm_terminal


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
    n_paths: int = 100_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Discounted mean payoff over n_paths simulated terminal prices, plus
    its standard error. Returns (price, stderr); stderr shrinks as
    1 / sqrt(n_paths) by the Central Limit Theorem, so quadrupling n_paths
    halves it."""
    rng = rng if rng is not None else np.random.default_rng()
    ST = gbm_terminal(S, T, r, sigma, q, n_paths, rng)
    payoffs = np.maximum(ST - K, 0.0) if option_type == OptionType.CALL else np.maximum(K - ST, 0.0)
    disc = np.exp(-r * T)
    return float(disc * payoffs.mean()), float(disc * payoffs.std(ddof=1) / np.sqrt(n_paths))


class MonteCarloEngine(PricingEngine):
    """European options, simulated. See module docstring for why this
    engine deterministically reseeds on every `price()` call."""

    name = "monte_carlo"

    def __init__(self, n_paths: int = 100_000, seed: int = 0):
        self.n_paths = n_paths
        self.seed = seed

    def price(self, option: OptionSpec, market: MarketData) -> PriceResult:
        if option.is_american:
            raise ValueError(
                "MonteCarloEngine only prices European exercise (no early-exercise "
                "boundary); use BinomialEngine for American options."
            )
        rng = np.random.default_rng(self.seed)
        p, se = price(
            market.spot, option.strike, option.maturity, market.rate, market.vol,
            option.option_type, market.dividend_yield, self.n_paths, rng,
        )
        return PriceResult(price=p, engine=self.name, stderr=se)
