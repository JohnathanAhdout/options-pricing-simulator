"""Monte Carlo pricing: price an option by simulating the risk-neutral
terminal distribution of the underlying and averaging discounted payoffs.
Converges to the Black-Scholes price as n_paths -> infinity by the Law of
Large Numbers. It exists here as an independent cross-check of the
closed form (see experiments/run_mc_convergence.py) and as a template for
pricing payoffs that don't have one, which is most of them.

The one design choice worth calling out: `MonteCarloEngine` reseeds its RNG
from `self.seed` on *every* call to `price()`, rather than advancing a
single stream. Two consequences fall out of that:

1. `engine.price(option, market)` is a pure function of its arguments: call
   it twice with the same inputs and you get bit-identical output. That's
   what "reproducible" means, and it's what lets a test suite assert exact
   equality on a Monte Carlo price instead of "close to within some
   tolerance I had to guess."
2. The default finite-difference `greeks()` from `PricingEngine` calls
   `price()` five times with slightly bumped market data. Because every one
   of those calls draws the *same* underlying Z's (same seed), the bumped
   and unbumped payoffs are correlated draws of the same random experiment,
   not independent ones, so the differences that go into delta/gamma/vega
   are (mostly) differences in the deterministic bump, not differences in
   sampling noise. This is the standard variance-reduction trick of
   common random numbers, and here it's a free side effect of reseeding
   deterministically rather than something bolted on separately.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

import numpy as np  # used only for the return-type annotation and passthrough; the actual random draws live in simulation.py

from optionspricer.market import MarketData, OptionSpec, OptionType, PriceResult  # the shared value objects this module reads and returns
from optionspricer.pricing.base import PricingEngine  # the interface MonteCarloEngine implements
from optionspricer.simulation import gbm_terminal  # the one function that actually draws random terminal prices


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
) -> tuple[float, float]:  # returns (price, stderr), not just price: the stderr is the whole point of running this many paths
    """Discounted mean payoff over n_paths simulated terminal prices, plus
    its standard error. Returns (price, stderr); stderr shrinks as
    1 / sqrt(n_paths) by the Central Limit Theorem, so quadrupling n_paths
    halves it."""
    rng = rng if rng is not None else np.random.default_rng()  # caller-supplied generator if given, otherwise a fresh unseeded one
    ST = gbm_terminal(S, T, r, sigma, q, n_paths, rng)  # one array of n_paths simulated terminal prices, all drawn under the risk-neutral measure
    payoffs = np.maximum(ST - K, 0.0) if option_type == OptionType.CALL else np.maximum(K - ST, 0.0)  # the option's payoff at each simulated terminal price
    disc = np.exp(-r * T)  # present-value factor, applied once to the whole batch of payoffs
    return float(disc * payoffs.mean()), float(disc * payoffs.std(ddof=1) / np.sqrt(n_paths))  # price = discounted sample mean; stderr = discounted sample std / sqrt(n), the CLT formula


class MonteCarloEngine(PricingEngine):  # the imperative shell: adapts the price() function above to the common PricingEngine interface
    """European options, simulated. See module docstring for why this
    engine deterministically reseeds on every `price()` call."""

    name = "monte_carlo"  # the string every factory/experiment uses to select this engine

    def __init__(self, n_paths: int = 100_000, seed: int = 0):  # n_paths and seed are stored on the instance, not passed at call time
        self.n_paths = n_paths  # how many simulated paths every price()/greeks() call uses
        self.seed = seed  # the fixed integer reseeded from on every price() call, see module docstring

    def price(self, option: OptionSpec, market: MarketData) -> PriceResult:  # note: shadows the module-level price() function above, same pattern as BlackScholesEngine
        if option.is_american:  # Monte Carlo as implemented here has no early-exercise boundary; refuse rather than silently mispricing
            raise ValueError(
                "MonteCarloEngine only prices European exercise (no early-exercise "
                "boundary); use BinomialEngine for American options."
            )
        rng = np.random.default_rng(self.seed)  # fresh generator, reseeded from the SAME integer every call: this is the reproducibility mechanism described above
        p, se = price(
            market.spot, option.strike, option.maturity, market.rate, market.vol,
            option.option_type, market.dividend_yield, self.n_paths, rng,
        )  # delegates to the module-level function, unpacking the two dataclasses into plain floats
        return PriceResult(price=p, engine=self.name, stderr=se)  # unlike BlackScholesEngine, stderr is populated here: this price is genuinely sampled, not exact
