"""Factory for pricing engines: turns a name (a string that could come from
a config file, a CLI flag, or a sweep over `["black_scholes", "monte_carlo",
"binomial"]`) into a live `PricingEngine`.

It's a registry rather than an if/elif chain on purpose: adding a new engine
means calling `register_engine` next to the class definition, not editing
this file. That's the open/closed principle in practice: this module is
closed for modification but open for extension, and it's what lets
experiments/*.py iterate over "every registered engine" generically instead
of hardcoding a list that has to be kept in sync by hand.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

from optionspricer.pricing.base import PricingEngine  # the common type every registry value is a subclass of
from optionspricer.pricing.binomial import BinomialEngine  # imported here (not lazily) so registration below can run at import time
from optionspricer.pricing.black_scholes import BlackScholesEngine
from optionspricer.pricing.monte_carlo import MonteCarloEngine

_REGISTRY: dict[str, type[PricingEngine]] = {}  # name -> class, not name -> instance; instantiated fresh on every create_pricing_engine call


def register_engine(name: str, engine_cls: type[PricingEngine]) -> None:  # called once per engine, at the bottom of this file
    _REGISTRY[name] = engine_cls  # mutates the module-level dict; no return value needed


def create_pricing_engine(name: str, **kwargs) -> PricingEngine:  # **kwargs forwards constructor args, e.g. n_steps=500 for BinomialEngine
    """Build a `PricingEngine` by name, e.g. `create_pricing_engine("binomial", n_steps=500)`."""
    try:
        engine_cls = _REGISTRY[name]  # KeyError here means the name was never registered
    except KeyError:
        raise ValueError(f"unknown pricing engine {name!r}; available: {sorted(_REGISTRY)}") from None  # `from None` suppresses the KeyError traceback, since it's not useful context for the caller
    return engine_cls(**kwargs)  # instantiate the class, passing through whatever kwargs the caller supplied


def available_engines() -> list[str]:  # what experiments/*.py loop over instead of hardcoding a list
    return sorted(_REGISTRY)  # sorted() on a dict iterates its keys; alphabetical order makes output/plots reproducible


register_engine("black_scholes", BlackScholesEngine)  # these three calls run once, when this module is first imported
register_engine("monte_carlo", MonteCarloEngine)
register_engine("binomial", BinomialEngine)
