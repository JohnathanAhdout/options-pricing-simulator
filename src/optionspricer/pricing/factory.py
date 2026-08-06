"""Factory for pricing engines: turns a name (a string that could come from
a config file, a CLI flag, or a sweep over `["black_scholes", "monte_carlo",
"binomial"]`) into a live `PricingEngine`.

It's a registry rather than an if/elif chain on purpose: adding a new engine
means calling `register_engine` next to the class definition, not editing
this file. That's the open/closed principle in practice -- this module is
closed for modification but open for extension -- and it's what lets
experiments/*.py iterate over "every registered engine" generically instead
of hardcoding a list that has to be kept in sync by hand.
"""

from __future__ import annotations

from optionspricer.pricing.base import PricingEngine
from optionspricer.pricing.binomial import BinomialEngine
from optionspricer.pricing.black_scholes import BlackScholesEngine
from optionspricer.pricing.monte_carlo import MonteCarloEngine

_REGISTRY: dict[str, type[PricingEngine]] = {}


def register_engine(name: str, engine_cls: type[PricingEngine]) -> None:
    _REGISTRY[name] = engine_cls


def create_pricing_engine(name: str, **kwargs) -> PricingEngine:
    """Build a `PricingEngine` by name, e.g. `create_pricing_engine("binomial", n_steps=500)`."""
    try:
        engine_cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown pricing engine {name!r}; available: {sorted(_REGISTRY)}") from None
    return engine_cls(**kwargs)


def available_engines() -> list[str]:
    return sorted(_REGISTRY)


register_engine("black_scholes", BlackScholesEngine)
register_engine("monte_carlo", MonteCarloEngine)
register_engine("binomial", BinomialEngine)
