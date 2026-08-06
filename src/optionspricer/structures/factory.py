"""Factory for named multi-leg structures. Same registry pattern as
`pricing/factory.py` and `implied_vol/factory.py`: each builder below prices
its legs with whatever `PricingEngine` it's handed, so
`create_structure("straddle", K=100, T=0.5, market=m, engine=binomial)` and
`create_structure("straddle", K=100, T=0.5, market=m, engine=black_scholes)`
build the same *shape* of position priced two different ways -- the
strategy (which legs) and the pricing algorithm (how each leg is valued)
are fully decoupled.
"""

from __future__ import annotations

from optionspricer.market import MarketData, OptionSpec, OptionType
from optionspricer.pricing.base import PricingEngine
from optionspricer.structures.base import OptionLeg, StockLeg, Structure


def _priced_leg(K: float, T: float, option_type: OptionType, market: MarketData, engine: PricingEngine, quantity: float) -> OptionLeg:
    opt = OptionSpec(strike=K, maturity=T, option_type=option_type)
    premium = engine.price(opt, market).price
    return OptionLeg(option=opt, quantity=quantity, entry_price=premium)


def build_long_call(K: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:
    return Structure("Long Call", option_legs=(_priced_leg(K, T, OptionType.CALL, market, engine, 1.0),))


def build_long_put(K: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:
    return Structure("Long Put", option_legs=(_priced_leg(K, T, OptionType.PUT, market, engine, 1.0),))


def build_covered_call(K: float, T: float, market: MarketData, engine: PricingEngine, S0: float | None = None) -> Structure:
    S0 = market.spot if S0 is None else S0
    return Structure(
        "Covered Call",
        option_legs=(_priced_leg(K, T, OptionType.CALL, market, engine, -1.0),),
        stock_legs=(StockLeg(1.0, S0),),
    )


def build_protective_put(K: float, T: float, market: MarketData, engine: PricingEngine, S0: float | None = None) -> Structure:
    S0 = market.spot if S0 is None else S0
    return Structure(
        "Protective Put",
        option_legs=(_priced_leg(K, T, OptionType.PUT, market, engine, 1.0),),
        stock_legs=(StockLeg(1.0, S0),),
    )


def build_straddle(K: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:
    return Structure(
        "Straddle",
        option_legs=(
            _priced_leg(K, T, OptionType.CALL, market, engine, 1.0),
            _priced_leg(K, T, OptionType.PUT, market, engine, 1.0),
        ),
    )


def build_strangle(K_put: float, K_call: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:
    return Structure(
        "Strangle",
        option_legs=(
            _priced_leg(K_call, T, OptionType.CALL, market, engine, 1.0),
            _priced_leg(K_put, T, OptionType.PUT, market, engine, 1.0),
        ),
    )


def build_bull_call_spread(K_low: float, K_high: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:
    return Structure(
        "Bull Call Spread",
        option_legs=(
            _priced_leg(K_low, T, OptionType.CALL, market, engine, 1.0),
            _priced_leg(K_high, T, OptionType.CALL, market, engine, -1.0),
        ),
    )


def build_short_straddle(K: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:
    return Structure(
        "Short Straddle",
        option_legs=(
            _priced_leg(K, T, OptionType.CALL, market, engine, -1.0),
            _priced_leg(K, T, OptionType.PUT, market, engine, -1.0),
        ),
    )


_REGISTRY = {
    "long_call": build_long_call,
    "long_put": build_long_put,
    "covered_call": build_covered_call,
    "protective_put": build_protective_put,
    "straddle": build_straddle,
    "strangle": build_strangle,
    "bull_call_spread": build_bull_call_spread,
    "short_straddle": build_short_straddle,
}


def create_structure(name: str, **kwargs) -> Structure:
    try:
        builder = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown structure {name!r}; available: {sorted(_REGISTRY)}") from None
    return builder(**kwargs)


def available_structures() -> list[str]:
    return sorted(_REGISTRY)
