"""Factory for named multi-leg structures. Same registry pattern as
`pricing/factory.py` and `implied_vol/factory.py`: each builder below prices
its legs with whatever `PricingEngine` it's handed, so
`create_structure("straddle", K=100, T=0.5, market=m, engine=binomial)` and
`create_structure("straddle", K=100, T=0.5, market=m, engine=black_scholes)`
build the same *shape* of position priced two different ways: the
strategy (which legs) and the pricing algorithm (how each leg is valued)
are fully decoupled.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

from optionspricer.market import MarketData, OptionSpec, OptionType  # the shared value objects every builder constructs and reads
from optionspricer.pricing.base import PricingEngine  # the interface used to price each leg at construction time
from optionspricer.structures.base import OptionLeg, StockLeg, Structure  # the types every builder assembles


def _priced_leg(K: float, T: float, option_type: OptionType, market: MarketData, engine: PricingEngine, quantity: float) -> OptionLeg:  # shared helper: every build_* function below calls this instead of constructing OptionLeg by hand
    opt = OptionSpec(strike=K, maturity=T, option_type=option_type)  # the contract terms, validated by OptionSpec.__post_init__ (see market.py)
    premium = engine.price(opt, market).price  # priced NOW, at construction time: this is what makes entry_price the actual entry price, not a stale placeholder
    return OptionLeg(option=opt, quantity=quantity, entry_price=premium)  # quantity's sign (passed in by the caller below) determines long vs. short


def build_long_call(K: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:  # the simplest structure: one leg, quantity +1.0 (long)
    return Structure("Long Call", option_legs=(_priced_leg(K, T, OptionType.CALL, market, engine, 1.0),))  # trailing comma: a one-element tuple, not a parenthesized expression


def build_long_put(K: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:  # same shape as build_long_call, PUT instead of CALL
    return Structure("Long Put", option_legs=(_priced_leg(K, T, OptionType.PUT, market, engine, 1.0),))


def build_covered_call(K: float, T: float, market: MarketData, engine: PricingEngine, S0: float | None = None) -> Structure:  # short call + long stock: caps upside, collects premium
    S0 = market.spot if S0 is None else S0  # defaults the stock leg's entry price to today's spot unless the caller specifies a different cost basis
    return Structure(
        "Covered Call",
        option_legs=(_priced_leg(K, T, OptionType.CALL, market, engine, -1.0),),  # -1.0: SHORT the call, collecting premium
        stock_legs=(StockLeg(1.0, S0),),  # +1.0: long the underlying share, the "covered" part
    )


def build_protective_put(K: float, T: float, market: MarketData, engine: PricingEngine, S0: float | None = None) -> Structure:  # long put + long stock: insurance against the stock falling
    S0 = market.spot if S0 is None else S0  # same defaulting logic as build_covered_call
    return Structure(
        "Protective Put",
        option_legs=(_priced_leg(K, T, OptionType.PUT, market, engine, 1.0),),  # +1.0: LONG the put, the insurance leg
        stock_legs=(StockLeg(1.0, S0),),  # +1.0: long the underlying share being insured
    )


def build_straddle(K: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:  # long call + long put, same strike: a bet on a big move, direction unknown
    return Structure(
        "Straddle",
        option_legs=(
            _priced_leg(K, T, OptionType.CALL, market, engine, 1.0),  # +1.0: long call
            _priced_leg(K, T, OptionType.PUT, market, engine, 1.0),  # +1.0: long put, same strike K
        ),
    )


def build_strangle(K_put: float, K_call: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:  # like a straddle, but the two strikes differ: cheaper, needs a bigger move to profit
    return Structure(
        "Strangle",
        option_legs=(
            _priced_leg(K_call, T, OptionType.CALL, market, engine, 1.0),  # +1.0: long call at the (typically higher) call strike
            _priced_leg(K_put, T, OptionType.PUT, market, engine, 1.0),  # +1.0: long put at the (typically lower) put strike
        ),
    )


def build_bull_call_spread(K_low: float, K_high: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:  # long the cheaper call, short the pricier one: caps both cost and payoff
    return Structure(
        "Bull Call Spread",
        option_legs=(
            _priced_leg(K_low, T, OptionType.CALL, market, engine, 1.0),  # +1.0: long call at the lower strike
            _priced_leg(K_high, T, OptionType.CALL, market, engine, -1.0),  # -1.0: short call at the higher strike, funding part of the premium
        ),
    )


def build_short_straddle(K: float, T: float, market: MarketData, engine: PricingEngine) -> Structure:  # short call + short put, same strike: the flagship position hedging.py delta-hedges in the gamma/theta experiments
    return Structure(
        "Short Straddle",
        option_legs=(
            _priced_leg(K, T, OptionType.CALL, market, engine, -1.0),  # -1.0: short call
            _priced_leg(K, T, OptionType.PUT, market, engine, -1.0),  # -1.0: short put, same strike K: negative gamma, positive theta, per BACKGROUND.md
        ),
    )


_REGISTRY = {  # name -> builder function, populated as a literal dict rather than via register_*() calls, since every entry is known upfront
    "long_call": build_long_call,
    "long_put": build_long_put,
    "covered_call": build_covered_call,
    "protective_put": build_protective_put,
    "straddle": build_straddle,
    "strangle": build_strangle,
    "bull_call_spread": build_bull_call_spread,
    "short_straddle": build_short_straddle,
}


def create_structure(name: str, **kwargs) -> Structure:  # **kwargs forwards builder args, e.g. K=100, T=0.5, market=m, engine=e
    try:
        builder = _REGISTRY[name]  # KeyError here means the name doesn't match any build_* function above
    except KeyError:
        raise ValueError(f"unknown structure {name!r}; available: {sorted(_REGISTRY)}") from None  # `from None` suppresses the KeyError traceback, since it's not useful context for the caller
    return builder(**kwargs)  # calls the matched build_* function, passing through whatever kwargs the caller supplied


def available_structures() -> list[str]:  # what a caller can loop over to build every named strategy generically
    return sorted(_REGISTRY)  # sorted() on a dict iterates its keys; alphabetical order makes output reproducible
