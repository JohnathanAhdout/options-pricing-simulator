"""The Strategy pattern for pricing algorithms.

`PricingEngine` is the interface every pricing algorithm implements:
Black-Scholes, Monte Carlo, and a CRR binomial tree all answer to the same
two methods, `price` and `greeks`, so client code (an experiment script, a
strategy backtest) can swap one for another without changing a single line
around it. That's the entire point of the Strategy pattern here: the
*choice* of pricing algorithm becomes a runtime value (see `factory.py`)
instead of something wired into the call site.

Only Black-Scholes has a closed-form derivative of price with respect to
each input, so only it overrides `greeks()`. Every other engine inherits the
default implementation below, which gets the Greeks the same way a trading
desk would if it only had a pricing calculator and no formula sheet: bump
one input at a time and re-price. This is a Template Method: the base
class owns the *algorithm* for computing Greeks (bump, reprice, finite
difference), and subclasses only have to supply `price()`.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

from abc import ABC, abstractmethod  # ABC: base class that can't be instantiated directly; abstractmethod: marks a method subclasses MUST implement
from dataclasses import dataclass, replace  # replace: builds a modified copy of a frozen dataclass without mutating the original

from optionspricer.market import Greeks, MarketData, OptionSpec, PriceResult  # the shared value objects every engine passes around


@dataclass(frozen=True, slots=True)  # same immutability rationale as OptionSpec/MarketData in market.py
class GreekBumps:
    """Step sizes for finite-difference Greeks. Kept as absolute (not
    relative) bumps so gamma, a second difference divided by d_spot**2,
    doesn't blow up or vanish depending on the scale of the underlying."""

    d_spot: float = 1e-2  # $0.01 bump to spot, used for delta and gamma
    d_vol: float = 1e-4  # 0.01 vol-point bump, used for vega
    d_rate: float = 1e-4  # 0.01%-point bump, used for rho
    d_time: float = 1.0 / 365.0  # one calendar day, expressed in years, used for theta


class PricingEngine(ABC):  # every concrete engine (BlackScholesEngine, MonteCarloEngine, BinomialEngine) inherits from this
    """Common interface for every pricing algorithm in this package."""

    name: str  # class-level type annotation, not an assignment: each subclass must set its own string (e.g. "black_scholes")

    @abstractmethod  # subclasses that don't override this can't be instantiated at all; Python enforces it at construction time
    def price(self, option: OptionSpec, market: MarketData) -> PriceResult:
        """Fair value of `option` under `market`."""

    def greeks(  # NOT abstract: this default implementation is inherited unless a subclass overrides it (as BlackScholesEngine and BinomialEngine do)
        self,
        option: OptionSpec,
        market: MarketData,
        bumps: GreekBumps | None = None,
    ) -> Greeks:
        """Central finite-difference Greeks, built entirely out of `price()`.

        Delta and gamma come from bumping spot up and down by `d_spot` and
        combining the three prices (base, up, down) into a first and second
        difference. Vega and rho are the same idea on vol and rate. Theta
        uses a *forward* difference in time only: there's no "future"
        maturity to bump up to, since T only ever counts down, so it's the
        (negative of the) price change from letting one day of calendar time
        pass with everything else fixed.

        Simulation-based engines (Monte Carlo) should reseed identically for
        every call inside one `greeks()` invocation (see `monte_carlo.py`),
        so that the bumped and unbumped prices share the same random draws
        (common random numbers). Without that, the subtraction below would
        mostly be measuring sampling noise, not the actual derivative.
        """
        b = bumps or GreekBumps()  # use the caller's step sizes if given, otherwise the defaults above
        S, sigma, r = market.spot, market.vol, market.rate  # unpack once, since all five Greeks below need at least one of these

        p0 = self.price(option, market).price  # the unbumped, base-case price; reused by gamma and theta below
        p_up_s = self.price(option, replace(market, spot=S + b.d_spot)).price  # replace() builds a new MarketData with spot nudged up, everything else identical
        p_dn_s = self.price(option, replace(market, spot=S - b.d_spot)).price  # same, nudged down
        delta = (p_up_s - p_dn_s) / (2 * b.d_spot)  # central difference: the standard f'(x) approximation
        gamma = (p_up_s - 2 * p0 + p_dn_s) / (b.d_spot**2)  # central second difference: the standard f''(x) approximation

        vol_dn = max(sigma - b.d_vol, 1e-8)  # floor at a tiny positive vol so a near-zero-sigma option never gets bumped negative
        p_up_v = self.price(option, replace(market, vol=sigma + b.d_vol)).price
        p_dn_v = self.price(option, replace(market, vol=vol_dn)).price
        vega = (p_up_v - p_dn_v) / (sigma + b.d_vol - vol_dn) / 100.0  # denominator uses the ACTUAL gap (accounts for the floor above clamping vol_dn), not a blind 2*d_vol; /100 to report $ per vol point

        p_up_r = self.price(option, replace(market, rate=r + b.d_rate)).price
        p_dn_r = self.price(option, replace(market, rate=r - b.d_rate)).price
        rho = (p_up_r - p_dn_r) / (2 * b.d_rate) / 100.0  # same central-difference pattern as delta, scaled to $ per 1% rate move

        if option.maturity - b.d_time > 0:  # guard: OptionSpec requires a strictly positive maturity, so this must stay above zero
            p_next = self.price(replace(option, maturity=option.maturity - b.d_time), market).price  # reprice the SAME option one day closer to expiry
            theta = (p_next - p0) / b.d_time / 365.0  # forward difference only, since there's no "T + dt" to bump to; /365 converts the per-step change to $ per calendar day
        else:
            theta = 0.0  # option expires within one day: skip the bump entirely rather than construct an invalid (non-positive maturity) OptionSpec

        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
