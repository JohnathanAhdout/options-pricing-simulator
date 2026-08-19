"""Black-Scholes-Merton closed-form pricing and analytic Greeks.

Every formula below is derived and dissected term-by-term in
`BACKGROUND.md`. This module intentionally keeps two layers apart:

1. A **functional core**, `d1`, `d2`, `price`, `analytic_greeks`: plain
   functions of plain floats, with no state and no side effects. These are
   the formulas, full stop. You can unit-test them, call them from a notebook,
   or reuse them inside the Monte Carlo and binomial engines (e.g. for
   comparison plots) without ever touching the class below.
2. A thin **imperative shell**, `BlackScholesEngine`: adapts the functional
   core to the `PricingEngine` interface so it can be swapped in wherever a
   pricing algorithm is expected.

Only European exercise is supported: the closed form assumes exercise is
possible exactly once, at T. American-style early exercise breaks the
derivation, so `BinomialEngine` exists for that case.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

import numpy as np  # vectorized math: exp, log, sqrt
from scipy.stats import norm  # the standard normal distribution: norm.cdf (N) and norm.pdf (phi) below

from optionspricer.market import Greeks, MarketData, OptionSpec, OptionType, PriceResult  # the shared value objects this module reads and returns
from optionspricer.pricing.base import PricingEngine  # the interface BlackScholesEngine implements


def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:  # S=spot, K=strike, T=years to expiry, r=rate, sigma=vol, q=dividend yield
    """Standardized log-moneyness of the stock leg, in units of one
    standard deviation of log-return. See BACKGROUND.md for the derivation
    of why the drift term is (r - q + sigma^2/2), not just (r - q)."""
    # log(S/K): log-moneyness, 0 exactly at-the-money, positive if S > K.
    # (r - q + sigma^2/2)*T: risk-neutral drift of log(S) over [0, T], including
    #   the Ito correction +sigma^2/2. Because log is concave, E[log S_T] under
    #   the risk-neutral measure is (r - q - sigma^2/2)*T, not (r - q)*T, and d1
    #   needs the other sign convention (+sigma^2/2) by construction. See
    #   BACKGROUND.md for exactly where that flips relative to `simulation.py`.
    # dividing by sigma*sqrt(T) (the std dev of log-return over [0, T]) turns the
    #   whole thing into a z-score, which is what lets norm.cdf below be used at all.
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))  # the d1 formula, computed in one expression


def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:  # same five market/contract inputs as d1
    """d1 minus one standard deviation of log-return; N(d2) is the
    risk-neutral probability the option finishes in the money."""
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)  # reuses d1 rather than recomputing log(S/K) from scratch


def price(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType, q: float = 0.0) -> float:  # returns a plain float, not a PriceResult: that wrapping happens in the class below
    """Fair value of a European option under Black-Scholes-Merton (with a
    continuous dividend yield q)."""
    D1, D2 = d1(S, K, T, r, sigma, q), d2(S, K, T, r, sigma, q)  # compute both standardized distances once, reused by call and put branches alike
    disc_r = np.exp(-r * T)  # present value of $1 paid at T, discounts the strike leg
    disc_q = np.exp(-q * T)  # present value of 1 share's dividend leakage, discounts the stock leg
    if option_type == OptionType.CALL:  # branch on contract type; the two formulas are mirror images, not the same expression negated
        # S*disc_q*N(D1): PV of the stock you receive, conditional on finishing ITM, probability-weighted
        # K*disc_r*N(D2): PV of the strike you pay on exercise, weighted by the risk-neutral P(ITM)
        return S * disc_q * norm.cdf(D1) - K * disc_r * norm.cdf(D2)  # the call price, exits the function here
    # put = the mirror image: N(-x) = 1 - N(x), so this is exactly put-call parity rearranged
    return K * disc_r * norm.cdf(-D2) - S * disc_q * norm.cdf(-D1)  # reached only when option_type is PUT


def analytic_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType, q: float = 0.0) -> Greeks:  # differentiates price() with respect to each input, in closed form
    """Exact partial derivatives of `price` with respect to (S, sigma, t, r).
    Vega and rho are scaled to $-per-1%-move; theta is scaled to $-per-day.
    Every term here is dissected in BACKGROUND.md."""
    D1, D2 = d1(S, K, T, r, sigma, q), d2(S, K, T, r, sigma, q)  # same two standardized distances as price(), needed again here
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)  # same two discount factors as price(), packed onto one line since both are used identically below
    pdf_d1 = norm.pdf(D1)  # the bell-curve height at D1. Shows up in gamma, vega, and the first term of theta because
    # differentiating N(D1(S)) or N(D1(sigma)) always pulls out phi(D1) via the chain rule (d/dx N(f(x)) = phi(f(x))*f'(x))

    # gamma is IDENTICAL for calls and puts (a consequence of put-call parity: C - P is
    # linear in S, so its second derivative is exactly zero, meaning d^2C/dS^2 = d^2P/dS^2)
    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))  # computed once, above the call/put branch, since it doesn't depend on option_type
    # vega too is identical for calls/puts, by the same parity argument (dC/dsigma - dP/dsigma = 0)
    vega = S * disc_q * np.sqrt(T) * pdf_d1 / 100.0  # /100: report $ per 1 vol POINT (0.01), not per unit of sigma

    if option_type == OptionType.CALL:  # delta, theta, and rho DO differ between calls and puts, unlike gamma/vega above
        delta = disc_q * norm.cdf(D1)  # roughly the risk-neutral P(ITM), discounted for the dividend leakage between now and T
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))  # "gamma rent": cost of carrying convexity, always < 0
            - r * K * disc_r * norm.cdf(D2)  # financing cost of the strike payment shrinking as T falls, pulls theta down further
            + q * S * disc_q * norm.cdf(D1)  # dividends the call holder is missing out on vs. owning the stock outright, pulls theta up
        ) / 365.0  # annual theta -> $ per calendar day, since desks always quote decay per day, not per year
        rho = K * T * disc_r * norm.cdf(D2) / 100.0  # positive: higher r raises the forward S*e^{(r-q)T}, so calls get more valuable
    else:  # option_type is PUT; every formula below is the call's mirror image, not its negation
        delta = disc_q * (norm.cdf(D1) - 1.0)  # = -disc_q*N(-D1), negative since puts gain when S falls
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))  # same gamma-rent term, convexity costs the same regardless of call/put
            + r * K * disc_r * norm.cdf(-D2)  # sign flips vs. call: the put holder RECEIVES K on exercise, so its rising PV helps here
            - q * S * disc_q * norm.cdf(-D1)  # sign flips vs. call: no missed dividends when short the (implicit) stock exposure
        ) / 365.0
        rho = -K * T * disc_r * norm.cdf(-D2) / 100.0  # negative: higher r hurts puts, mirroring the call case

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)  # bundle all five into the shared output type


class BlackScholesEngine(PricingEngine):  # the imperative shell: adapts the functions above to the common PricingEngine interface
    """European options, closed form. The reference engine every other
    engine gets checked against."""

    name = "black_scholes"  # the string every factory/experiment uses to select this engine

    def price(self, option: OptionSpec, market: MarketData) -> PriceResult:  # note: shadows the module-level price() function above; called as self.price or BlackScholesEngine().price
        if option.is_american:  # this engine has no early-exercise logic at all; refuse rather than silently mispricing
            raise ValueError(
                "BlackScholesEngine only prices European exercise; "
                "use BinomialEngine for American options."
            )
        p = price(market.spot, option.strike, option.maturity, market.rate, market.vol, option.option_type, market.dividend_yield)  # delegates to the module-level function, unpacking the two dataclasses into plain floats
        return PriceResult(price=float(p), engine=self.name)  # wraps the raw float with provenance (which engine produced it); stderr defaults to None since this is exact, not sampled

    def greeks(self, option: OptionSpec, market: MarketData, bumps=None) -> Greeks:  # overrides PricingEngine's default finite-difference greeks() with the exact closed form
        if option.is_american:  # same restriction as price() above, checked independently since greeks() can be called without price() first
            raise ValueError(
                "BlackScholesEngine only prices European exercise; "
                "use BinomialEngine for American options."
            )
        return analytic_greeks(
            market.spot, option.strike, option.maturity, market.rate, market.vol, option.option_type, market.dividend_yield
        )  # `bumps` is accepted but ignored: it only matters for the finite-difference default this method replaces
