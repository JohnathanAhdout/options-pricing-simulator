"""Jaeckel's method: a rational initial guess plus Halley's iteration.

Two ideas, applied in sequence:

1. **Normalize the problem.** Rewrite everything in terms of the forward
   price F = S*e^{(r-q)T}, log-forward-moneyness x = ln(F/K), and total
   variance theta = sigma*sqrt(T). Every Black-Scholes call price collapses
   to a function of exactly two dimensionless numbers, (x, theta), instead
   of five separate (S, K, T, r, sigma). That collapse is what makes it
   possible to write down a closed-form, *good* initial guess for theta
   directly from the normalized price -- see `_initial_theta` below -- so
   the iteration that follows starts close enough to the root that a
   handful of steps is enough, regardless of how deep in/out-of-the-money
   the option is.

2. **Iterate with Halley's method instead of Newton's.** Halley's update
   uses the price's curvature in sigma (f'') on top of its slope (f'),
   which gives *cubic* convergence -- each step roughly triples the number
   of correct digits, versus doubling for Newton -- and the curvature term
   also damps the step near-automatically, so it doesn't blow up the way
   Newton can when vega is small.

If Halley still fails to converge within its iteration budget (can happen
on genuinely pathological inputs), this falls back to `BrentSolver`, which
is slower but can't fail to bracket a root that's known to exist.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from optionspricer.implied_vol.base import IVSolver
from optionspricer.implied_vol.brent import solve as brent_solve
from optionspricer.market import OptionType
from optionspricer.pricing.black_scholes import price as bs_price


def _initial_theta(x: float, beta_tv: float) -> float:
    """Closed-form starting guess for theta = sigma*sqrt(T), inverting the
    small- and large-|x| asymptotics of the normalized price. Derived in
    full in BACKGROUND.md; the three branches are the ATM, OTM, and ITM
    regimes of the normalized Black-Scholes formula."""
    eps = 0.01
    if abs(x) < eps:
        # ATM: both N(d1), N(d2) ~= 0.5, and beta_tv ~= theta / sqrt(2 pi)
        return np.sqrt(2.0 * np.pi) * beta_tv
    if x < 0:
        # OTM call: N(d2) ~= 0, beta_tv driven by the N(d1) term alone
        u = np.clip(beta_tv * np.exp(-x / 2), 1e-12, 1 - 1e-12)
        y = norm.ppf(u)  # ~= d1 = x/theta + theta/2
        return y + np.sqrt(max(y**2 - 2 * x, 0.0))  # solves theta^2 - 2*y*theta - 2x = 0
    # ITM call: N(d1) ~= 1, beta_tv driven by the N(-d2) term
    u = np.clip(beta_tv * np.exp(x / 2), 1e-12, 1 - 1e-12)
    y = norm.ppf(u)  # ~= -d2 = -x/theta + theta/2
    return y + np.sqrt(max(y**2 + 2 * x, 0.0))


def solve(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    q: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 10,
) -> float:
    market_price_orig, option_type_orig = market_price, option_type

    if option_type == OptionType.PUT:
        # put-call parity via the forward: C = P + disc*(F - K); implied vol is
        # identical for a put and its parity-equivalent call at the same (K, T)
        market_price = market_price + S * np.exp(-q * T) - K * np.exp(-r * T)

    F = S * np.exp((r - q) * T)  # forward price: the risk-neutral expectation of S_T, E^Q[S_T] = F
    discount = np.exp(-r * T)  # PV of $1 at T
    beta = market_price / (discount * np.sqrt(F * K))  # normalized price -- depends only on (x, theta), not on S/K/T/r separately
    x = np.log(F / K)  # log-forward-moneyness: 0 exactly ATM (forward), <0 OTM call, >0 ITM call

    beta_intrinsic = max(np.exp(x / 2) - np.exp(-x / 2), 0.0)  # normalized intrinsic value = 2*sinh(x/2) for x>0, else 0
    beta_tv = beta - beta_intrinsic  # normalized TIME value -- the part that actually depends on sigma; this is what we invert for
    if beta_tv <= 0:
        return 1e-8  # price is at or below intrinsic: the market is implying essentially zero vol (or a bad/stale quote)

    theta = np.clip(abs(_initial_theta(x, beta_tv)), 1e-6, 5.0 * np.sqrt(T)) if T > 0 else 1e-6  # closed-form starting guess for total vol theta = sigma*sqrt(T)
    sigma = np.clip(theta / np.sqrt(T), 1e-6, 5.0)  # convert theta back to a per-year sigma, clipped to a sane range

    for _ in range(max_iter):  # Halley's method: cubic convergence, so this budget is generous -- 3-5 steps is typical
        model_price = bs_price(S, K, T, r, sigma, OptionType.CALL, q)  # always price in call space; puts were already converted via parity above
        diff = model_price - market_price  # f(sigma): how far the current guess is from the target price
        if abs(diff) < tol:
            return sigma
        d1_ = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))  # d1 at the current sigma guess, needed for both f' and f''
        d2_ = d1_ - sigma * np.sqrt(T)
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1_)  # f'(sigma) = raw vega
        if abs(vega) < 1e-12:
            break  # flat curve here -- Halley's correction term below would divide by ~0; bail to the Brent fallback instead
        vega2 = vega * d1_ * d2_ / sigma  # f'' = d(vega)/d(sigma), standard BS second-derivative result
        # Halley's update = the plain Newton step (diff/vega), divided by a curvature correction; when
        # f is exactly linear (f''=0) this correction is 1 and Halley collapses to Newton exactly --
        # the curvature term is what damps the step and prevents Newton-style overshoot near-zero-vega
        denom = 1.0 - (diff * vega2) / (2.0 * vega**2)
        sigma -= (diff / vega) / denom if abs(denom) > 1e-12 else diff / vega
        sigma = max(sigma, 1e-8)

    if abs(bs_price(S, K, T, r, sigma, OptionType.CALL, q) - market_price) > tol * 100:
        return brent_solve(market_price_orig, S, K, T, r, option_type_orig, q, tol=tol)
    return sigma


class JaeckelSolver(IVSolver):
    name = "jaeckel"

    def __init__(self, tol: float = 1e-10, max_iter: int = 10):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, market_price: float, S: float, K: float, T: float, r: float, option_type: OptionType, q: float = 0.0) -> float:
        return solve(market_price, S, K, T, r, option_type, q, self.tol, self.max_iter)
