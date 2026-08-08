"""Newton-Raphson: use the slope (vega) to jump straight at the root.

sigma_{n+1} = sigma_n - f(sigma_n) / f'(sigma_n), where f(sigma) is the
pricing error `bs_price(sigma) - market_price` and f'(sigma) is vega. Near
the root this converges *quadratically*: each step roughly squares the
number of correct decimal digits, which is the fastest of the three
solvers here when it works. Its one failure mode is exactly where vega is
small: deep out-of-the-money or very close to expiry, where the price
curve is nearly flat in sigma, so a tiny pricing error implies a huge step
and the iteration can fly off to a nonsensical sigma. `BrentSolver` exists
precisely to have something safe to fall back on in that regime.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from optionspricer.implied_vol.base import IVSolver
from optionspricer.market import OptionType
from optionspricer.pricing.black_scholes import d1, price as bs_price


def solve(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    q: float = 0.0,
    sigma0: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    sigma = sigma0  # start from a generic guess; unlike Jaeckel there's no attempt at a smart initial value here
    for _ in range(max_iter):
        model_price = bs_price(S, K, T, r, sigma, option_type, q)
        diff = model_price - market_price  # f(sigma) = model price at the current guess, minus what we're trying to match
        if abs(diff) < tol:
            return sigma  # close enough: f(sigma) ~= 0
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1(S, K, T, r, sigma, q))  # f'(sigma), i.e. raw (unscaled) vega
        if abs(vega) < 1e-12:
            break  # curve is locally flat here; dividing by ~0 next would send sigma flying, so bail out instead
        sigma -= diff / vega  # the Newton step itself: sigma_{n+1} = sigma_n - f(sigma_n)/f'(sigma_n)
        sigma = max(sigma, 1e-8)  # keep sigma positive: d1/d2 are undefined at sigma <= 0
    return sigma


class NewtonSolver(IVSolver):
    name = "newton"

    def __init__(self, sigma0: float = 0.2, tol: float = 1e-8, max_iter: int = 50):
        self.sigma0 = sigma0
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, market_price: float, S: float, K: float, T: float, r: float, option_type: OptionType, q: float = 0.0) -> float:
        return solve(market_price, S, K, T, r, option_type, q, self.sigma0, self.tol, self.max_iter)
