"""Brent's method (1973): bisection's guarantee, secant/IQI's speed.

Maintains a bracket [a, b] with f(a) and f(b) of opposite sign, so the root
is trapped inside no matter what. That's bisection's guarantee, and it's
why this solver never needs a starting guess or can fly off to a bad sigma
the way Newton can. But instead of always bisecting, it first tries a
faster step: the secant line through the two most recent points, or (once
three points are available) inverse quadratic interpolation, fitting a
parabola through the last three (sigma, f(sigma)) pairs in the "sigma as a
function of f" direction, and jumping to where that parabola hits zero.
If the fast step would land outside the safe part of the bracket, or isn't
converging quickly enough, Brent falls back to a plain bisection step for
that iteration. The result is guaranteed convergence with superlinear
speed in practice (roughly the golden-ratio rate, ~1.62 extra correct
digits per step, at best), slower per step than Newton near a nice root,
but immune to Newton's blow-up when vega is tiny.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

from optionspricer.implied_vol.base import IVSolver  # the interface BrentSolver implements
from optionspricer.market import OptionType
from optionspricer.pricing.black_scholes import price as bs_price  # aliased to avoid shadowing this module's own solve()


def solve(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 200,  # much larger than Newton's default: bisection steps alone converge only linearly
) -> float:
    def f(sigma: float) -> float:  # local closure over S, K, T, r, option_type, q, market_price, so callers below can just write f(x)
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price  # the pricing error at a candidate sigma: this is the root-finding target

    low, high = 1e-4, 5.0  # BS price is monotone in sigma, so any real market price has exactly one root in [0.01%, 500%] vol
    a, fa = low, f(low)  # a: one end of the bracket
    b, fb = high, f(high)  # b: the other end, and always our current *best* estimate of the root (see the swap below)
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa  # enforce the invariant |f(b)| <= |f(a)|: b is always the "closer to zero" endpoint
    c, fc = a, fa  # c: the previous value of b, kept around so inverse quadratic interpolation has 3 distinct points to fit
    mflag = True  # whether the *last* step taken was a plain bisection, feeds the two "cond_slow_vs_*" progress checks below
    d = 0.0  # c from two iterations ago; only used by the (not mflag) progress check

    for _ in range(max_iter):
        if abs(fb) < tol or abs(b - a) < tol:
            return b  # converged: either f(b) is close enough to zero, or the bracket itself has shrunk to nothing

        if fa != fc and fb != fc:
            # inverse quadratic interpolation (IQI): fit the unique parabola through (a,fa), (b,fb), (c,fc)
            # in the "sigma as a function of f" direction, and jump straight to where that parabola crosses f=0
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # only two distinct f-values available (first iteration, or a degenerate step), so fall back to
            # the secant method: draw a straight line through (a,fa) and (b,fb), use where it crosses zero
            s = b - fb * (b - a) / (fb - fa)

        lo, hi = sorted(((3 * a + b) / 4, b))  # the "safe zone" a fast step must land in: the inner quarter of the bracket nearest b
        cond_outside_bracket = not (lo < s < hi)  # the IQI/secant step overshot outside that safe zone, can't trust it
        cond_slow_vs_bisection = mflag and abs(s - b) >= abs(b - c) / 2  # last step was bisection, but this step isn't converging at least as fast, not worth it
        cond_slow_vs_prior = (not mflag) and abs(s - b) >= abs(c - d) / 2  # same progress check, for when the last step *wasn't* a bisection
        cond_bracket_tiny = mflag and abs(b - c) < tol  # bracket's basically a point already; another fast step risks numerical garbage
        cond_prior_tiny = (not mflag) and abs(c - d) < tol

        if cond_outside_bracket or cond_slow_vs_bisection or cond_slow_vs_prior or cond_bracket_tiny or cond_prior_tiny:
            s, mflag = (a + b) / 2, True  # any rejection reason -> fall back to plain bisection, which always halves the bracket safely
        else:
            mflag = False  # the fast step was accepted; remember that for next iteration's progress check

        fs = f(s)  # the ONE new function evaluation this iteration spends, regardless of which branch chose s
        d, c, fc = c, b, fb  # shift the 3-point history forward: this iteration's b/fb become next iteration's c/fc
        if fa * fs < 0:
            b, fb = s, fs  # f changes sign between a and s, so the root is in [a, s] and s becomes the new "close" endpoint b
        else:
            a, fa = s, fs  # otherwise the root is in [s, b], so s replaces the far endpoint a instead
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa  # re-enforce |f(b)| <= |f(a)| before the next iteration starts

    return b  # max_iter exhausted without hitting the tolerance check; return the best endpoint found so far


class BrentSolver(IVSolver):  # the imperative shell: adapts the solve() function above to the common IVSolver interface
    name = "brent"  # the string every factory/experiment uses to select this solver

    def __init__(self, tol: float = 1e-8, max_iter: int = 200):  # no starting-guess parameter, unlike NewtonSolver: Brent never needs one
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, market_price: float, S: float, K: float, T: float, r: float, option_type: OptionType, q: float = 0.0) -> float:  # note: shadows the module-level solve() function above
        return solve(market_price, S, K, T, r, option_type, q, self.tol, self.max_iter)  # delegates to the module-level function, filling in the instance's stored settings
