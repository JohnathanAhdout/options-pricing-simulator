import numpy as np
import pytest

from optionspricer.implied_vol import BrentSolver, JaeckelSolver, NewtonSolver
from optionspricer.market import OptionType
from optionspricer.pricing.black_scholes import price as bs_price

CASES = [
    ("ATM call", 100, 100, 0.5, 0.05, 0.20, OptionType.CALL, 0.0),
    ("ATM put", 100, 100, 0.5, 0.05, 0.20, OptionType.PUT, 0.0),
    ("Deep OTM call", 100, 130, 0.1, 0.05, 0.30, OptionType.CALL, 0.0),
    ("Deep OTM put", 100, 70, 0.1, 0.05, 0.30, OptionType.PUT, 0.0),
    ("Near expiry", 100, 100, 0.01, 0.05, 0.25, OptionType.CALL, 0.0),
    ("Long-dated", 100, 100, 2.0, 0.05, 0.15, OptionType.CALL, 0.0),
    ("Deep ITM call", 100, 70, 0.5, 0.05, 0.20, OptionType.CALL, 0.0),
    ("With dividend yield", 100, 100, 1.0, 0.05, 0.20, OptionType.CALL, 0.03),
    ("Put with dividend", 100, 100, 1.0, 0.05, 0.20, OptionType.PUT, 0.03),
]


@pytest.mark.parametrize("solver_cls", [BrentSolver, JaeckelSolver])
@pytest.mark.parametrize("name,S,K,T,r,sigma,otype,q", CASES, ids=[c[0] for c in CASES])
def test_robust_solvers_recover_true_vol(solver_cls, name, S, K, T, r, sigma, otype, q):
    market_price = bs_price(S, K, T, r, sigma, otype, q)
    iv = solver_cls().solve(market_price, S, K, T, r, otype, q)
    assert iv == pytest.approx(sigma, abs=1e-4)


@pytest.mark.parametrize("name,S,K,T,r,sigma,otype,q", CASES, ids=[c[0] for c in CASES])
def test_newton_recovers_true_vol_in_well_conditioned_cases(name, S, K, T, r, sigma, otype, q):
    if name == "Deep OTM put":
        pytest.skip("known Newton failure mode -- see test_newton_fails_on_near_zero_vega")
    market_price = bs_price(S, K, T, r, sigma, otype, q)
    iv = NewtonSolver().solve(market_price, S, K, T, r, otype, q)
    assert iv == pytest.approx(sigma, abs=1e-4)


def test_newton_fails_on_near_zero_vega():
    """Pins down a real failure mode rather than papering over it: a deep
    OTM put close to expiry has vega near zero, so Newton's division by
    vega blows the iterate up to a nonsensical volatility. Brent and
    Jaeckel (which falls back to Brent) both stay correct on the exact
    same input -- this is why the package defaults to Jaeckel for surface
    construction instead of Newton."""
    S, K, T, r, sigma = 100, 70, 0.1, 0.05, 0.30
    market_price = bs_price(S, K, T, r, sigma, OptionType.PUT)

    newton_iv = NewtonSolver().solve(market_price, S, K, T, r, OptionType.PUT)
    brent_iv = BrentSolver().solve(market_price, S, K, T, r, OptionType.PUT)
    jaeckel_iv = JaeckelSolver().solve(market_price, S, K, T, r, OptionType.PUT)

    assert abs(newton_iv - sigma) > 1.0  # Newton is not just imprecise here, it's nonsensical
    assert brent_iv == pytest.approx(sigma, abs=1e-4)
    assert jaeckel_iv == pytest.approx(sigma, abs=1e-4)


def test_root_is_unique_price_strictly_increasing_in_vol():
    # the whole premise of a root-finder existing is that BS price is
    # monotone in sigma (vega > 0 everywhere); spot-check that directly
    S, K, T, r = 100, 100, 0.5, 0.05
    prices = [bs_price(S, K, T, r, sigma, OptionType.CALL) for sigma in np.linspace(0.01, 2.0, 50)]
    assert all(a < b for a, b in zip(prices, prices[1:]))
