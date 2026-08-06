import numpy as np
import pytest

from optionspricer.implied_vol import JaeckelSolver
from optionspricer.market import OptionType
from optionspricer.pricing.black_scholes import price as bs_price
from optionspricer.surface import Quote, implied_vols, smile_at_maturity, surface_grid, term_structure

S0, R, Q = 100.0, 0.05, 0.0
SOLVER = JaeckelSolver()


def _fake_smile(K: float, T: float) -> float:
    m = np.log(K / S0)
    return 0.20 + 0.15 * m**2 - 0.02 * m


def _synthetic_quotes(maturities=(0.1, 0.25, 0.5, 1.0), strikes=np.linspace(80, 120, 15)):
    quotes = []
    for T in maturities:
        for K in strikes:
            sigma = _fake_smile(K, T)
            quotes.append(Quote(strike=K, maturity=T, mid_price=bs_price(S0, K, T, R, sigma, OptionType.CALL, Q), option_type=OptionType.CALL))
    return quotes


def test_recovers_every_quote_when_all_are_valid():
    quotes = _synthetic_quotes()
    points = implied_vols(quotes, S0, R, SOLVER, Q)
    assert len(points) == len(quotes)


def test_recovers_exact_synthetic_smile():
    quotes = _synthetic_quotes()
    points = implied_vols(quotes, S0, R, SOLVER, Q)
    strikes, ivs = smile_at_maturity(points, 0.5)
    expected = np.array([_fake_smile(k, 0.5) for k in strikes])
    np.testing.assert_allclose(ivs, expected, atol=1e-6)


def test_below_intrinsic_quotes_are_dropped():
    bad_quote = Quote(strike=50.0, maturity=0.5, mid_price=0.01, option_type=OptionType.CALL)  # far below intrinsic ~= S - K
    points = implied_vols([bad_quote], S0, R, SOLVER, Q)
    assert points == []


def test_term_structure_has_one_point_per_maturity():
    quotes = _synthetic_quotes(maturities=(0.1, 0.5, 1.0))
    points = implied_vols(quotes, S0, R, SOLVER, Q)
    maturities, ivs = term_structure(points)
    assert list(maturities) == [0.1, 0.5, 1.0]
    assert len(ivs) == 3


def test_surface_grid_matches_smile_shape_reasonably():
    quotes = _synthetic_quotes()
    points = implied_vols(quotes, S0, R, SOLVER, Q)
    K, T, IV = surface_grid(points, n_strikes=30, n_maturities=20)
    valid = ~np.isnan(IV)
    assert valid.mean() > 0.8
    # ATM (near K=100) should have lower IV than deep in either wing, given the fake convex smile
    atm_col = np.argmin(np.abs(K[0] - 100))
    wing_col = 0
    assert np.nanmean(IV[:, atm_col]) < np.nanmean(IV[:, wing_col])
