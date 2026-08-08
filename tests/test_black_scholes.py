import numpy as np
import pytest

from optionspricer.market import MarketData, OptionSpec, OptionType
from optionspricer.pricing.black_scholes import BlackScholesEngine, analytic_greeks, price


def test_put_call_parity():
    # C - P = S*e^{-qT} - K*e^{-rT}, for every (S, K, T, r, sigma, q). This must hold
    # exactly regardless of vol, since it follows from static replication, not from BS itself
    S, K, T, r, sigma, q = 100.0, 105.0, 0.75, 0.05, 0.3, 0.02
    call = price(S, K, T, r, sigma, OptionType.CALL, q)
    put = price(S, K, T, r, sigma, OptionType.PUT, q)
    assert call - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_price_matches_known_value():
    # Textbook example (Hull): S=42, K=40, T=0.5, r=0.10, sigma=0.20 -> call ~= 4.76
    assert price(42, 40, 0.5, 0.10, 0.20, OptionType.CALL) == pytest.approx(4.759, abs=1e-3)


def test_deep_itm_call_approaches_intrinsic_minus_discounted_strike():
    S, K, T, r, sigma = 1000.0, 10.0, 0.1, 0.05, 0.2
    p = price(S, K, T, r, sigma, OptionType.CALL)
    assert p == pytest.approx(S - K * np.exp(-r * T), rel=1e-3)


def test_deep_otm_put_is_near_zero():
    p = price(100, 10, 0.1, 0.05, 0.2, OptionType.PUT)
    assert 0 <= p < 1e-6


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_analytic_greeks_match_finite_difference(option_type):
    S, K, T, r, sigma, q = 100.0, 95.0, 0.4, 0.03, 0.25, 0.01
    analytic = analytic_greeks(S, K, T, r, sigma, option_type, q)

    h = 1e-4
    d_delta = (price(S + h, K, T, r, sigma, option_type, q) - price(S - h, K, T, r, sigma, option_type, q)) / (2 * h)
    d_vega = (price(S, K, T, r, sigma + h, option_type, q) - price(S, K, T, r, sigma - h, option_type, q)) / (2 * h) / 100
    d_rho = (price(S, K, T, r + h, sigma, option_type, q) - price(S, K, T, r - h, sigma, option_type, q)) / (2 * h) / 100
    d_theta = -(price(S, K, T + h, r, sigma, option_type, q) - price(S, K, T - h, r, sigma, option_type, q)) / (2 * h) / 365

    assert analytic.delta == pytest.approx(d_delta, abs=1e-5)
    assert analytic.vega == pytest.approx(d_vega, abs=1e-5)
    assert analytic.rho == pytest.approx(d_rho, abs=1e-5)
    assert analytic.theta == pytest.approx(d_theta, abs=1e-4)


def test_call_delta_bounded_0_1():
    for S in [50, 90, 100, 110, 200]:
        g = analytic_greeks(S, 100, 0.5, 0.05, 0.2, OptionType.CALL)
        assert 0.0 <= g.delta <= 1.0


def test_put_delta_bounded_neg1_0():
    for S in [50, 90, 100, 110, 200]:
        g = analytic_greeks(S, 100, 0.5, 0.05, 0.2, OptionType.PUT)
        assert -1.0 <= g.delta <= 0.0


def test_gamma_positive_and_equal_for_call_and_put():
    g_call = analytic_greeks(100, 100, 0.5, 0.05, 0.2, OptionType.CALL)
    g_put = analytic_greeks(100, 100, 0.5, 0.05, 0.2, OptionType.PUT)
    assert g_call.gamma > 0
    assert g_call.gamma == pytest.approx(g_put.gamma, rel=1e-10)  # gamma is identical for calls/puts by put-call parity


def test_engine_rejects_american_exercise():
    engine = BlackScholesEngine()
    opt = OptionSpec(strike=100, maturity=0.5, option_type=OptionType.CALL, exercise="american")
    market = MarketData(spot=100, rate=0.05, vol=0.2)
    with pytest.raises(ValueError, match="European"):
        engine.price(opt, market)
