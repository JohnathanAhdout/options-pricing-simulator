import numpy as np
import pytest

from optionspricer.market import MarketData
from optionspricer.pricing.black_scholes import BlackScholesEngine
from optionspricer.structures import create_structure, mark_to_market, payoff_at_expiry, portfolio_greeks

ENGINE = BlackScholesEngine()
MARKET = MarketData(spot=100.0, rate=0.05, vol=0.20)


def test_straddle_payoff_matches_manual_formula():
    straddle = create_structure("straddle", K=100, T=0.5, market=MARKET, engine=ENGINE)
    call_premium = straddle.option_legs[0].entry_price
    put_premium = straddle.option_legs[1].entry_price

    S_T = np.array([70.0, 100.0, 130.0])
    expected = np.maximum(S_T - 100, 0) - call_premium + np.maximum(100 - S_T, 0) - put_premium
    np.testing.assert_allclose(payoff_at_expiry(straddle, S_T), expected)


def test_short_straddle_is_mirror_image_of_long():
    long_s = create_structure("straddle", K=100, T=0.5, market=MARKET, engine=ENGINE)
    short_s = create_structure("short_straddle", K=100, T=0.5, market=MARKET, engine=ENGINE)
    S_T = np.linspace(60, 140, 50)
    np.testing.assert_allclose(payoff_at_expiry(long_s, S_T), -payoff_at_expiry(short_s, S_T))


@pytest.mark.parametrize(
    "name,kwargs,expect",
    [
        ("straddle", dict(K=100, T=0.5), dict(gamma=">0", theta="<0", vega=">0")),
        ("short_straddle", dict(K=100, T=0.5), dict(gamma="<0", theta=">0", vega="<0")),
        ("covered_call", dict(K=105, T=0.5), dict(gamma="<0", vega="<0")),
        ("protective_put", dict(K=95, T=0.5), dict(gamma=">0", vega=">0")),
    ],
)
def test_portfolio_greeks_have_expected_signs(name, kwargs, expect):
    structure = create_structure(name, market=MARKET, engine=ENGINE, **kwargs)
    g = portfolio_greeks(structure, MARKET, ENGINE)
    values = {"gamma": g.gamma, "theta": g.theta, "vega": g.vega}
    for greek, condition in expect.items():
        value = values[greek]
        if condition == ">0":
            assert value > 0, f"{name}.{greek} = {value}, expected > 0"
        else:
            assert value < 0, f"{name}.{greek} = {value}, expected < 0"


def test_straddle_is_approximately_delta_neutral_atm():
    straddle = create_structure("straddle", K=100, T=0.5, market=MARKET, engine=ENGINE)
    g = portfolio_greeks(straddle, MARKET, ENGINE)
    assert abs(g.delta) < 0.25  # ATM call delta ~0.5-ish offset by put delta ~-0.5-ish, small residual from drift/rates


def test_mark_to_market_is_zero_at_inception():
    for name, kwargs in [("straddle", dict(K=100, T=0.5)), ("bull_call_spread", dict(K_low=100, K_high=110, T=0.5))]:
        structure = create_structure(name, market=MARKET, engine=ENGINE, **kwargs)
        assert mark_to_market(structure, MARKET, 0.0, ENGINE) == pytest.approx(0.0, abs=1e-8)


def test_mark_to_market_at_expiry_matches_payoff():
    straddle = create_structure("straddle", K=100, T=0.5, market=MARKET, engine=ENGINE)
    S_final = 115.0
    mtm = mark_to_market(straddle, MarketData(spot=S_final, rate=0.05, vol=0.2), elapsed=0.5, engine=ENGINE)
    expected = float(payoff_at_expiry(straddle, np.array(S_final)))
    assert mtm == pytest.approx(expected, abs=1e-6)


def test_entry_cost_sign_convention():
    long_call = create_structure("long_call", K=100, T=0.5, market=MARKET, engine=ENGINE)
    assert long_call.entry_cost > 0  # paying a premium is a net debit

    short_straddle = create_structure("short_straddle", K=100, T=0.5, market=MARKET, engine=ENGINE)
    assert short_straddle.entry_cost < 0  # collecting two premiums is a net credit
