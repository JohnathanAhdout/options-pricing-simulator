import pytest

from optionspricer.market import ExerciseStyle, MarketData, OptionSpec, OptionType
from optionspricer.pricing.binomial import BinomialEngine, price, price_and_greeks
from optionspricer.pricing.black_scholes import analytic_greeks, price as bs_price


def test_converges_to_black_scholes_as_steps_increase():
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    true_price = bs_price(S, K, T, r, sigma, OptionType.CALL)
    err_coarse = abs(price(S, K, T, r, sigma, OptionType.CALL, ExerciseStyle.EUROPEAN, n_steps=25) - true_price)
    err_fine = abs(price(S, K, T, r, sigma, OptionType.CALL, ExerciseStyle.EUROPEAN, n_steps=2000) - true_price)
    assert err_fine < err_coarse
    assert err_fine < 1e-2


def test_american_put_worth_at_least_european_put():
    # early exercise is a free option to the holder, so American >= European always
    S, K, T, r, sigma = 90.0, 100.0, 1.0, 0.05, 0.3
    eu = price(S, K, T, r, sigma, OptionType.PUT, ExerciseStyle.EUROPEAN, n_steps=500)
    am = price(S, K, T, r, sigma, OptionType.PUT, ExerciseStyle.AMERICAN, n_steps=500)
    assert am >= eu - 1e-9
    assert am > eu  # for a deep ITM put with no dividends, the premium should be strictly positive


def test_american_call_no_dividends_equals_european():
    # with zero dividend yield, early exercise of a call is never optimal
    # (you'd throw away remaining time value for nothing), so American == European
    S, K, T, r, sigma = 100.0, 90.0, 1.0, 0.05, 0.25
    eu = price(S, K, T, r, sigma, OptionType.CALL, ExerciseStyle.EUROPEAN, n_steps=500)
    am = price(S, K, T, r, sigma, OptionType.CALL, ExerciseStyle.AMERICAN, n_steps=500)
    assert am == pytest.approx(eu, abs=1e-6)


def test_tree_native_gamma_matches_black_scholes():
    # regression test for the piecewise-linearity pathology: a naive bump-and-reprice
    # gamma on a CRR tree comes back as floating point noise near zero; the tree-native
    # method should instead land close to the true (Black-Scholes) gamma
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    _, _, gamma, _ = price_and_greeks(S, K, T, r, sigma, OptionType.CALL, ExerciseStyle.EUROPEAN, n_steps=500)
    bs_gamma = analytic_greeks(S, K, T, r, sigma, OptionType.CALL).gamma
    assert gamma == pytest.approx(bs_gamma, rel=0.02)
    assert gamma > 1e-4  # sanity: not the near-zero floating point noise a naive bump would give


def test_raises_on_too_coarse_a_tree():
    with pytest.raises(ValueError, match="n_steps must be >= 2"):
        price_and_greeks(100, 100, 0.5, 0.05, 0.2, OptionType.CALL, n_steps=1)


def test_engine_prices_and_greeks_are_finite():
    engine = BinomialEngine(n_steps=200)
    opt = OptionSpec(strike=100, maturity=0.5, option_type=OptionType.PUT, exercise=ExerciseStyle.AMERICAN)
    market = MarketData(spot=95, rate=0.05, vol=0.25)
    result = engine.price(opt, market)
    greeks = engine.greeks(opt, market)
    assert result.price > 0
    assert all(map(lambda x: x == x, [greeks.delta, greeks.gamma, greeks.theta, greeks.vega, greeks.rho]))  # no NaNs
