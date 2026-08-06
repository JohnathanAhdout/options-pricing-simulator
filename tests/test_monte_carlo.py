import numpy as np
import pytest

from optionspricer.market import MarketData, OptionSpec, OptionType
from optionspricer.pricing.black_scholes import price as bs_price
from optionspricer.pricing.monte_carlo import MonteCarloEngine


def test_converges_to_black_scholes_within_confidence_interval():
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    engine = MonteCarloEngine(n_paths=200_000, seed=0)
    opt = OptionSpec(strike=K, maturity=T, option_type=OptionType.CALL)
    result = engine.price(opt, MarketData(spot=S, rate=r, vol=sigma))
    true_price = bs_price(S, K, T, r, sigma, OptionType.CALL)
    # 6-sigma band: astronomically unlikely to fail from sampling noise alone
    assert abs(result.price - true_price) < 6 * result.stderr


def test_reproducible_given_same_seed():
    engine = MonteCarloEngine(n_paths=10_000, seed=42)
    opt = OptionSpec(strike=100, maturity=1.0, option_type=OptionType.CALL)
    market = MarketData(spot=100, rate=0.05, vol=0.2)
    p1 = engine.price(opt, market)
    p2 = engine.price(opt, market)
    assert p1.price == p2.price  # bit-for-bit, not just "close"


def test_different_seeds_give_different_prices():
    opt = OptionSpec(strike=100, maturity=1.0, option_type=OptionType.CALL)
    market = MarketData(spot=100, rate=0.05, vol=0.2)
    p1 = MonteCarloEngine(n_paths=1000, seed=1).price(opt, market)
    p2 = MonteCarloEngine(n_paths=1000, seed=2).price(opt, market)
    assert p1.price != p2.price


def test_stderr_shrinks_like_inverse_sqrt_n():
    opt = OptionSpec(strike=105, maturity=1.0, option_type=OptionType.CALL)
    market = MarketData(spot=100, rate=0.05, vol=0.2)
    se_small = MonteCarloEngine(n_paths=1_000, seed=0).price(opt, market).stderr
    se_large = MonteCarloEngine(n_paths=100_000, seed=0).price(opt, market).stderr
    # 100x more paths -> ~10x smaller stderr (1/sqrt(100) = 0.1); allow generous slack
    ratio = se_small / se_large
    assert 7.0 < ratio < 14.0


def test_common_random_numbers_give_smooth_greeks():
    # Finite-diff greeks on a Monte Carlo engine are only usable if the bumped
    # price() calls share the same underlying draws; this checks delta computed
    # from a reseeding engine is close to the Black-Scholes analytic value,
    # which would NOT reliably hold if each call used independent randomness
    # at only 20,000 paths.
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.2
    engine = MonteCarloEngine(n_paths=20_000, seed=7)
    opt = OptionSpec(strike=K, maturity=T, option_type=OptionType.CALL)
    mc_delta = engine.greeks(opt, MarketData(spot=S, rate=r, vol=sigma)).delta
    from optionspricer.pricing.black_scholes import analytic_greeks
    bs_delta = analytic_greeks(S, K, T, r, sigma, OptionType.CALL).delta
    assert mc_delta == pytest.approx(bs_delta, abs=0.02)


def test_rejects_american_exercise():
    engine = MonteCarloEngine(n_paths=100, seed=0)
    opt = OptionSpec(strike=100, maturity=0.5, option_type=OptionType.CALL, exercise="american")
    with pytest.raises(ValueError, match="European"):
        engine.price(opt, MarketData(spot=100, rate=0.05, vol=0.2))
