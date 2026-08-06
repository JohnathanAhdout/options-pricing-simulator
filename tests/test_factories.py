import pytest

from optionspricer.implied_vol import available_solvers, create_iv_solver
from optionspricer.pricing import BinomialEngine, BlackScholesEngine, MonteCarloEngine, available_engines, create_pricing_engine
from optionspricer.structures import available_structures, create_structure
from optionspricer.market import MarketData
from optionspricer.pricing.black_scholes import BlackScholesEngine as BSEngine


@pytest.mark.parametrize(
    "name,cls",
    [("black_scholes", BlackScholesEngine), ("monte_carlo", MonteCarloEngine), ("binomial", BinomialEngine)],
)
def test_create_pricing_engine_returns_right_type(name, cls):
    assert isinstance(create_pricing_engine(name), cls)


def test_create_pricing_engine_passes_kwargs():
    engine = create_pricing_engine("binomial", n_steps=777)
    assert engine.n_steps == 777


def test_unknown_pricing_engine_raises_with_helpful_message():
    with pytest.raises(ValueError, match="unknown pricing engine"):
        create_pricing_engine("quantum_annealer")


def test_available_engines_lists_all_three():
    assert set(available_engines()) == {"black_scholes", "monte_carlo", "binomial"}


def test_create_iv_solver_unknown_raises():
    with pytest.raises(ValueError, match="unknown IV solver"):
        create_iv_solver("ouija_board")


def test_available_solvers_lists_all_three():
    assert set(available_solvers()) == {"newton", "brent", "jaeckel"}


def test_create_structure_unknown_raises():
    market = MarketData(spot=100, rate=0.05, vol=0.2)
    with pytest.raises(ValueError, match="unknown structure"):
        create_structure("iron_condor", market=market, engine=BSEngine())


def test_available_structures_lists_named_strategies():
    names = available_structures()
    for expected in ["long_call", "long_put", "covered_call", "protective_put", "straddle", "strangle", "bull_call_spread", "short_straddle"]:
        assert expected in names
