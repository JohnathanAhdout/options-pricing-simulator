import pytest

from optionspricer.market import MarketData, OptionSpec, OptionType


def test_option_spec_rejects_non_positive_strike():
    with pytest.raises(ValueError):
        OptionSpec(strike=0, maturity=1.0, option_type=OptionType.CALL)


def test_option_spec_rejects_non_positive_maturity():
    with pytest.raises(ValueError):
        OptionSpec(strike=100, maturity=0.0, option_type=OptionType.CALL)


def test_market_data_rejects_non_positive_spot():
    with pytest.raises(ValueError):
        MarketData(spot=-1, rate=0.05, vol=0.2)


def test_market_data_rejects_negative_vol():
    with pytest.raises(ValueError):
        MarketData(spot=100, rate=0.05, vol=-0.1)


def test_option_spec_is_frozen():
    opt = OptionSpec(strike=100, maturity=1.0, option_type=OptionType.CALL)
    with pytest.raises(AttributeError):
        opt.strike = 200
