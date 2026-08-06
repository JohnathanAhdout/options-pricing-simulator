import pytest

from optionspricer.market import MarketData, OptionSpec, OptionType


@pytest.fixture
def atm_option() -> OptionSpec:
    return OptionSpec(strike=100.0, maturity=0.5, option_type=OptionType.CALL)


@pytest.fixture
def market() -> MarketData:
    return MarketData(spot=100.0, rate=0.05, vol=0.20)
