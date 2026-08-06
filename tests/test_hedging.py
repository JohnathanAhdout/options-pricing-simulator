import numpy as np
import pytest

from optionspricer.hedging import simulate_delta_hedge
from optionspricer.market import MarketData
from optionspricer.pricing.black_scholes import BlackScholesEngine
from optionspricer.structures import create_structure

ENGINE = BlackScholesEngine()
S0, R, K, T, HEDGE_VOL = 100.0, 0.05, 100.0, 0.5, 0.20


def _run_many(realized_vol: float, n_seeds: int = 300, n_steps: int = 126):
    finals, theos = [], []
    entry_market = MarketData(spot=S0, rate=R, vol=HEDGE_VOL)
    for seed in range(n_seeds):
        structure = create_structure("short_straddle", K=K, T=T, market=entry_market, engine=ENGINE)
        rng = np.random.default_rng(seed)
        result = simulate_delta_hedge(structure, S0, R, HEDGE_VOL, realized_vol, T, n_steps, ENGINE, rng)
        finals.append(result.final_pnl)
        theos.append(result.theoretical_pnl)
    return np.array(finals), np.array(theos)


def test_short_gamma_loses_when_realized_exceeds_hedge_vol():
    finals, theos = _run_many(realized_vol=0.35)
    assert finals.mean() < 0
    assert finals.mean() == pytest.approx(theos.mean(), rel=0.1)


def test_short_gamma_wins_when_realized_below_hedge_vol():
    finals, theos = _run_many(realized_vol=0.10)
    assert finals.mean() > 0
    assert finals.mean() == pytest.approx(theos.mean(), rel=0.1)


def test_unbiased_when_realized_equals_hedge_vol():
    finals, _ = _run_many(realized_vol=HEDGE_VOL, n_seeds=500)
    se = finals.std(ddof=1) / np.sqrt(len(finals))
    assert abs(finals.mean()) < 4 * se  # no systematic bias beyond sampling noise


def test_more_frequent_hedging_reduces_pnl_dispersion():
    # discrete-hedging error variance should shrink as rebalancing gets more
    # frequent (theory converges in expectation regardless of frequency; only
    # the variance around it should shrink)
    finals_coarse, _ = _run_many(realized_vol=0.20, n_seeds=200, n_steps=12)
    finals_fine, _ = _run_many(realized_vol=0.20, n_seeds=200, n_steps=252)
    assert finals_fine.std() < finals_coarse.std()


def test_rejects_structure_with_stock_legs():
    entry_market = MarketData(spot=S0, rate=R, vol=HEDGE_VOL)
    covered_call = create_structure("covered_call", K=K, T=T, market=entry_market, engine=ENGINE)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="options-only"):
        simulate_delta_hedge(covered_call, S0, R, HEDGE_VOL, 0.2, T, 50, ENGINE, rng)
