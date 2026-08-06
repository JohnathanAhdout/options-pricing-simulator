import numpy as np
import pytest

from optionspricer.simulation import gbm_path, gbm_terminal, regime_switching_gbm_path


def test_gbm_terminal_mean_matches_risk_neutral_drift():
    rng = np.random.default_rng(0)
    S0, T, r, sigma, q = 100.0, 1.0, 0.05, 0.2, 0.0
    ST = gbm_terminal(S0, T, r, sigma, q, 500_000, rng)
    expected_mean = S0 * np.exp((r - q) * T)  # E[S_T] = S0 * e^{(r-q)T} under the risk-neutral measure
    se = ST.std() / np.sqrt(len(ST))
    assert abs(ST.mean() - expected_mean) < 4 * se


def test_gbm_path_starts_at_s0_and_has_right_shape():
    rng = np.random.default_rng(0)
    paths = gbm_path(100.0, 1.0, 0.05, 0.2, 0.0, n_steps=50, n_paths=10, rng=rng)
    assert paths.shape == (10, 51)
    np.testing.assert_allclose(paths[:, 0], 100.0)
    assert np.all(paths > 0)  # GBM can never go negative


def test_gbm_path_reproducible_with_same_rng_state():
    p1 = gbm_path(100.0, 1.0, 0.05, 0.2, 0.0, 20, 5, np.random.default_rng(3))
    p2 = gbm_path(100.0, 1.0, 0.05, 0.2, 0.0, 20, 5, np.random.default_rng(3))
    np.testing.assert_array_equal(p1, p2)


def test_regime_switching_path_visits_both_states():
    rng = np.random.default_rng(1)
    vols = np.array([0.1, 0.4])
    transmat = np.array([[0.9, 0.1], [0.1, 0.9]])
    prices, states = regime_switching_gbm_path(100.0, 2000, 1 / 252, 0.05, vols, transmat, rng)
    assert prices.shape == (2001,)
    assert states.shape == (2001,)
    assert set(np.unique(states)) == {0, 1}
    assert np.all(prices > 0)


def test_regime_switching_high_vol_state_has_larger_realized_moves():
    rng = np.random.default_rng(1)
    vols = np.array([0.05, 0.60])
    transmat = np.array([[0.995, 0.005], [0.005, 0.995]])  # very persistent, so regimes have enough length to measure
    prices, states = regime_switching_gbm_path(100.0, 5000, 1 / 252, 0.0, vols, transmat, rng)
    log_returns = np.diff(np.log(prices))
    ret_low = log_returns[states[1:] == 0]
    ret_high = log_returns[states[1:] == 1]
    assert ret_high.std() > ret_low.std()
