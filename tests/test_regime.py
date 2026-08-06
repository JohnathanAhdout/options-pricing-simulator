import numpy as np
import pytest

from optionspricer import regime


def _simulate_two_regime_returns(n=2000, seed=7, sigma_low=0.005, sigma_high=0.025):
    rng = np.random.default_rng(seed)
    A_true = np.array([[0.97, 0.03], [0.05, 0.95]])
    sigmas_true = np.array([sigma_low, sigma_high])
    states = np.zeros(n, dtype=int)
    returns = np.zeros(n)
    state = 0
    for t in range(n):
        if t > 0:
            state = rng.choice(2, p=A_true[state])
        states[t] = state
        returns[t] = rng.normal(0, sigmas_true[state])
    return returns, states, sigmas_true


def test_fit_recovers_true_volatilities():
    returns, _, sigmas_true = _simulate_two_regime_returns()
    params, _ = regime.fit(returns, n_states=2, n_iter=200, seed=1)
    np.testing.assert_allclose(sorted(params.stdevs), sorted(sigmas_true), rtol=0.1)


def test_log_likelihood_is_monotonically_non_decreasing():
    # the core correctness property of EM: each iteration cannot make the
    # observed-data log-likelihood worse
    returns, _, _ = _simulate_two_regime_returns()
    _, log_likelihoods = regime.fit(returns, n_states=2, n_iter=200, seed=1)
    diffs = np.diff(log_likelihoods)
    assert np.all(diffs >= -1e-6)


def test_forward_pass_log_likelihood_matches_brute_force():
    from scipy.special import logsumexp

    returns, _, _ = _simulate_two_regime_returns(n=200)
    params, _ = regime.fit(returns, n_states=2, n_iter=50, seed=1)

    alpha_hat, c = regime._forward(returns, params)
    scaled_ll = np.log(c).sum()

    log_alpha = np.zeros((len(returns), params.n_states))
    logB = regime._log_emission(returns, params)
    log_transmat = np.log(params.transmat)
    log_alpha[0] = np.log(params.startprob) + logB[0]
    for t in range(1, len(returns)):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0) + logB[t]
    brute_ll = logsumexp(log_alpha[-1])

    assert scaled_ll == pytest.approx(brute_ll, abs=1e-6)


def test_viterbi_recovers_regimes_when_well_separated():
    returns, true_states, _ = _simulate_two_regime_returns()
    params, _ = regime.fit(returns, n_states=2, n_iter=200, seed=1)
    path = regime.viterbi(returns, params)

    low_label = int(np.argmin(params.stdevs))
    aligned = (path != low_label).astype(int)
    accuracy = (aligned == true_states).mean()
    assert accuracy > 0.9


def test_filtered_probs_are_causal_and_sum_to_one():
    returns, _, _ = _simulate_two_regime_returns(n=500)
    params, _ = regime.fit(returns, n_states=2, n_iter=100, seed=1)
    filtered = regime.filtered_state_probs(returns, params)
    assert filtered.shape == (500, 2)
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-8)

    # causality: truncating the series shouldn't change filtered probs for
    # times before the truncation point (nothing after t may leak backward)
    filtered_truncated = regime.filtered_state_probs(returns[:100], params)
    np.testing.assert_allclose(filtered[:100], filtered_truncated, atol=1e-10)


def test_smoothed_probs_sum_to_one():
    returns, _, _ = _simulate_two_regime_returns(n=500)
    params, _ = regime.fit(returns, n_states=2, n_iter=100, seed=1)
    smoothed = regime.smoothed_state_probs(returns, params)
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-8)
