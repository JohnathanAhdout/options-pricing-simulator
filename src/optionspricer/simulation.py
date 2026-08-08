"""Pure functions for simulating the underlying. Three variants, each used
by a different part of the package:

- `gbm_terminal`: only the distribution at expiry. All Monte Carlo pricing
  needs, since a European payoff only looks at S_T.
- `gbm_path`: the whole trajectory at a fixed volatility, what the
  delta-hedging simulator walks tick by tick.
- `regime_switching_gbm_path`: a trajectory whose volatility itself jumps
  between a small number of states according to a Markov chain: the
  synthetic world the regime-detection experiment tries to detect and react
  to. It's also just "GBM," except sigma(t) is itself the output of a
  discrete-time Markov chain instead of a constant.

None of these take an implicit global random state: every function takes an
explicit `numpy.random.Generator`, so a caller who wants byte-for-byte
reproducible output controls it by seeding once, upstream, and every draw in
this module traces back to that one seed.
"""

from __future__ import annotations

import numpy as np


def gbm_terminal(S0: float, T: float, r: float, sigma: float, q: float, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """S_T for n_paths independent draws of geometric Brownian motion:
    S_T = S0 * exp((r - q - sigma^2/2) T + sigma sqrt(T) Z), Z ~ N(0, 1).
    The (r - q - sigma^2/2) drift is the Ito-corrected risk-neutral drift;
    see BACKGROUND.md for why the naive (r - q) is wrong."""
    Z = rng.standard_normal(n_paths)  # n_paths independent draws of a standard normal, one terminal shock per path
    return S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)  # vectorized: every path's S_T computed in one shot


def gbm_path(S0: float, T: float, r: float, sigma: float, q: float, n_steps: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """n_paths independent GBM trajectories of shape (n_paths, n_steps + 1),
    column 0 is S0. Built by cumulatively summing per-step log-returns, which
    is both faster and numerically better-conditioned than compounding
    prices step by step."""
    dt = T / n_steps  # length of one step, in years
    Z = rng.standard_normal((n_paths, n_steps))  # one independent N(0,1) shock per (path, step), draws the whole batch at once
    log_returns = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z  # each step's log-return: Ito-corrected drift*dt + vol*sqrt(dt)*Z
    log_paths = np.cumsum(log_returns, axis=1)  # running sum along the time axis = log(S_t / S0) at every step, for every path
    S = S0 * np.exp(log_paths)  # convert cumulative log-returns back into absolute prices
    return np.hstack([np.full((n_paths, 1), S0), S])  # prepend the known starting price so column 0 is always exactly S0


def regime_switching_gbm_path(
    S0: float,
    n_steps: int,
    dt: float,
    r: float,
    vols: np.ndarray,
    transition_matrix: np.ndarray,
    rng: np.random.Generator,
    initial_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A single GBM trajectory whose volatility is a hidden 2-state (or
    more) Markov chain: at each step the regime either stays put or jumps
    according to `transition_matrix`, and that step's return is drawn from
    N((r - vols[state]^2/2) dt, vols[state]^2 dt).

    Returns (prices, states) both of length n_steps + 1, where states[t] is
    the regime that generated the return *into* prices[t] (states[0] is the
    initial regime, arbitrary since no return has happened yet).

    This is the synthetic world used to test regime detection: `states` is
    the ground truth an HMM fit only on `prices` never gets to see, exactly
    like a real market's volatility regime is never directly observable.
    """
    n_states = len(vols)
    states = np.zeros(n_steps + 1, dtype=int)
    states[0] = initial_state
    log_returns = np.zeros(n_steps)

    # inherently sequential (each state depends on the previous one), so this one can't be
    # vectorized across t the way gbm_path is: the Markov-chain draw has to happen one step at a time
    for t in range(1, n_steps + 1):
        prev_state = states[t - 1]
        states[t] = rng.choice(n_states, p=transition_matrix[prev_state])  # roll the Markov chain forward one tick
        sigma_t = vols[states[t]]  # today's return uses WHICHEVER regime we just landed in, not the previous one
        Z = rng.standard_normal()
        log_returns[t - 1] = (r - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * Z  # same GBM step formula as gbm_path, just with a time-varying sigma

    prices = S0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))  # prepend log-return 0 so prices[0] = S0 exactly
    return prices, states
