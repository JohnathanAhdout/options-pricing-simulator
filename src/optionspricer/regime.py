"""A Gaussian Hidden Markov Model for volatility-regime detection, fit from
scratch with the Baum-Welch (EM) algorithm.

The motivating problem: Black-Scholes assumes one constant sigma, but
`surface.py` (and the live SPY smile in `experiments/run_vol_surface.py`)
shows real markets don't behave that way. Volatility clusters into calm
and turbulent stretches. A Hidden Markov Model is the standard tool for
that shape of problem: returns are assumed to be drawn from one of a small
number of Gaussian "regimes," and which regime is active follows its own
Markov chain that we don't observe directly, only through its fingerprint
on the returns. Fitting one here does two things: it gives
`experiments/run_regime_gamma_scalping.py` a real (if simple) volatility
forecast to trade on, and it's a second, independent demonstration,
alongside the vol smile, that "volatility" is not the single fixed
number Black-Scholes wants it to be.

**The moving parts, briefly:**

- `startprob[i]` = P(state_0 = i), `transmat[i, j]` = P(state_{t+1} = j |
  state_t = i), `means[i]` / `stdevs[i]` = the Gaussian emission
  distribution of a return while in state i.
- **Forward algorithm**: computes `alpha[t, i]` = P(state_t = i | returns up
  to and including t), filtering the data causally. `alpha[t]` never
  looks at returns after t, which is exactly what a live trading signal is
  allowed to use.
- **Backward algorithm**: computes `beta[t, i]`, the (scaled) probability of
  every future return given state_t = i. Forward and backward together give
  `gamma[t, i]` = P(state_t = i | *all* the data, past and future): the
  best possible hindsight estimate, only valid for offline analysis.
- **Baum-Welch**: alternates computing gamma (E-step) with re-estimating
  startprob/transmat/means/stdevs from it (M-step), a special case of EM
  where the "soft assignment" is a full forward-backward pass instead of a
  simple posterior.
- **Viterbi**: the single most likely *whole path* of states given all the
  data (as opposed to gamma's most likely state *at each t* independently),
  found by dynamic programming in log-space.

All of it is scaled/log-space where numerically necessary (see `_forward`),
because raw products of a few hundred probabilities underflow to zero in
double precision almost immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HMMParams:
    startprob: np.ndarray  # shape (n_states,)
    transmat: np.ndarray  # shape (n_states, n_states), transmat[i, j] = P(j | i)
    means: np.ndarray  # shape (n_states,)
    stdevs: np.ndarray  # shape (n_states,)

    @property
    def n_states(self) -> int:
        return len(self.means)


def _log_emission(returns: np.ndarray, params: HMMParams) -> np.ndarray:
    """log N(returns[t]; means[i], stdevs[i]^2) for every (t, i); shape
    (n_obs, n_states)."""
    r = returns[:, None]
    mu, sd = params.means[None, :], params.stdevs[None, :]
    return -0.5 * np.log(2 * np.pi * sd**2) - 0.5 * ((r - mu) / sd) ** 2


def _emission_probs(returns: np.ndarray, params: HMMParams) -> np.ndarray:
    """exp(log-emission), clipped before exponentiating so a wildly
    off-distribution observation underflows to a very small (not exactly
    zero, not inf/nan) probability rather than crashing the recursion.
    Deliberately *not* row-max-stabilized: with normalization happening at
    every step of `_forward` anyway, that stabilization is unnecessary here
    given realistic z-score ranges, and it would corrupt the log-likelihood
    reported by `fit()`; see the note there."""
    logB = _log_emission(returns, params)
    return np.exp(np.clip(logB, -700.0, None))


def _forward(returns: np.ndarray, params: HMMParams) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward pass (Rabiner 1989): alpha_hat[t, i] is normalized to
    sum to 1 at every t to prevent underflow, with the per-step normalizer
    c[t] recorded so the backward pass can reuse it and so the log-likelihood
    can be recovered exactly as sum(log(c))."""
    n_obs = len(returns)
    n_states = params.n_states
    B = _emission_probs(returns, params)

    alpha_hat = np.zeros((n_obs, n_states))
    c = np.zeros(n_obs)

    alpha_hat[0] = params.startprob * B[0]  # unnormalized: P(state_0=i) * P(return_0 | state_0=i), for every i
    c[0] = alpha_hat[0].sum()  # normalizer = P(return_0), i.e. the total probability mass across all states
    alpha_hat[0] /= c[0]  # alpha_hat[0, i] is now P(state_0=i | return_0), a genuine probability distribution over i

    for t in range(1, n_obs):
        # (alpha_hat[t-1] @ transmat)[j] = sum_i P(state_{t-1}=i | past) * P(state_t=j | state_{t-1}=i)
        #   = P(state_t=j | returns up to t-1): the one-step-ahead prediction, before seeing return_t
        # multiplying by B[t] folds in the new observation: P(state_t=j | returns up to t) (unnormalized)
        alpha_hat[t] = B[t] * (alpha_hat[t - 1] @ params.transmat)
        c[t] = alpha_hat[t].sum()  # normalizer = P(return_t | returns up to t-1); the product of all c's is P(all returns)
        alpha_hat[t] /= c[t]  # renormalize back to a probability distribution before moving to the next step

    return alpha_hat, c


def _backward(returns: np.ndarray, params: HMMParams, c: np.ndarray) -> np.ndarray:
    """Scaled backward pass, reusing the forward pass's normalizers `c` so
    beta_hat stays on the same scale as alpha_hat (required for
    `gamma = alpha_hat * beta_hat` to be a valid probability without a
    further renormalization)."""
    n_obs = len(returns)
    n_states = params.n_states
    B = _emission_probs(returns, params)

    beta_hat = np.zeros((n_obs, n_states))
    beta_hat[-1] = 1.0  # by convention: "probability of the future given state_{T-1}" is trivially 1 when there's no future left
    for t in range(n_obs - 2, -1, -1):
        # (B[t+1] * beta_hat[t+1])[j] = P(return_{t+1} | state_{t+1}=j) * (scaled) P(returns after t+1 | state_{t+1}=j)
        # transmat @ (...) sums that over all j, weighted by P(state_{t+1}=j | state_t=i), marginalizing out state_{t+1}
        # dividing by c[t+1] keeps beta_hat on the same normalized scale that alpha_hat is on at every t
        beta_hat[t] = (params.transmat @ (B[t + 1] * beta_hat[t + 1])) / c[t + 1]
    return beta_hat


def filtered_state_probs(returns: np.ndarray, params: HMMParams) -> np.ndarray:
    """P(state_t = i | returns[:t+1]) for every t: the *causal* regime
    probability, safe for a live strategy to condition on since it never
    looks ahead. This is exactly the forward pass's alpha_hat."""
    alpha_hat, _ = _forward(returns, params)
    return alpha_hat


def smoothed_state_probs(returns: np.ndarray, params: HMMParams) -> np.ndarray:
    """P(state_t = i | *all* returns): the full-hindsight regime
    probability. Only valid for post-hoc analysis of a completed series,
    never for a trading decision made at time t."""
    alpha_hat, c = _forward(returns, params)
    beta_hat = _backward(returns, params, c)
    gamma = alpha_hat * beta_hat
    return gamma / gamma.sum(axis=1, keepdims=True)


def viterbi(returns: np.ndarray, params: HMMParams) -> np.ndarray:
    """Most likely single state path (as opposed to `smoothed_state_probs`'s
    most likely state *at each t independently*), via log-space dynamic
    programming."""
    n_obs = len(returns)
    n_states = params.n_states
    logB = _log_emission(returns, params)
    log_transmat = np.log(params.transmat + 1e-300)
    log_startprob = np.log(params.startprob + 1e-300)

    delta = np.zeros((n_obs, n_states))  # delta[t, j] = log-probability of the BEST path ending in state j at time t
    psi = np.zeros((n_obs, n_states), dtype=int)  # psi[t, j] = which state at t-1 that best path came from: the backpointer
    delta[0] = log_startprob + logB[0]

    for t in range(1, n_obs):
        scores = delta[t - 1][:, None] + log_transmat  # scores[i, j] = best path score ending in i, then transitioning to j
        psi[t] = np.argmax(scores, axis=0)  # for each destination j, which source i gave the best score: the backpointer to store
        delta[t] = scores[psi[t], np.arange(n_states)] + logB[t]  # take that best incoming score, then add this step's emission

    path = np.zeros(n_obs, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))  # the final state of the single best path is whichever has the highest overall score
    for t in range(n_obs - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]  # walk the backpointers in reverse to recover the whole path, not just its endpoint
    return path


def fit(
    returns: np.ndarray,
    n_states: int = 2,
    n_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
) -> tuple[HMMParams, list[float]]:
    """Baum-Welch: alternate an E-step (forward-backward, producing gamma
    and the pairwise xi) with an M-step (closed-form re-estimation of every
    parameter from those soft assignments) until the log-likelihood stops
    improving by more than `tol`.

    Initialization matters for a mixture-of-Gaussians-style model: starting
    every state with identical parameters is a fixed point of EM (perfect
    symmetry never breaks), so states are seeded with *different* standard
    deviations spread around the data's overall std, which is enough to let
    EM sort out which regime is calm and which is turbulent. The
    transition matrix starts strongly persistent (0.95 on the diagonal),
    matching the prior that vol regimes last for a while, not one tick.
    """
    rng = np.random.default_rng(seed)
    overall_std = returns.std()
    overall_mean = returns.mean()

    spread = np.linspace(0.5, 1.5, n_states)
    means = overall_mean + rng.normal(scale=overall_std * 0.05, size=n_states)
    stdevs = overall_std * spread
    transmat = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
    np.fill_diagonal(transmat, 0.95)
    startprob = np.full(n_states, 1.0 / n_states)

    params = HMMParams(startprob=startprob, transmat=transmat, means=means, stdevs=stdevs)
    log_likelihoods: list[float] = []

    for _ in range(n_iter):
        # --- E-step: run forward-backward under the CURRENT params ---
        alpha_hat, c = _forward(returns, params)
        beta_hat = _backward(returns, params, c)
        ll = float(np.log(c).sum())  # log-likelihood of the data under the params BEFORE this iteration's update
        log_likelihoods.append(ll)

        gamma = alpha_hat * beta_hat  # P(state_t=i | ALL the data) for every t. See module docstring for why this product is exact
        gamma /= gamma.sum(axis=1, keepdims=True)  # defensive renormalization; should already sum to ~1 given the scaling invariant above

        B = _emission_probs(returns, params)
        xi_sum = np.zeros((n_states, n_states))
        for t in range(len(returns) - 1):
            # xi_t(i, j) = P(state_t=i, state_{t+1}=j | all the data): the joint (not just marginal)
            # posterior over consecutive states, which is exactly what re-estimating a TRANSITION
            # matrix needs (as opposed to gamma, which only has the marginal at a single t)
            num = np.outer(alpha_hat[t], B[t + 1] * beta_hat[t + 1]) * params.transmat / c[t + 1]
            xi_sum += num

        # --- M-step: closed-form MLE re-estimates from the soft (gamma/xi) assignments ---
        new_startprob = gamma[0].copy()  # best guess at the initial-state distribution is just the posterior at t=0
        # transmat[i,j] = (expected # of i->j transitions) / (expected # of times in state i, excluding the last step)
        new_transmat = xi_sum / gamma[:-1].sum(axis=0, keepdims=True).T
        new_transmat /= new_transmat.sum(axis=1, keepdims=True)  # defensive renormalization, same rationale as gamma above

        weights = gamma.sum(axis=0)  # "effective sample size" in each state: how much of the data (softly) belongs to state i
        new_means = (gamma * returns[:, None]).sum(axis=0) / weights  # weighted average return, weighted by P(state=i) at each t
        new_var = (gamma * (returns[:, None] - new_means[None, :]) ** 2).sum(axis=0) / weights  # weighted variance around the NEW mean (the M-step maximizer)
        new_stdevs = np.sqrt(np.maximum(new_var, 1e-10))  # floor at a tiny variance so a state can't collapse to a literal Dirac delta

        params = HMMParams(startprob=new_startprob, transmat=new_transmat, means=new_means, stdevs=new_stdevs)

        if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return params, log_likelihoods
