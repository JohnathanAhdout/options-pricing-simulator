"""The flagship experiment, part 2: can a volatility-regime detector turn
the theory from `run_gamma_theta_pnl.py` into an actual trading edge?

Part 1 established the mechanism: a delta-hedged position's P&L is driven
by the gap between realized and hedged volatility, weighted by gamma. That
suggests a strategy -- be short gamma (sell options, collect theta) when
you expect calm markets, be long gamma (buy options, pay theta) when you
expect turbulence -- but it requires an actual forecast of which regime is
coming, not hindsight.

This experiment builds a synthetic world with two hidden volatility
regimes (calm and turbulent, Markov-switching, see
`simulation.regime_switching_gbm_path`), fits the from-scratch HMM in
`regime.py` on a rolling, strictly causal window of realized returns (no
peeking at the future), and uses the filtered regime probability to choose
a stance -- long gamma, short gamma, or flat when unsure -- re-hedged
daily via the same `simulate_delta_hedge` engine part 1 already validated.
It's compared against two baselines: always-short-gamma (pure theta
harvesting, part 1's position) and always-long-gamma (pure long
volatility), across many independent regime-switching seeds.

This is graded honestly: if the regime-aware strategy doesn't beat the
better baseline, that's reported as the finding, not hidden.

Run: uv run python experiments/run_regime_gamma_scalping.py
Output: experiments/results/regime_gamma_scalping.png
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optionspricer import regime
from optionspricer.hedging import simulate_delta_hedge
from optionspricer.market import MarketData
from optionspricer.pricing.black_scholes import BlackScholesEngine
from optionspricer.simulation import regime_switching_gbm_path
from optionspricer.structures import create_structure

ENGINE = BlackScholesEngine()
S0, R = 100.0, 0.05
VOLS_TRUE = np.array([0.12, 0.35])  # calm, turbulent
TRANSMAT_TRUE = np.array([[0.98, 0.02], [0.04, 0.96]])  # regimes persist for ~50 / ~25 trading days on average
HEDGE_VOL = float(VOLS_TRUE.mean())  # the option is priced/sold as if vol were the long-run blend of both regimes
K, T, N_STEPS = 100.0, 0.5, 126  # one trading "episode": 6 months, daily rebalancing
FIT_WINDOW = 120  # trailing days of returns the HMM is refit on before each episode's decision


def stance_from_regime_prob(p_turbulent: float, confidence: float = 0.65) -> str:
    """Long gamma if the detector is confident the turbulent regime is
    active, short gamma if confident it's calm, flat (sit out) if unsure --
    a real strategy has to be able to say "I don't know" instead of always
    taking a position."""
    if p_turbulent > confidence:
        return "long"
    if p_turbulent < (1 - confidence):
        return "short"
    return "flat"


def run_episode(seed: int, strategy: str) -> float:
    """One 6-month episode: simulate a regime-switching path, decide a
    stance (for 'regime', via the HMM; fixed for the baselines), build the
    corresponding straddle, and delta-hedge it to expiry. Returns final P&L."""
    rng = np.random.default_rng(seed)

    burn_in = FIT_WINDOW + 5
    warmup_prices, warmup_states = regime_switching_gbm_path(S0, burn_in, T / N_STEPS, R, VOLS_TRUE, TRANSMAT_TRUE, rng, initial_state=rng.integers(2))
    episode_prices, episode_states = regime_switching_gbm_path(warmup_prices[-1], N_STEPS, T / N_STEPS, R, VOLS_TRUE, TRANSMAT_TRUE, rng, initial_state=warmup_states[-1])

    if strategy == "regime":
        trailing_returns = np.diff(np.log(warmup_prices))[-FIT_WINDOW:]
        params, _ = regime.fit(trailing_returns, n_states=2, n_iter=100, seed=seed)
        filtered = regime.filtered_state_probs(trailing_returns, params)
        turbulent_label = int(np.argmax(params.stdevs))
        p_turbulent_now = filtered[-1, turbulent_label]
        stance = stance_from_regime_prob(p_turbulent_now)
    elif strategy in ("long", "short"):
        stance = strategy
    else:
        raise ValueError(strategy)

    if stance == "flat":
        return 0.0

    entry_market = MarketData(spot=episode_prices[0], rate=R, vol=HEDGE_VOL)
    name = "straddle" if stance == "long" else "short_straddle"
    structure = create_structure(name, K=K, T=T, market=entry_market, engine=ENGINE)

    # `simulate_delta_hedge` draws its own fresh path internally, which would hedge
    # a *different* random path than the one the stance decision above was actually
    # based on -- so the hedge P&L has to be computed along this exact episode_prices
    # path instead, via the small local reimplementation below.
    return _hedge_along_path(structure, episode_prices, R, HEDGE_VOL, T, N_STEPS)


def _hedge_along_path(structure, path: np.ndarray, r: float, hedge_vol: float, maturity: float, n_steps: int) -> float:
    """Same self-financing delta-hedge accounting as `simulate_delta_hedge`,
    but driven by an externally supplied path instead of simulating its own
    -- needed here so the hedge P&L is computed along the *actual* regime-
    switching path the stance decision was based on."""
    from optionspricer.structures.base import mark_to_market, portfolio_greeks
    from optionspricer.hedging import _shift

    dt = maturity / n_steps

    def hedge_market(S):
        return MarketData(spot=S, rate=r, vol=hedge_vol)

    entry_cost = structure.entry_cost
    cash = -entry_cost
    shares = 0.0
    for i in range(n_steps):
        elapsed = i * dt
        S_i = path[i]
        g = portfolio_greeks(_shift(structure, elapsed), hedge_market(S_i), ENGINE)
        target_shares = -g.delta
        cash -= (target_shares - shares) * S_i
        shares = target_shares
        cash *= np.exp(r * dt)

    elapsed_next = n_steps * dt
    S_next = path[n_steps]
    raw_option_value = mark_to_market(structure, hedge_market(S_next), elapsed_next, ENGINE) + entry_cost
    return float(cash + shares * S_next + raw_option_value)


N_EPISODES = 400
results = {name: np.array([run_episode(seed, name) for seed in range(N_EPISODES)]) for name in ["short", "long", "regime"]}

print(f"Hedge vol (priced-in, blend of regimes) = {HEDGE_VOL:.3f}; true regime vols = {VOLS_TRUE}\n")
print(f"{'strategy':<10} {'mean P&L':>10} {'std':>10} {'SE':>8} {'win rate':>9} {'sharpe-like':>12} {'p5':>8} {'p95':>8}")
for name, pnl in results.items():
    mean, std = pnl.mean(), pnl.std(ddof=1)
    se = std / np.sqrt(len(pnl))
    win_rate = (pnl > 0).mean()
    sharpe = mean / std if std > 0 else float("nan")
    p5, p95 = np.percentile(pnl, [5, 95])
    print(f"{name:<10} {mean:>10.3f} {std:>10.3f} {se:>8.3f} {win_rate:>9.1%} {sharpe:>12.3f} {p5:>8.3f} {p95:>8.3f}")

improvement_vs_best_baseline = results["regime"].mean() - max(results["short"].mean(), results["long"].mean())
print(f"\nRegime strategy mean P&L minus the better fixed baseline: {improvement_vs_best_baseline:+.3f}")

# Does the regime edge actually come from detection accuracy? Refit at several
# window lengths and check classification accuracy against the (simulation-only,
# never used by the strategy itself) ground-truth regime label at the decision
# point -- a short window means too few observations for 2-state EM to separate
# "calm" from "turbulent" reliably, so accuracy should rise with window length.
print(f"\n{'fit window':>10} {'accuracy vs. true regime':>26} {'n decisive calls':>18}")
for window in [40, 80, 120, 200]:
    correct = total = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        burn_in = window + 5
        warmup_prices, warmup_states = regime_switching_gbm_path(S0, burn_in, T / N_STEPS, R, VOLS_TRUE, TRANSMAT_TRUE, rng, initial_state=rng.integers(2))
        trailing_returns = np.diff(np.log(warmup_prices))[-window:]
        params, _ = regime.fit(trailing_returns, n_states=2, n_iter=100, seed=seed)
        filtered = regime.filtered_state_probs(trailing_returns, params)
        turbulent_label = int(np.argmax(params.stdevs))
        stance = stance_from_regime_prob(filtered[-1, turbulent_label])
        if stance == "flat":
            continue
        true_stance = "long" if warmup_states[-1] == 1 else "short"
        total += 1
        correct += stance == true_stance
    print(f"{window:>10} {correct / total:>26.1%} {total:>18}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Regime-aware gamma scalping vs. fixed long/short-gamma baselines", fontweight="bold")

ax = axes[0]
bins = np.linspace(min(p.min() for p in results.values()), max(p.max() for p in results.values()), 40)
for name, color in zip(results, ["firebrick", "seagreen", "steelblue"]):
    ax.hist(results[name], bins=bins, alpha=0.5, label=name, color=color)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("episode P&L ($)")
ax.set_ylabel("count")
ax.set_title(f"P&L distribution across {N_EPISODES} episodes")
ax.legend()

ax = axes[1]
means = [results[n].mean() for n in results]
ses = [results[n].std(ddof=1) / np.sqrt(N_EPISODES) for n in results]
ax.bar(list(results.keys()), means, yerr=[1.96 * s for s in ses], capsize=5, color=["firebrick", "seagreen", "steelblue"])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("mean P&L ($, +/- 95% CI)")
ax.set_title("Mean P&L per strategy")

ax = axes[2]
rng = np.random.default_rng(11)
warmup_prices, warmup_states = regime_switching_gbm_path(S0, FIT_WINDOW + 5, T / N_STEPS, R, VOLS_TRUE, TRANSMAT_TRUE, rng, initial_state=0)
episode_prices, episode_states = regime_switching_gbm_path(warmup_prices[-1], N_STEPS, T / N_STEPS, R, VOLS_TRUE, TRANSMAT_TRUE, rng, initial_state=warmup_states[-1])
full_prices = np.concatenate([warmup_prices, episode_prices[1:]])
full_states = np.concatenate([warmup_states, episode_states[1:]])
returns = np.diff(np.log(full_prices))
params, _ = regime.fit(returns, n_states=2, n_iter=100, seed=0)
turbulent_label = int(np.argmax(params.stdevs))
filtered = regime.filtered_state_probs(returns, params)[:, turbulent_label]
ax2 = ax.twinx()
ax.plot(full_prices, color="steelblue", lw=1)
ax.set_ylabel("price", color="steelblue")
ax2.plot(np.arange(1, len(full_prices)), filtered, color="firebrick", lw=1.2, alpha=0.8)
ax2.fill_between(np.arange(1, len(full_prices)), 0, full_states[1:], step="pre", alpha=0.15, color="grey")
ax2.set_ylabel("P(turbulent | data so far)", color="firebrick")
ax.set_xlabel("trading day")
ax.set_title("One example path: filtered regime probability vs. price")

plt.tight_layout()
plt.savefig("experiments/results/regime_gamma_scalping.png", dpi=150)
print("\nSaved experiments/results/regime_gamma_scalping.png")
