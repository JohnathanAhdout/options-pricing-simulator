"""The flagship experiment, part 2: can a volatility-regime detector turn
the theory from `run_gamma_theta_pnl.py` into an actual trading edge?

Part 1 established the mechanism: a delta-hedged position's P&L is driven
by the gap between realized and hedged volatility, weighted by gamma. That
suggests a strategy: be short gamma (sell options, collect theta) when
you expect calm markets, be long gamma (buy options, pay theta) when you
expect turbulence, but it requires an actual forecast of which regime is
coming, not hindsight.

This experiment builds a synthetic world with two hidden volatility
regimes (calm and turbulent, Markov-switching, see
`simulation.regime_switching_gbm_path`), fits the from-scratch HMM in
`regime.py` on a rolling, strictly causal window of realized returns (no
peeking at the future), and uses the filtered regime probability to choose
a stance (long gamma, short gamma, or flat when unsure), re-hedged
daily via the same `simulate_delta_hedge` engine part 1 already validated.
It's compared against two baselines: always-short-gamma (pure theta
harvesting, part 1's position) and always-long-gamma (pure long
volatility), across many independent regime-switching seeds.

This is graded honestly: if the regime-aware strategy doesn't beat the
better baseline, that's reported as the finding, not hidden.

Run: uv run python experiments/run_regime_gamma_scalping.py
Output: experiments/results/regime_gamma_scalping.png
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

from dataclasses import replace  # imported but unused directly in this file (structure aging happens inside hedging._shift instead)

import matplotlib  # imported separately from pyplot so the backend can be set BEFORE pyplot is imported, below

matplotlib.use("Agg")  # non-interactive, file-only backend: this script never opens a window, just saves a PNG
import matplotlib.pyplot as plt
import numpy as np

from optionspricer import regime  # imported as a module (not `from ... import fit`), since both regime.fit and regime.filtered_state_probs are used below
from optionspricer.hedging import simulate_delta_hedge  # used for the fixed-strategy baselines, which CAN use their own freshly simulated path
from optionspricer.market import MarketData
from optionspricer.pricing.black_scholes import BlackScholesEngine
from optionspricer.simulation import regime_switching_gbm_path  # builds the synthetic 2-regime world every episode is drawn from
from optionspricer.structures import create_structure

ENGINE = BlackScholesEngine()  # one shared engine instance, reused everywhere below
S0, R = 100.0, 0.05  # fixed starting spot and risk-free rate every episode reuses
VOLS_TRUE = np.array([0.12, 0.35])  # calm, turbulent
TRANSMAT_TRUE = np.array([[0.98, 0.02], [0.04, 0.96]])  # regimes persist for ~50 / ~25 trading days on average
HEDGE_VOL = float(VOLS_TRUE.mean())  # the option is priced/sold as if vol were the long-run blend of both regimes
K, T, N_STEPS = 100.0, 0.5, 126  # one trading "episode": 6 months, daily rebalancing
FIT_WINDOW = 120  # trailing days of returns the HMM is refit on before each episode's decision


def stance_from_regime_prob(p_turbulent: float, confidence: float = 0.65) -> str:  # confidence: how sure the detector must be before it commits to a stance
    """Long gamma if the detector is confident the turbulent regime is
    active, short gamma if confident it's calm, flat (sit out) if unsure.
    A real strategy has to be able to say "I don't know" instead of always
    taking a position."""
    if p_turbulent > confidence:
        return "long"  # confident it's turbulent: be long gamma (buy convexity, pay theta)
    if p_turbulent < (1 - confidence):
        return "short"  # confident it's calm: be short gamma (sell convexity, collect theta)
    return "flat"  # neither confident: sit out entirely rather than guess


def run_episode(seed: int, strategy: str) -> float:  # strategy: "short", "long", or "regime"; returns that one episode's final P&L
    """One 6-month episode: simulate a regime-switching path, decide a
    stance (for 'regime', via the HMM; fixed for the baselines), build the
    corresponding straddle, and delta-hedge it to expiry. Returns final P&L."""
    rng = np.random.default_rng(seed)  # one generator per episode, driving BOTH the warmup and episode paths below, for full reproducibility

    burn_in = FIT_WINDOW + 5  # a few extra days of slack beyond FIT_WINDOW, so the trailing-window slice below always has enough history
    warmup_prices, warmup_states = regime_switching_gbm_path(S0, burn_in, T / N_STEPS, R, VOLS_TRUE, TRANSMAT_TRUE, rng, initial_state=rng.integers(2))  # history the HMM gets to see; random initial regime
    episode_prices, episode_states = regime_switching_gbm_path(warmup_prices[-1], N_STEPS, T / N_STEPS, R, VOLS_TRUE, TRANSMAT_TRUE, rng, initial_state=warmup_states[-1])  # the actual hedged episode, continuing from where warmup left off

    if strategy == "regime":
        trailing_returns = np.diff(np.log(warmup_prices))[-FIT_WINDOW:]  # log-returns from ONLY the trailing FIT_WINDOW days: strictly causal, no lookahead into the episode itself
        params, _ = regime.fit(trailing_returns, n_states=2, n_iter=100, seed=seed)  # refit from scratch every episode; discards the log-likelihood trace (second return value)
        filtered = regime.filtered_state_probs(trailing_returns, params)  # the causal (not smoothed) regime probabilities, safe to use for a live decision
        turbulent_label = int(np.argmax(params.stdevs))  # the fitted state labels are arbitrary (0/1 could mean either regime); identify "turbulent" as whichever has the larger fitted stdev
        p_turbulent_now = filtered[-1, turbulent_label]  # the MOST RECENT filtered probability of being in the turbulent state: the actual decision input
        stance = stance_from_regime_prob(p_turbulent_now)
    elif strategy in ("long", "short"):
        stance = strategy  # the fixed baselines just take their name as the stance, no detection involved
    else:
        raise ValueError(strategy)  # guards against a typo'd strategy name silently doing nothing

    if stance == "flat":
        return 0.0  # sitting out has zero P&L by construction, no structure to build or hedge

    entry_market = MarketData(spot=episode_prices[0], rate=R, vol=HEDGE_VOL)  # priced at the EPISODE's starting spot, always at the fixed HEDGE_VOL regardless of stance
    name = "straddle" if stance == "long" else "short_straddle"  # maps the stance string onto the two structure names create_structure understands
    structure = create_structure(name, K=K, T=T, market=entry_market, engine=ENGINE)

    # `simulate_delta_hedge` draws its own fresh path internally, which would hedge
    # a *different* random path than the one the stance decision above was actually
    # based on, so the hedge P&L has to be computed along this exact episode_prices
    # path instead, via the small local reimplementation below.
    return _hedge_along_path(structure, episode_prices, R, HEDGE_VOL, T, N_STEPS)


def _hedge_along_path(structure, path: np.ndarray, r: float, hedge_vol: float, maturity: float, n_steps: int) -> float:  # a path-driven twin of hedging.simulate_delta_hedge, for when the path already exists
    """Same self-financing delta-hedge accounting as `simulate_delta_hedge`,
    but driven by an externally supplied path instead of simulating its own.
    Needed here so the hedge P&L is computed along the *actual* regime-
    switching path the stance decision was based on."""
    from optionspricer.structures.base import mark_to_market, portfolio_greeks  # imported locally (not at module top) to keep this helper's dependencies visibly scoped to just this function
    from optionspricer.hedging import _shift  # reuses hedging.py's private leg-aging helper rather than duplicating its logic

    dt = maturity / n_steps  # same rebalancing-interval calculation as simulate_delta_hedge

    def hedge_market(S):  # local closure: builds a MarketData at a given spot, ALWAYS at hedge_vol, same pattern as hedging.py's hedge_market
        return MarketData(spot=S, rate=r, vol=hedge_vol)

    entry_cost = structure.entry_cost  # the net premium paid (or received) to put the structure on
    cash = -entry_cost  # same sign convention as hedging.py: a debit spends cash, a credit adds it
    shares = 0.0  # stock held for hedging; starts flat
    for i in range(n_steps):
        elapsed = i * dt  # time since inception, at the start of this rebalance
        S_i = path[i]  # today's spot, taken from the SUPPLIED path rather than a freshly simulated one
        g = portfolio_greeks(_shift(structure, elapsed), hedge_market(S_i), ENGINE)  # portfolio delta/gamma as of right now, priced at hedge_vol
        target_shares = -g.delta  # hold enough stock to offset the option legs' delta exactly, same definition as hedging.py
        cash -= (target_shares - shares) * S_i  # buy/sell the difference at today's price, self-financing
        shares = target_shares
        cash *= np.exp(r * dt)  # cash accrues interest at the risk-free rate between rebalances

    elapsed_next = n_steps * dt  # time elapsed at the FINAL rebalance, i.e. at maturity
    S_next = path[n_steps]  # the path's terminal spot
    raw_option_value = mark_to_market(structure, hedge_market(S_next), elapsed_next, ENGINE) + entry_cost  # same accounting as hedging.py: undo mark_to_market's own entry_cost netting
    return float(cash + shares * S_next + raw_option_value)  # total book value at expiry: cash + stock leg + option leg


N_EPISODES = 400  # independent regime-switching episodes averaged per strategy below
results = {name: np.array([run_episode(seed, name) for seed in range(N_EPISODES)]) for name in ["short", "long", "regime"]}  # one array of 400 final P&Ls per strategy; same seeds 0..399 reused across all three for a fair, paired comparison

print(f"Hedge vol (priced-in, blend of regimes) = {HEDGE_VOL:.3f}; true regime vols = {VOLS_TRUE}\n")
print(f"{'strategy':<10} {'mean P&L':>10} {'std':>10} {'SE':>8} {'win rate':>9} {'sharpe-like':>12} {'p5':>8} {'p95':>8}")  # column headers, aligned to match the formatted rows below
for name, pnl in results.items():
    mean, std = pnl.mean(), pnl.std(ddof=1)
    se = std / np.sqrt(len(pnl))  # standard error of the mean, ddof=1 for the sample standard deviation
    win_rate = (pnl > 0).mean()  # fraction of episodes with positive P&L; a boolean array's mean is the proportion of True values
    sharpe = mean / std if std > 0 else float("nan")  # a crude, per-episode Sharpe-like ratio (mean over std), guarded against a zero-std edge case
    p5, p95 = np.percentile(pnl, [5, 95])  # the 5th and 95th percentiles, characterizing the tails of the P&L distribution
    print(f"{name:<10} {mean:>10.3f} {std:>10.3f} {se:>8.3f} {win_rate:>9.1%} {sharpe:>12.3f} {p5:>8.3f} {p95:>8.3f}")

improvement_vs_best_baseline = results["regime"].mean() - max(results["short"].mean(), results["long"].mean())  # the honest headline number: regime strategy vs. whichever fixed baseline actually did better
print(f"\nRegime strategy mean P&L minus the better fixed baseline: {improvement_vs_best_baseline:+.3f}")

# Does the regime edge actually come from detection accuracy? Refit at several
# window lengths and check classification accuracy against the (simulation-only,
# never used by the strategy itself) ground-truth regime label at the decision
# point. A short window means too few observations for 2-state EM to separate
# "calm" from "turbulent" reliably, so accuracy should rise with window length.
print(f"\n{'fit window':>10} {'accuracy vs. true regime':>26} {'n decisive calls':>18}")
for window in [40, 80, 120, 200]:  # four trailing-window lengths, from noticeably data-starved to comfortably large
    correct = total = 0  # both start at 0; total only counts DECISIVE calls (stance != "flat"), correct counts how many of those matched the true regime
    for seed in range(200):  # 200 seeds per window length, fewer than the 400 used in the main comparison above, since this section only needs the accuracy trend
        rng = np.random.default_rng(seed)
        burn_in = window + 5  # same slack rationale as run_episode above, sized to THIS window length
        warmup_prices, warmup_states = regime_switching_gbm_path(S0, burn_in, T / N_STEPS, R, VOLS_TRUE, TRANSMAT_TRUE, rng, initial_state=rng.integers(2))
        trailing_returns = np.diff(np.log(warmup_prices))[-window:]  # the trailing `window` days of returns, the ONLY data this fit is allowed to see
        params, _ = regime.fit(trailing_returns, n_states=2, n_iter=100, seed=seed)
        filtered = regime.filtered_state_probs(trailing_returns, params)
        turbulent_label = int(np.argmax(params.stdevs))  # same label-identification trick as run_episode
        stance = stance_from_regime_prob(filtered[-1, turbulent_label])
        if stance == "flat":
            continue  # "I don't know" calls are excluded from the accuracy denominator, not counted as wrong
        true_stance = "long" if warmup_states[-1] == 1 else "short"  # the GROUND TRUTH regime at the decision point, from the simulator, never seen by the HMM itself
        total += 1
        correct += stance == true_stance  # True/False added to an int: True behaves as 1, False as 0
    print(f"{window:>10} {correct / total:>26.1%} {total:>18}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))  # three panels: P&L distributions, mean P&L per strategy, one example path with filtered probability overlay
fig.suptitle("Regime-aware gamma scalping vs. fixed long/short-gamma baselines", fontweight="bold")

ax = axes[0]  # panel 1: how do the three strategies' P&L distributions compare, overlaid?
bins = np.linspace(min(p.min() for p in results.values()), max(p.max() for p in results.values()), 40)  # shared bin edges across all three histograms, spanning the full range of all results combined
for name, color in zip(results, ["firebrick", "seagreen", "steelblue"]):  # iterating a dict yields its keys; zip pairs each strategy name with a fixed color
    ax.hist(results[name], bins=bins, alpha=0.5, label=name, color=color)  # alpha=0.5: semi-transparent, so overlapping histograms stay visible
ax.axvline(0, color="black", lw=0.8)  # zero-P&L reference line
ax.set_xlabel("episode P&L ($)")
ax.set_ylabel("count")
ax.set_title(f"P&L distribution across {N_EPISODES} episodes")
ax.legend()

ax = axes[1]  # panel 2: a simple bar chart of mean P&L per strategy, with confidence intervals
means = [results[n].mean() for n in results]  # list, not array: matplotlib's bar() accepts either
ses = [results[n].std(ddof=1) / np.sqrt(N_EPISODES) for n in results]
ax.bar(list(results.keys()), means, yerr=[1.96 * s for s in ses], capsize=5, color=["firebrick", "seagreen", "steelblue"])  # yerr: 95% CI error bars on each bar
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("mean P&L ($, +/- 95% CI)")
ax.set_title("Mean P&L per strategy")

ax = axes[2]  # panel 3: one concrete example, showing price alongside the HMM's filtered turbulent-regime probability over time
rng = np.random.default_rng(11)  # a fixed seed chosen just for a representative, reproducible example plot
warmup_prices, warmup_states = regime_switching_gbm_path(S0, FIT_WINDOW + 5, T / N_STEPS, R, VOLS_TRUE, TRANSMAT_TRUE, rng, initial_state=0)  # this example always starts in the calm regime (initial_state=0), for a clean illustrative plot
episode_prices, episode_states = regime_switching_gbm_path(warmup_prices[-1], N_STEPS, T / N_STEPS, R, VOLS_TRUE, TRANSMAT_TRUE, rng, initial_state=warmup_states[-1])
full_prices = np.concatenate([warmup_prices, episode_prices[1:]])  # [1:]: episode_prices[0] duplicates warmup_prices[-1], so drop it to avoid double-counting that day
full_states = np.concatenate([warmup_states, episode_states[1:]])  # same de-duplication, applied to the parallel ground-truth-state array
returns = np.diff(np.log(full_prices))  # log-returns across the FULL combined history, for this illustrative fit (unlike run_episode, which only fits the trailing window)
params, _ = regime.fit(returns, n_states=2, n_iter=100, seed=0)
turbulent_label = int(np.argmax(params.stdevs))
filtered = regime.filtered_state_probs(returns, params)[:, turbulent_label]  # just the turbulent-state column, one probability per return
ax2 = ax.twinx()  # a second y-axis sharing the same x-axis, so price and probability can share one panel at different scales
ax.plot(full_prices, color="steelblue", lw=1)
ax.set_ylabel("price", color="steelblue")
ax2.plot(np.arange(1, len(full_prices)), filtered, color="firebrick", lw=1.2, alpha=0.8)  # starts at index 1: there's no return (and hence no filtered probability) for day 0
ax2.fill_between(np.arange(1, len(full_prices)), 0, full_states[1:], step="pre", alpha=0.15, color="grey")  # shaded background showing the TRUE (hidden) regime, for visual comparison against the filtered estimate
ax2.set_ylabel("P(turbulent | data so far)", color="firebrick")
ax.set_xlabel("trading day")
ax.set_title("One example path: filtered regime probability vs. price")

plt.tight_layout()
plt.savefig("experiments/results/regime_gamma_scalping.png", dpi=150)
print("\nSaved experiments/results/regime_gamma_scalping.png")
