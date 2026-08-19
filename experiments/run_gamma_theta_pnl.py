"""The flagship experiment, part 1: does delta-hedging P&L actually behave
the way the theory in `hedging.py` says it should?

A market maker sells a straddle at an implied vol sigma_h and delta-hedges
it continuously. `hedging.py` derives (and BACKGROUND.md re-derives from
scratch) a closed-form prediction for the resulting P&L in terms of gamma
and the *gap* between what volatility the stock actually realizes and what
volatility was priced in:

    P&L = -0.5 * integral[ Gamma(t) * S(t)^2 * (sigma_realized^2 - sigma_h^2) dt ]

(negative sign because a short straddle's gamma is itself negative; see
`portfolio_greeks`). Three things get checked against simulation here:

1. Sweeping realized vol away from the hedge vol in both directions:
   does the mean simulated P&L actually track the theoretical curve, and
   is the sign right (short gamma wins when the world is calmer than
   priced, loses when it's wilder)?
2. Holding realized == hedge vol: is the hedge P&L unbiased, i.e. does
   selling "fair" volatility and hedging it perfectly break even in
   expectation, exactly as theory demands?
3. Rebalancing frequency: does hedging more often leave the *mean* P&L
   basically where it was (theory says discretization doesn't bias the
   result) while shrinking its *variance* (theory says the hedging-error
   variance should fall as rebalancing gets more frequent)?

Run: uv run python experiments/run_gamma_theta_pnl.py
Output: experiments/results/gamma_theta_pnl.png
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

import matplotlib  # imported separately from pyplot so the backend can be set BEFORE pyplot is imported, below

matplotlib.use("Agg")  # non-interactive, file-only backend: this script never opens a window, just saves a PNG
import matplotlib.pyplot as plt
import numpy as np

from optionspricer.hedging import simulate_delta_hedge  # the mechanical hedge simulation this whole script is built around
from optionspricer.market import MarketData
from optionspricer.pricing.black_scholes import BlackScholesEngine  # the pricing engine used to build and hedge the straddle
from optionspricer.structures import create_structure  # builds the short straddle via the factory, not by hand

ENGINE = BlackScholesEngine()  # one shared engine instance, reused everywhere below
S0, R, K, T, HEDGE_VOL = 100.0, 0.05, 100.0, 0.5, 0.20  # fixed spot/rate/strike/maturity/hedge-vol scenario every section reuses
N_SEEDS = 500  # independent simulated episodes averaged per realized-vol point in the sweep below
N_STEPS = 126  # ~daily rebalancing over 6 months


def run_sweep(realized_vols, n_steps=N_STEPS, n_seeds=N_SEEDS):  # runs N_SEEDS independent hedge simulations at each realized vol in the input array
    entry_market = MarketData(spot=S0, rate=R, vol=HEDGE_VOL)  # the market snapshot the straddle is PRICED and SOLD into, at hedge_vol
    structure = create_structure("short_straddle", K=K, T=T, market=entry_market, engine=ENGINE)  # built ONCE, reused across every realized vol and every seed below
    means, ses, theory_means = [], [], []  # three parallel accumulators, one entry appended per realized-vol point
    for sr in realized_vols:
        finals = np.empty(n_seeds)  # preallocated; filled with one simulated final P&L per seed below
        theos = np.empty(n_seeds)  # preallocated; filled with one theoretical P&L per seed below
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)  # a fresh, seeded generator per episode: each seed is an independent simulated world
            result = simulate_delta_hedge(structure, S0, R, HEDGE_VOL, sr, T, n_steps, ENGINE, rng)  # sr: this iteration's realized vol; HEDGE_VOL stays fixed, only the realized path's vol changes
            finals[seed] = result.final_pnl
            theos[seed] = result.theoretical_pnl
        means.append(finals.mean())  # average simulated P&L across all n_seeds episodes at this realized vol
        ses.append(finals.std(ddof=1) / np.sqrt(n_seeds))  # standard error of that mean, ddof=1 for the sample (not population) standard deviation
        theory_means.append(theos.mean())  # average of the closed-form prediction, computed along each seed's own simulated path
    return np.array(means), np.array(ses), np.array(theory_means)


realized_vols = np.array([0.08, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32, 0.38])  # swept both below and above HEDGE_VOL=0.20, denser near it
sim_means, sim_ses, theory_means = run_sweep(realized_vols)  # runs the full 500-seed-per-point sweep defined above

print(f"Short straddle, sold at hedge_vol={HEDGE_VOL}, K={K}, T={T}, {N_STEPS} rebalances, {N_SEEDS} seeds\n")
print(f"{'realized_vol':>13} {'sim mean P&L':>13} {'sim SE':>9} {'theory P&L':>12} {'sim-theory':>11}")  # column headers, aligned to match the formatted rows below
for sr, m, se, th in zip(realized_vols, sim_means, sim_ses, theory_means):
    print(f"{sr:>13.2f} {m:>13.3f} {se:>9.3f} {th:>12.3f} {m - th:>11.3f}")

# unbiasedness check at realized == hedge
idx_atm = np.argmin(np.abs(realized_vols - HEDGE_VOL))  # finds the sweep point closest to realized_vol == hedge_vol, since 0.20 is exactly in realized_vols anyway
print(f"\nAt realized_vol == hedge_vol ({HEDGE_VOL}): mean P&L = {sim_means[idx_atm]:.3f} +/- {sim_ses[idx_atm]:.3f} (1 SE)")

# rebalancing-frequency experiment: mean should hold, variance should shrink
print("\nRebalancing frequency (realized_vol=0.30, fixed away from hedge_vol to keep signal strong):")
freq_grid = [6, 12, 26, 63, 126, 252]  # rebalances per 6-month episode, from roughly monthly to twice-daily
freq_means, freq_stds = [], []  # two parallel accumulators, one entry appended per frequency below
entry_market = MarketData(spot=S0, rate=R, vol=HEDGE_VOL)  # rebuilt here (not reused from run_sweep's local scope) since this section runs independently
structure = create_structure("short_straddle", K=K, T=T, market=entry_market, engine=ENGINE)  # a fresh structure instance, same shape as before
for n_steps in freq_grid:
    finals = np.empty(300)  # 300 seeds per frequency point, fewer than the 500 used in the main sweep above (this section only needs the variance trend, not tight point estimates)
    for seed in range(300):
        rng = np.random.default_rng(seed)
        finals[seed] = simulate_delta_hedge(structure, S0, R, HEDGE_VOL, 0.30, T, n_steps, ENGINE, rng).final_pnl  # realized vol fixed at 0.30 throughout; only n_steps (rebalancing frequency) varies
    freq_means.append(finals.mean())
    freq_stds.append(finals.std(ddof=1))
    print(f"  n_steps={n_steps:>4}  mean={finals.mean():>8.3f}  std={finals.std(ddof=1):>8.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))  # three panels: P&L vs. realized vol, one example path, hedging error vs. frequency
fig.suptitle("Delta-hedged short straddle: simulated P&L vs. closed-form gamma/theta prediction", fontweight="bold")

ax = axes[0]  # panel 1: does the simulated mean P&L track the theoretical curve across the whole realized-vol sweep?
ax.errorbar(realized_vols, sim_means, yerr=1.96 * sim_ses, fmt="o", color="steelblue", label="simulated (95% CI)", capsize=3)  # yerr=1.96*se: a 95% confidence interval on each point
ax.plot(realized_vols, theory_means, "-", color="firebrick", lw=2, label="theory: 0.5*Gamma*S^2*(sigma_h^2-sigma_r^2)*dt")
ax.axvline(HEDGE_VOL, color="grey", ls=":", label=f"hedge_vol = {HEDGE_VOL}")  # marks the point where realized should equal hedge (the unbiasedness check)
ax.axhline(0, color="black", lw=0.6)  # zero-P&L reference line
ax.set_xlabel("realized volatility")
ax.set_ylabel("final P&L ($)")
ax.set_title("P&L vs. realized vol")
ax.legend(fontsize=8)

ax = axes[1]  # panel 2: what does a single simulated hedge's running P&L actually look like over time?
example_rng = np.random.default_rng(1)  # a fixed seed, chosen just for a representative, reproducible example plot
result = simulate_delta_hedge(structure, S0, R, HEDGE_VOL, 0.30, T, N_STEPS, ENGINE, example_rng)  # realized=0.30 > hedge=0.20: expected to be a losing path on average, per the theory
ax.plot(result.times, result.pnl_path, color="steelblue")
ax.axhline(0, color="black", lw=0.6, ls="--")
ax.set_xlabel("time (years)")
ax.set_ylabel("running P&L ($)")
ax.set_title("One example path (realized=0.30 > hedge=0.20: losing)")

ax = axes[2]  # panel 3: does the standard deviation across seeds actually shrink as rebalancing gets more frequent?
ax.plot(freq_grid, freq_stds, "o-", color="seagreen")
ax.set_xlabel("rebalances over the option's life")
ax.set_ylabel("P&L std dev across seeds ($)")
ax.set_title("Hedging error shrinks with rebalancing frequency")

plt.tight_layout()
plt.savefig("experiments/results/gamma_theta_pnl.png", dpi=150)
print("\nSaved experiments/results/gamma_theta_pnl.png")
