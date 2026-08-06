"""The flagship experiment, part 1: does delta-hedging P&L actually behave
the way the theory in `hedging.py` says it should?

A market maker sells a straddle at an implied vol sigma_h and delta-hedges
it continuously. `hedging.py` derives (and BACKGROUND.md re-derives from
scratch) a closed-form prediction for the resulting P&L in terms of gamma
and the *gap* between what volatility the stock actually realizes and what
volatility was priced in:

    P&L = -0.5 * integral[ Gamma(t) * S(t)^2 * (sigma_realized^2 - sigma_h^2) dt ]

(negative sign because a short straddle's gamma is itself negative -- see
`portfolio_greeks`). Three things get checked against simulation here:

1. Sweeping realized vol away from the hedge vol in both directions --
   does the mean simulated P&L actually track the theoretical curve, and
   is the sign right (short gamma wins when the world is calmer than
   priced, loses when it's wilder)?
2. Holding realized == hedge vol -- is the hedge P&L unbiased, i.e. does
   selling "fair" volatility and hedging it perfectly break even in
   expectation, exactly as theory demands?
3. Rebalancing frequency -- does hedging more often leave the *mean* P&L
   basically where it was (theory says discretization doesn't bias the
   result) while shrinking its *variance* (theory says the hedging-error
   variance should fall as rebalancing gets more frequent)?

Run: uv run python experiments/run_gamma_theta_pnl.py
Output: experiments/results/gamma_theta_pnl.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optionspricer.hedging import simulate_delta_hedge
from optionspricer.market import MarketData
from optionspricer.pricing.black_scholes import BlackScholesEngine
from optionspricer.structures import create_structure

ENGINE = BlackScholesEngine()
S0, R, K, T, HEDGE_VOL = 100.0, 0.05, 100.0, 0.5, 0.20
N_SEEDS = 500
N_STEPS = 126  # ~daily rebalancing over 6 months


def run_sweep(realized_vols, n_steps=N_STEPS, n_seeds=N_SEEDS):
    entry_market = MarketData(spot=S0, rate=R, vol=HEDGE_VOL)
    structure = create_structure("short_straddle", K=K, T=T, market=entry_market, engine=ENGINE)
    means, ses, theory_means = [], [], []
    for sr in realized_vols:
        finals = np.empty(n_seeds)
        theos = np.empty(n_seeds)
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            result = simulate_delta_hedge(structure, S0, R, HEDGE_VOL, sr, T, n_steps, ENGINE, rng)
            finals[seed] = result.final_pnl
            theos[seed] = result.theoretical_pnl
        means.append(finals.mean())
        ses.append(finals.std(ddof=1) / np.sqrt(n_seeds))
        theory_means.append(theos.mean())
    return np.array(means), np.array(ses), np.array(theory_means)


realized_vols = np.array([0.08, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32, 0.38])
sim_means, sim_ses, theory_means = run_sweep(realized_vols)

print(f"Short straddle, sold at hedge_vol={HEDGE_VOL}, K={K}, T={T}, {N_STEPS} rebalances, {N_SEEDS} seeds\n")
print(f"{'realized_vol':>13} {'sim mean P&L':>13} {'sim SE':>9} {'theory P&L':>12} {'sim-theory':>11}")
for sr, m, se, th in zip(realized_vols, sim_means, sim_ses, theory_means):
    print(f"{sr:>13.2f} {m:>13.3f} {se:>9.3f} {th:>12.3f} {m - th:>11.3f}")

# unbiasedness check at realized == hedge
idx_atm = np.argmin(np.abs(realized_vols - HEDGE_VOL))
print(f"\nAt realized_vol == hedge_vol ({HEDGE_VOL}): mean P&L = {sim_means[idx_atm]:.3f} +/- {sim_ses[idx_atm]:.3f} (1 SE)")

# rebalancing-frequency experiment: mean should hold, variance should shrink
print("\nRebalancing frequency (realized_vol=0.30, fixed away from hedge_vol to keep signal strong):")
freq_grid = [6, 12, 26, 63, 126, 252]
freq_means, freq_stds = [], []
entry_market = MarketData(spot=S0, rate=R, vol=HEDGE_VOL)
structure = create_structure("short_straddle", K=K, T=T, market=entry_market, engine=ENGINE)
for n_steps in freq_grid:
    finals = np.empty(300)
    for seed in range(300):
        rng = np.random.default_rng(seed)
        finals[seed] = simulate_delta_hedge(structure, S0, R, HEDGE_VOL, 0.30, T, n_steps, ENGINE, rng).final_pnl
    freq_means.append(finals.mean())
    freq_stds.append(finals.std(ddof=1))
    print(f"  n_steps={n_steps:>4}  mean={finals.mean():>8.3f}  std={finals.std(ddof=1):>8.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Delta-hedged short straddle: simulated P&L vs. closed-form gamma/theta prediction", fontweight="bold")

ax = axes[0]
ax.errorbar(realized_vols, sim_means, yerr=1.96 * sim_ses, fmt="o", color="steelblue", label="simulated (95% CI)", capsize=3)
ax.plot(realized_vols, theory_means, "-", color="firebrick", lw=2, label="theory: 0.5*Gamma*S^2*(sigma_h^2-sigma_r^2)*dt")
ax.axvline(HEDGE_VOL, color="grey", ls=":", label=f"hedge_vol = {HEDGE_VOL}")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("realized volatility")
ax.set_ylabel("final P&L ($)")
ax.set_title("P&L vs. realized vol")
ax.legend(fontsize=8)

ax = axes[1]
example_rng = np.random.default_rng(1)
result = simulate_delta_hedge(structure, S0, R, HEDGE_VOL, 0.30, T, N_STEPS, ENGINE, example_rng)
ax.plot(result.times, result.pnl_path, color="steelblue")
ax.axhline(0, color="black", lw=0.6, ls="--")
ax.set_xlabel("time (years)")
ax.set_ylabel("running P&L ($)")
ax.set_title("One example path (realized=0.30 > hedge=0.20: losing)")

ax = axes[2]
ax.plot(freq_grid, freq_stds, "o-", color="seagreen")
ax.set_xlabel("rebalances over the option's life")
ax.set_ylabel("P&L std dev across seeds ($)")
ax.set_title("Hedging error shrinks with rebalancing frequency")

plt.tight_layout()
plt.savefig("experiments/results/gamma_theta_pnl.png", dpi=150)
print("\nSaved experiments/results/gamma_theta_pnl.png")
