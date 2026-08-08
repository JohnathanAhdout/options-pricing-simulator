"""Does Monte Carlo actually converge to Black-Scholes at the rate theory
predicts?

Every M in `M_VALUES` reprices the same option from scratch with an
independent batch of paths, and we track three things against it:
the point estimate itself, the width of its 95% confidence interval, and
its raw error against the (known-exact) Black-Scholes price. All three
are predicted by the Central Limit Theorem to shrink as 1/sqrt(M). This
script checks that prediction against actual simulated numbers instead of
asserting it and moving on.

Run: uv run python experiments/run_mc_convergence.py
Output: experiments/results/mc_convergence.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optionspricer.market import MarketData, OptionSpec, OptionType
from optionspricer.pricing.black_scholes import price as bs_price
from optionspricer.pricing.monte_carlo import price as mc_price
from optionspricer.simulation import gbm_terminal

S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.20
option_type = OptionType.CALL
true_price = bs_price(S, K, T, r, sigma, option_type)

M_VALUES = np.array([100, 200, 500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 500_000, 1_000_000])
rng = np.random.default_rng(0)

mc_prices, ci_widths, errors = [], [], []
for M in M_VALUES:
    price, se = mc_price(S, K, T, r, sigma, option_type, n_paths=int(M), rng=rng)
    mc_prices.append(price)
    ci_widths.append(2 * 1.96 * se)
    errors.append(abs(price - true_price))

mc_prices, ci_widths, errors = map(np.array, (mc_prices, ci_widths, errors))

print(f"Black-Scholes price: {true_price:.6f}\n")
print(f"{'M':>10} {'MC price':>12} {'95% CI width':>14} {'|error|':>12}")
for M, p, w, e in zip(M_VALUES, mc_prices, ci_widths, errors):
    print(f"{M:>10} {p:>12.6f} {w:>14.6f} {e:>12.6f}")

# fit log(error) ~ slope * log(M) + const, expect slope ~= -0.5
log_M, log_err = np.log(M_VALUES), np.log(errors)
slope_err, _ = np.polyfit(log_M, log_err, 1)
slope_ci, _ = np.polyfit(np.log(M_VALUES), np.log(ci_widths), 1)
print(f"\nFitted convergence rate (error):     M^{slope_err:.3f}  (theory: -0.5)")
print(f"Fitted convergence rate (CI width):  M^{slope_ci:.3f}  (theory: -0.5)")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle(f"Monte Carlo convergence to Black-Scholes  (S={S}, K={K}, T={T}, r={r}, sigma={sigma})", fontweight="bold")

ax = axes[0]
ax.axhline(true_price, color="firebrick", lw=2, label=f"BS price = {true_price:.4f}")
ax.plot(M_VALUES, mc_prices, "o-", color="steelblue", label="MC estimate")
ax.fill_between(M_VALUES, mc_prices - ci_widths / 2, mc_prices + ci_widths / 2, alpha=0.25, color="steelblue", label="95% CI")
ax.set_xscale("log")
ax.set_xlabel("paths M")
ax.set_ylabel("price")
ax.set_title("Estimate vs. ground truth")
ax.legend()

ax = axes[1]
ax.loglog(M_VALUES, errors, "o-", color="steelblue", label="|MC - BS|")
ref = errors[0] * np.sqrt(M_VALUES[0] / M_VALUES)
ax.loglog(M_VALUES, ref, "--", color="darkorange", label=r"$M^{-1/2}$ reference")
ax.set_xlabel("paths M")
ax.set_ylabel("absolute error")
ax.set_title(f"Error decay (fitted slope {slope_err:.3f})")
ax.legend()

ax = axes[2]
_, _, ST, _ = (lambda ST: (None, None, ST, None))(gbm_terminal(S, T, r, sigma, 0.0, 200_000, np.random.default_rng(1)))
ax.hist(ST, bins=100, color="steelblue", alpha=0.75, density=True)
ax.axvline(K, color="firebrick", ls="--", label=f"K = {K}")
ax.axvline(S, color="seagreen", ls="--", label=f"S0 = {S}")
ax.set_xlabel("terminal price $S_T$")
ax.set_title("Simulated terminal distribution (200k paths)")
ax.legend()

plt.tight_layout()
plt.savefig("experiments/results/mc_convergence.png", dpi=150)
print("\nSaved experiments/results/mc_convergence.png")
