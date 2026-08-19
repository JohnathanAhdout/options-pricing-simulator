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

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

import matplotlib  # imported separately from pyplot so the backend can be set BEFORE pyplot is imported, below

matplotlib.use("Agg")  # non-interactive, file-only backend: this script never opens a window, just saves a PNG
import matplotlib.pyplot as plt
import numpy as np

from optionspricer.market import MarketData, OptionSpec, OptionType  # MarketData/OptionSpec imported but unused directly here; the pricing calls below use plain floats instead
from optionspricer.pricing.black_scholes import price as bs_price  # the known-exact reference price this experiment measures Monte Carlo against
from optionspricer.pricing.monte_carlo import price as mc_price  # the function under test
from optionspricer.simulation import gbm_terminal  # used only for the third subplot's histogram, not for pricing itself

S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.20  # fixed contract/market scenario every M in this sweep reuses
option_type = OptionType.CALL
true_price = bs_price(S, K, T, r, sigma, option_type)  # the ground truth every Monte Carlo estimate below is compared against

M_VALUES = np.array([100, 200, 500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 500_000, 1_000_000])  # path counts spanning 4 orders of magnitude, log-spaced by eye
rng = np.random.default_rng(0)  # ONE generator, reused (not reset) across every M below, so successive estimates share a common random stream

mc_prices, ci_widths, errors = [], [], []  # three parallel accumulators, one entry appended per M in the loop below
for M in M_VALUES:
    price, se = mc_price(S, K, T, r, sigma, option_type, n_paths=int(M), rng=rng)  # local var `price` shadows the imported bs_price name only within this loop body
    mc_prices.append(price)
    ci_widths.append(2 * 1.96 * se)  # a 95% CI is +/- 1.96 standard errors, so its full width is 2*1.96*se
    errors.append(abs(price - true_price))  # this run's actual error against the known-exact price

mc_prices, ci_widths, errors = map(np.array, (mc_prices, ci_widths, errors))  # convert all three lists to arrays at once, for the vectorized fitting and plotting below

print(f"Black-Scholes price: {true_price:.6f}\n")
print(f"{'M':>10} {'MC price':>12} {'95% CI width':>14} {'|error|':>12}")  # column headers, right-aligned to match the formatted rows below
for M, p, w, e in zip(M_VALUES, mc_prices, ci_widths, errors):
    print(f"{M:>10} {p:>12.6f} {w:>14.6f} {e:>12.6f}")

# fit log(error) ~ slope * log(M) + const, expect slope ~= -0.5
log_M, log_err = np.log(M_VALUES), np.log(errors)  # log-log space: a power law M^slope becomes a straight line here, fittable by ordinary least squares
slope_err, _ = np.polyfit(log_M, log_err, 1)  # degree-1 fit; slope_err is the fitted exponent, the intercept (second return value) is discarded
slope_ci, _ = np.polyfit(np.log(M_VALUES), np.log(ci_widths), 1)  # same fit, applied to CI width instead of raw error
print(f"\nFitted convergence rate (error):     M^{slope_err:.3f}  (theory: -0.5)")
print(f"Fitted convergence rate (CI width):  M^{slope_ci:.3f}  (theory: -0.5)")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))  # three panels side by side: estimate-vs-truth, error decay, terminal distribution
fig.suptitle(f"Monte Carlo convergence to Black-Scholes  (S={S}, K={K}, T={T}, r={r}, sigma={sigma})", fontweight="bold")

ax = axes[0]  # panel 1: does the MC estimate + its CI actually bracket the true price as M grows?
ax.axhline(true_price, color="firebrick", lw=2, label=f"BS price = {true_price:.4f}")  # horizontal reference line at the known-exact price
ax.plot(M_VALUES, mc_prices, "o-", color="steelblue", label="MC estimate")
ax.fill_between(M_VALUES, mc_prices - ci_widths / 2, mc_prices + ci_widths / 2, alpha=0.25, color="steelblue", label="95% CI")  # shaded band showing the CI shrinking as M grows
ax.set_xscale("log")  # M spans 4 orders of magnitude; a log x-axis keeps small-M points from being crushed together
ax.set_xlabel("paths M")
ax.set_ylabel("price")
ax.set_title("Estimate vs. ground truth")
ax.legend()

ax = axes[1]  # panel 2: does the error actually decay at the theoretical M^-0.5 rate?
ax.loglog(M_VALUES, errors, "o-", color="steelblue", label="|MC - BS|")  # log-log axes: a power law appears as a straight line here
ref = errors[0] * np.sqrt(M_VALUES[0] / M_VALUES)  # a theoretical M^-0.5 reference curve, anchored to match the first data point exactly
ax.loglog(M_VALUES, ref, "--", color="darkorange", label=r"$M^{-1/2}$ reference")  # dashed reference line for visual comparison against the actual error curve
ax.set_xlabel("paths M")
ax.set_ylabel("absolute error")
ax.set_title(f"Error decay (fitted slope {slope_err:.3f})")
ax.legend()

ax = axes[2]  # panel 3: what does the simulated terminal-price distribution actually look like?
_, _, ST, _ = (lambda ST: (None, None, ST, None))(gbm_terminal(S, T, r, sigma, 0.0, 200_000, np.random.default_rng(1)))  # a roundabout way of writing ST = gbm_terminal(...); the lambda/unpacking does nothing functional here
ax.hist(ST, bins=100, color="steelblue", alpha=0.75, density=True)  # density=True: normalized to a probability density, not raw counts
ax.axvline(K, color="firebrick", ls="--", label=f"K = {K}")  # marks the strike on the distribution
ax.axvline(S, color="seagreen", ls="--", label=f"S0 = {S}")  # marks today's spot on the distribution, for comparison against K
ax.set_xlabel("terminal price $S_T$")
ax.set_title("Simulated terminal distribution (200k paths)")
ax.legend()

plt.tight_layout()  # adjusts spacing so subplot titles/labels don't overlap
plt.savefig("experiments/results/mc_convergence.png", dpi=150)  # the actual output artifact this script produces
print("\nSaved experiments/results/mc_convergence.png")
