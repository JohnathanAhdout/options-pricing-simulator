"""Two questions for the binomial tree: does it converge to Black-Scholes
(a sanity check that the lattice and the closed form agree, which is real
evidence both are right, not just mutually consistent), and how big is the
early-exercise premium an American option actually commands over its
European counterpart?

Textbook fact under test: with no dividends, an American *call* is never
worth exercising early (you'd throw away remaining time value for nothing,
since the alternative of just selling the call captures the same payoff
plus whatever time value is left), so American call == European call.
American *puts* have no such shield: a deep ITM put on a stock that
can't go below zero has an early-exercise incentive from time value on
the strike alone, so the premium should show up clearly there, and
should grow as the put gets deeper in the money.

Run: uv run python experiments/run_american_premium.py
Output: experiments/results/american_premium.png
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

import matplotlib  # imported separately from pyplot so the backend can be set BEFORE pyplot is imported, below

matplotlib.use("Agg")  # non-interactive, file-only backend: this script never opens a window, just saves a PNG
import matplotlib.pyplot as plt
import numpy as np

from optionspricer.market import ExerciseStyle, OptionType
from optionspricer.pricing.binomial import price as binomial_price  # the engine under test throughout this script
from optionspricer.pricing.black_scholes import price as bs_price  # the closed-form reference the convergence check compares against

S, r, sigma, T = 100.0, 0.05, 0.25, 1.0  # fixed spot/rate/vol/maturity scenario every section below reuses

# --- convergence to Black-Scholes (European, both option types) ---
steps_grid = np.array([5, 10, 25, 50, 100, 250, 500, 1000, 2000])  # tree resolutions spanning roughly 3 orders of magnitude
print("Binomial -> Black-Scholes convergence (European call, ATM):")
print(f"{'n_steps':>8} {'binomial price':>16} {'|error|':>12}")  # column headers, aligned to match the formatted rows below
true_call = bs_price(S, S, T, r, sigma, OptionType.CALL)  # S, S: strike equals spot, i.e. exactly at the money; the ground truth every n_steps below is measured against
conv_errors = []  # accumulator: one error per n_steps in steps_grid
for n in steps_grid:
    p = binomial_price(S, S, T, r, sigma, OptionType.CALL, ExerciseStyle.EUROPEAN, n_steps=int(n))  # European exercise: this section is purely about convergence, not early exercise
    err = abs(p - true_call)
    conv_errors.append(err)
    print(f"{n:>8} {p:>16.6f} {err:>12.2e}")

# --- American call == European call (no dividends) ---
K_grid = np.linspace(70, 130, 13)  # 13 strikes from 30% OTM to 30% ITM, reused by this section and the next
call_diffs = []  # accumulator: American minus European price, one entry per strike; textbook says this should be ~0 everywhere
for K in K_grid:
    eu = binomial_price(S, K, T, r, sigma, OptionType.CALL, ExerciseStyle.EUROPEAN, n_steps=500)  # 500 steps: enough resolution that convergence error is negligible next to the effect being measured
    am = binomial_price(S, K, T, r, sigma, OptionType.CALL, ExerciseStyle.AMERICAN, n_steps=500)
    call_diffs.append(am - eu)
call_diffs = np.array(call_diffs)
print(f"\nAmerican - European call (no dividends), max diff across strikes: {np.max(np.abs(call_diffs)):.2e} (expect ~0)")

# --- American put premium across moneyness ---
premiums = []  # accumulator: American minus European PUT price, one entry per strike; unlike calls, this is expected to be strictly positive
for K in K_grid:
    eu = binomial_price(S, K, T, r, sigma, OptionType.PUT, ExerciseStyle.EUROPEAN, n_steps=500)
    am = binomial_price(S, K, T, r, sigma, OptionType.PUT, ExerciseStyle.AMERICAN, n_steps=500)
    premiums.append(am - eu)
premiums = np.array(premiums)

print(f"\n{'K':>6} {'K/S':>6} {'European put':>14} {'American put':>14} {'premium':>10} {'premium %':>10}")
for K, prem in zip(K_grid, premiums):
    eu = binomial_price(S, K, T, r, sigma, OptionType.PUT, ExerciseStyle.EUROPEAN, n_steps=500)  # recomputed here (not reused from the loop above) purely for this print statement's local scope
    am = eu + prem  # reconstructs the American price from the European price and the already-computed premium, rather than a third binomial_price call
    print(f"{K:>6.1f} {K/S:>6.2f} {eu:>14.4f} {am:>14.4f} {prem:>10.4f} {100*prem/eu if eu > 1e-9 else float('nan'):>9.1f}%")  # guards against a division by a near-zero European price

# --- premium vs. volatility, at a fixed deep-ITM put ---
K_itm = 120.0  # a strike well above spot (deep ITM for a put), where the early-exercise incentive is strongest
vol_grid = np.linspace(0.05, 0.6, 20)  # 20 volatility levels from 5% to 60%
premium_vs_vol = []  # accumulator: American minus European premium at K_itm, one entry per volatility level
for v in vol_grid:
    eu = binomial_price(S, K_itm, T, r, v, OptionType.PUT, ExerciseStyle.EUROPEAN, n_steps=500)
    am = binomial_price(S, K_itm, T, r, v, OptionType.PUT, ExerciseStyle.AMERICAN, n_steps=500)
    premium_vs_vol.append(am - eu)
premium_vs_vol = np.array(premium_vs_vol)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))  # three panels: convergence, premium vs. moneyness, premium vs. vol
fig.suptitle("CRR binomial tree: convergence and the American early-exercise premium", fontweight="bold")

ax = axes[0]  # panel 1: does the tree actually converge to Black-Scholes as steps grow?
ax.loglog(steps_grid, conv_errors, "o-", color="steelblue", label="|binomial - BS|")  # log-log axes: the O(1/N) convergence rate should appear as a straight line
ref = conv_errors[0] * (steps_grid[0] / steps_grid)  # a theoretical N^-1 reference curve, anchored to match the first data point exactly
ax.loglog(steps_grid, ref, "--", color="darkorange", label=r"$N^{-1}$ reference")
ax.set_xlabel("tree steps N")
ax.set_ylabel("absolute error vs. BS")
ax.set_title("European call: binomial -> Black-Scholes")
ax.legend()

ax = axes[1]  # panel 2: does the American put premium grow with moneyness, as the textbook predicts?
ax.plot(K_grid / S, premiums, "o-", color="firebrick", label="American - European put")
ax.axhline(0, color="black", lw=0.8, ls="--")  # zero-premium reference line
ax.axvline(1.0, color="grey", lw=0.8, ls=":", label="ATM")  # marks exactly at-the-money on the moneyness axis
ax.set_xlabel("moneyness K/S")
ax.set_ylabel("early-exercise premium ($)")
ax.set_title(f"Put early-exercise premium vs. moneyness (T={T}, sigma={sigma})")
ax.legend()

ax = axes[2]  # panel 3: how does the premium at a fixed deep-ITM strike depend on volatility?
ax.plot(vol_grid, premium_vs_vol, "o-", color="seagreen")
ax.set_xlabel("volatility sigma")
ax.set_ylabel("early-exercise premium ($)")
ax.set_title(f"Put premium vs. vol (K={K_itm}, deep ITM)")

plt.tight_layout()
plt.savefig("experiments/results/american_premium.png", dpi=150)
print("\nSaved experiments/results/american_premium.png")
