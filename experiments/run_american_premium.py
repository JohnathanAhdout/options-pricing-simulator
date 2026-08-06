"""Two questions for the binomial tree: does it converge to Black-Scholes
(a sanity check that the lattice and the closed form agree, which is real
evidence both are right, not just mutually consistent), and how big is the
early-exercise premium an American option actually commands over its
European counterpart?

Textbook fact under test: with no dividends, an American *call* is never
worth exercising early (you'd throw away remaining time value for nothing,
since the alternative of just selling the call captures the same payoff
plus whatever time value is left), so American call == European call.
American *puts* have no such shield -- a deep ITM put on a stock that
can't go below zero has an early-exercise incentive from time value on
the strike alone -- so the premium should show up clearly there, and
should grow as the put gets deeper in the money.

Run: uv run python experiments/run_american_premium.py
Output: experiments/results/american_premium.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optionspricer.market import ExerciseStyle, OptionType
from optionspricer.pricing.binomial import price as binomial_price
from optionspricer.pricing.black_scholes import price as bs_price

S, r, sigma, T = 100.0, 0.05, 0.25, 1.0

# --- convergence to Black-Scholes (European, both option types) ---
steps_grid = np.array([5, 10, 25, 50, 100, 250, 500, 1000, 2000])
print("Binomial -> Black-Scholes convergence (European call, ATM):")
print(f"{'n_steps':>8} {'binomial price':>16} {'|error|':>12}")
true_call = bs_price(S, S, T, r, sigma, OptionType.CALL)
conv_errors = []
for n in steps_grid:
    p = binomial_price(S, S, T, r, sigma, OptionType.CALL, ExerciseStyle.EUROPEAN, n_steps=int(n))
    err = abs(p - true_call)
    conv_errors.append(err)
    print(f"{n:>8} {p:>16.6f} {err:>12.2e}")

# --- American call == European call (no dividends) ---
K_grid = np.linspace(70, 130, 13)
call_diffs = []
for K in K_grid:
    eu = binomial_price(S, K, T, r, sigma, OptionType.CALL, ExerciseStyle.EUROPEAN, n_steps=500)
    am = binomial_price(S, K, T, r, sigma, OptionType.CALL, ExerciseStyle.AMERICAN, n_steps=500)
    call_diffs.append(am - eu)
call_diffs = np.array(call_diffs)
print(f"\nAmerican - European call (no dividends), max diff across strikes: {np.max(np.abs(call_diffs)):.2e} (expect ~0)")

# --- American put premium across moneyness ---
premiums = []
for K in K_grid:
    eu = binomial_price(S, K, T, r, sigma, OptionType.PUT, ExerciseStyle.EUROPEAN, n_steps=500)
    am = binomial_price(S, K, T, r, sigma, OptionType.PUT, ExerciseStyle.AMERICAN, n_steps=500)
    premiums.append(am - eu)
premiums = np.array(premiums)

print(f"\n{'K':>6} {'K/S':>6} {'European put':>14} {'American put':>14} {'premium':>10} {'premium %':>10}")
for K, prem in zip(K_grid, premiums):
    eu = binomial_price(S, K, T, r, sigma, OptionType.PUT, ExerciseStyle.EUROPEAN, n_steps=500)
    am = eu + prem
    print(f"{K:>6.1f} {K/S:>6.2f} {eu:>14.4f} {am:>14.4f} {prem:>10.4f} {100*prem/eu if eu > 1e-9 else float('nan'):>9.1f}%")

# --- premium vs. volatility, at a fixed deep-ITM put ---
K_itm = 120.0
vol_grid = np.linspace(0.05, 0.6, 20)
premium_vs_vol = []
for v in vol_grid:
    eu = binomial_price(S, K_itm, T, r, v, OptionType.PUT, ExerciseStyle.EUROPEAN, n_steps=500)
    am = binomial_price(S, K_itm, T, r, v, OptionType.PUT, ExerciseStyle.AMERICAN, n_steps=500)
    premium_vs_vol.append(am - eu)
premium_vs_vol = np.array(premium_vs_vol)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("CRR binomial tree: convergence and the American early-exercise premium", fontweight="bold")

ax = axes[0]
ax.loglog(steps_grid, conv_errors, "o-", color="steelblue", label="|binomial - BS|")
ref = conv_errors[0] * (steps_grid[0] / steps_grid)
ax.loglog(steps_grid, ref, "--", color="darkorange", label=r"$N^{-1}$ reference")
ax.set_xlabel("tree steps N")
ax.set_ylabel("absolute error vs. BS")
ax.set_title("European call: binomial -> Black-Scholes")
ax.legend()

ax = axes[1]
ax.plot(K_grid / S, premiums, "o-", color="firebrick", label="American - European put")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.axvline(1.0, color="grey", lw=0.8, ls=":", label="ATM")
ax.set_xlabel("moneyness K/S")
ax.set_ylabel("early-exercise premium ($)")
ax.set_title(f"Put early-exercise premium vs. moneyness (T={T}, sigma={sigma})")
ax.legend()

ax = axes[2]
ax.plot(vol_grid, premium_vs_vol, "o-", color="seagreen")
ax.set_xlabel("volatility sigma")
ax.set_ylabel("early-exercise premium ($)")
ax.set_title(f"Put premium vs. vol (K={K_itm}, deep ITM)")

plt.tight_layout()
plt.savefig("experiments/results/american_premium.png", dpi=150)
print("\nSaved experiments/results/american_premium.png")
