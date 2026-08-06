"""Newton vs. Brent vs. Jaeckel, head to head: accuracy, iteration count,
and -- the interesting part -- what actually happens in the region where
Black-Scholes price stops depending meaningfully on volatility at all.

For every (option type, maturity, moneyness) cell in a grid, price an
option at a known sigma, hand only the resulting market price to each
solver, and check how close it gets back to the sigma that generated it.
Deep in-the-money and deep out-of-the-money contracts, especially near
expiry, have vega collapsing toward zero -- price is almost entirely
intrinsic value there, barely a function of sigma -- which makes *inverting*
for sigma an ill-conditioned problem no solver can fix. The interesting
result isn't just "who has smaller error," it's *how* each solver fails
when it fails: Newton diverges visibly to a nonsensical sigma (easy to
catch), while Brent (and Jaeckel, which falls back to Brent) can return a
plausible-looking but still-wrong sigma from within a nearly-flat bracket
(much harder to catch without an independent check on time value).

Run: uv run python experiments/run_iv_solver_benchmark.py
Output: experiments/results/iv_solver_benchmark.png
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from optionspricer.implied_vol.brent import solve as brent_solve
from optionspricer.implied_vol.jaeckel import solve as jaeckel_solve
from optionspricer.implied_vol.newton import solve as newton_solve
from optionspricer.market import OptionType
from optionspricer.pricing.black_scholes import d1, price as bs_price

S, r, sigma_true = 100.0, 0.05, 0.35
moneyness = np.linspace(0.7, 1.3, 25)  # K/S
maturities = [0.05, 0.1, 0.25, 1.0, 2.0]
option_types = [OptionType.CALL, OptionType.PUT]
SOLVERS = {"newton": newton_solve, "brent": brent_solve, "jaeckel": jaeckel_solve}


def newton_iters_to_converge(market_price, S, K, T, r, otype, tol=1e-8, max_iter=50) -> int:
    """Re-run Newton by hand, one step at a time, purely to count iterations
    (the library solver doesn't expose this, and bolting a counter onto the
    production solver for a one-off benchmark isn't worth the API noise)."""
    sigma = 0.2
    for i in range(max_iter):
        price = bs_price(S, K, T, r, sigma, otype)
        diff = price - market_price
        if abs(diff) < tol:
            return i
        vega = S * np.sqrt(T) * norm.pdf(d1(S, K, T, r, sigma))
        if abs(vega) < 1e-12:
            return -1
        sigma = max(sigma - diff / vega, 1e-8)
    return max_iter


errors = {name: [] for name in SOLVERS}
worst = {name: (0.0, None) for name in SOLVERS}
newton_iter_counts = []

for otype in option_types:
    for T in maturities:
        for m in moneyness:
            K = S * m
            market_price = bs_price(S, K, T, r, sigma_true, otype)
            if market_price < 1e-8:
                continue
            for name, solve_fn in SOLVERS.items():
                iv = solve_fn(market_price, S, K, T, r, otype)
                err = abs(iv - sigma_true)
                errors[name].append(err)
                if err > worst[name][0]:
                    worst[name] = (err, (otype.value, T, m, iv))
            newton_iter_counts.append(newton_iters_to_converge(market_price, S, K, T, r, otype))

print(f"{'solver':<10} {'median |err|':>14} {'p99 |err|':>12} {'max |err|':>12} {'n evaluated':>12}")
for name in SOLVERS:
    e = np.array(errors[name])
    print(f"{name:<10} {np.median(e):>14.2e} {np.percentile(e, 99):>12.2e} {e.max():>12.2e} {len(e):>12}")

print("\nWorst case per solver (option_type, T, K/S, recovered sigma; true sigma = {:.2f}):".format(sigma_true))
for name, (err, case) in worst.items():
    otype, T, m, iv = case
    print(f"  {name:<10} err={err:.4f}  {otype:<5} T={T:<5} K/S={m:.2f}  recovered sigma={iv:.4f}")

newton_iter_counts = np.array(newton_iter_counts)
converged = newton_iter_counts[newton_iter_counts >= 0]
print(f"\nNewton: mean iterations where it converged: {converged.mean():.2f}")
print(f"Newton: fraction that hit vega underflow (returned early/unconverged): {(newton_iter_counts < 0).mean():.1%}")

K_atm = S
market_price_atm = bs_price(S, K_atm, 1.0, r, sigma_true, OptionType.CALL)
n_reps = 2000
print(f"\n{'solver':<10} {'us/solve (ATM, 1y, well-conditioned)':>38}")
for name, fn in SOLVERS.items():
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn(market_price_atm, S, K_atm, 1.0, r, OptionType.CALL)
    print(f"{name:<10} {(time.perf_counter() - t0) / n_reps * 1e6:>38.2f}")

fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
fig.suptitle(f"IV solver |error| across moneyness x maturity, calls (top) and puts (bottom); true sigma={sigma_true}", fontweight="bold")

for row, otype in enumerate(option_types):
    grids = {name: np.full((len(maturities), len(moneyness)), np.nan) for name in SOLVERS}
    for i, T in enumerate(maturities):
        for j, m in enumerate(moneyness):
            K = S * m
            market_price = bs_price(S, K, T, r, sigma_true, otype)
            if market_price < 1e-8:
                continue
            for name, solve_fn in SOLVERS.items():
                grids[name][i, j] = abs(solve_fn(market_price, S, K, T, r, otype) - sigma_true)

    for col, name in enumerate(SOLVERS):
        ax = axes[row, col]
        err_log = np.log10(np.clip(grids[name], 1e-16, None))
        im = ax.imshow(err_log, aspect="auto", origin="lower", cmap="RdYlGn_r", vmin=-12, vmax=0,
                       extent=[moneyness.min(), moneyness.max(), 0, len(maturities)])
        ax.set_yticks(np.arange(len(maturities)) + 0.5)
        ax.set_yticklabels([f"T={t}" for t in maturities])
        ax.set_xlabel("moneyness K/S")
        ax.set_title(f"{name} ({otype.value})")
        fig.colorbar(im, ax=ax, label="log10|error|")

plt.tight_layout()
plt.savefig("experiments/results/iv_solver_benchmark.png", dpi=150)
print("\nSaved experiments/results/iv_solver_benchmark.png")
