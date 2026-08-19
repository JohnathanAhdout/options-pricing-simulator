"""Newton vs. Brent vs. Jaeckel, head to head: accuracy, iteration count,
and, the interesting part, what actually happens in the region where
Black-Scholes price stops depending meaningfully on volatility at all.

For every (option type, maturity, moneyness) cell in a grid, price an
option at a known sigma, hand only the resulting market price to each
solver, and check how close it gets back to the sigma that generated it.
Deep in-the-money and deep out-of-the-money contracts, especially near
expiry, have vega collapsing toward zero. Price is almost entirely
intrinsic value there, barely a function of sigma, which makes *inverting*
for sigma an ill-conditioned problem no solver can fix. The interesting
result isn't just "who has smaller error," it's *how* each solver fails
when it fails: Newton diverges visibly to a nonsensical sigma (easy to
catch), while Brent (and Jaeckel, which falls back to Brent) can return a
plausible-looking but still-wrong sigma from within a nearly-flat bracket
(much harder to catch without an independent check on time value).

Run: uv run python experiments/run_iv_solver_benchmark.py
Output: experiments/results/iv_solver_benchmark.png
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

import time  # wall-clock timing for the speed benchmark near the bottom of this script

import matplotlib  # imported separately from pyplot so the backend can be set BEFORE pyplot is imported, below

matplotlib.use("Agg")  # non-interactive, file-only backend: this script never opens a window, just saves a PNG
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm  # norm.pdf, used in the hand-rolled Newton re-implementation below

from optionspricer.implied_vol.brent import solve as brent_solve  # the three module-level solve() functions, imported directly rather than via IVSolver classes
from optionspricer.implied_vol.jaeckel import solve as jaeckel_solve  # since this script benchmarks raw function-call overhead, not the class wrapper
from optionspricer.implied_vol.newton import solve as newton_solve
from optionspricer.market import OptionType
from optionspricer.pricing.black_scholes import d1, price as bs_price  # d1 needed for the hand-rolled Newton vega calculation below

S, r, sigma_true = 100.0, 0.05, 0.35  # fixed spot/rate/volatility every cell in the grid below reuses
moneyness = np.linspace(0.7, 1.3, 25)  # K/S: 25 moneyness levels from 30% OTM to 30% ITM
maturities = [0.05, 0.1, 0.25, 1.0, 2.0]  # five maturities, from about 18 days to 2 years
option_types = [OptionType.CALL, OptionType.PUT]
SOLVERS = {"newton": newton_solve, "brent": brent_solve, "jaeckel": jaeckel_solve}  # name -> function, so the loops below can iterate generically over all three


def newton_iters_to_converge(market_price, S, K, T, r, otype, tol=1e-8, max_iter=50) -> int:  # returns -1 on vega underflow, max_iter if the budget ran out, otherwise the iteration count
    """Re-run Newton by hand, one step at a time, purely to count iterations
    (the library solver doesn't expose this, and bolting a counter onto the
    production solver for a one-off benchmark isn't worth the API noise)."""
    sigma = 0.2  # same starting guess as NewtonSolver's default
    for i in range(max_iter):
        price = bs_price(S, K, T, r, sigma, otype)  # local var `price` shadows the module-level bs_price import only within this loop body
        diff = price - market_price
        if abs(diff) < tol:
            return i  # converged: report how many iterations it took
        vega = S * np.sqrt(T) * norm.pdf(d1(S, K, T, r, sigma))  # same vega formula as newton.py's solve(), reimplemented here just for the iteration count
        if abs(vega) < 1e-12:
            return -1  # sentinel: vega underflowed before convergence, distinguishes this from a genuine max_iter exhaustion
        sigma = max(sigma - diff / vega, 1e-8)
    return max_iter  # loop exhausted without converging or underflowing


errors = {name: [] for name in SOLVERS}  # name -> list of |recovered_sigma - sigma_true| across every grid cell
worst = {name: (0.0, None) for name in SOLVERS}  # name -> (largest error seen so far, the (otype, T, moneyness, recovered_sigma) that produced it)
newton_iter_counts = []  # one entry per grid cell, from newton_iters_to_converge above

for otype in option_types:
    for T in maturities:
        for m in moneyness:
            K = S * m  # strike implied by this cell's moneyness level
            market_price = bs_price(S, K, T, r, sigma_true, otype)  # the "market" price every solver below is handed, generated at the KNOWN true sigma
            if market_price < 1e-8:
                continue  # a price this close to zero has no real signal to invert; skip rather than let a solver chase noise
            for name, solve_fn in SOLVERS.items():
                iv = solve_fn(market_price, S, K, T, r, otype)  # each solver only ever sees market_price, never sigma_true directly
                err = abs(iv - sigma_true)
                errors[name].append(err)
                if err > worst[name][0]:
                    worst[name] = (err, (otype.value, T, m, iv))  # track the single worst case per solver, for the printed report below
            newton_iter_counts.append(newton_iters_to_converge(market_price, S, K, T, r, otype))  # same market_price, timed with the hand-rolled counter above

print(f"{'solver':<10} {'median |err|':>14} {'p99 |err|':>12} {'max |err|':>12} {'n evaluated':>12}")  # column headers, aligned to match the formatted rows below
for name in SOLVERS:
    e = np.array(errors[name])  # this solver's full error distribution across every grid cell, as one array
    print(f"{name:<10} {np.median(e):>14.2e} {np.percentile(e, 99):>12.2e} {e.max():>12.2e} {len(e):>12}")

print("\nWorst case per solver (option_type, T, K/S, recovered sigma; true sigma = {:.2f}):".format(sigma_true))
for name, (err, case) in worst.items():
    otype, T, m, iv = case  # unpack the worst-case tuple stashed during the grid loop above
    print(f"  {name:<10} err={err:.4f}  {otype:<5} T={T:<5} K/S={m:.2f}  recovered sigma={iv:.4f}")

newton_iter_counts = np.array(newton_iter_counts)
converged = newton_iter_counts[newton_iter_counts >= 0]  # boolean-mask out the -1 sentinels (vega underflow) before averaging
print(f"\nNewton: mean iterations where it converged: {converged.mean():.2f}")
print(f"Newton: fraction that hit vega underflow (returned early/unconverged): {(newton_iter_counts < 0).mean():.1%}")

K_atm = S  # at the money: the best-conditioned case, used for the pure speed comparison below (accuracy is already covered by the grid above)
market_price_atm = bs_price(S, K_atm, 1.0, r, sigma_true, OptionType.CALL)  # one fixed, easy market price every solver times itself against
n_reps = 2000  # repeated many times so the timing isn't dominated by one-off interpreter/measurement overhead
print(f"\n{'solver':<10} {'us/solve (ATM, 1y, well-conditioned)':>38}")
for name, fn in SOLVERS.items():
    t0 = time.perf_counter()  # wall-clock start, reset for each solver
    for _ in range(n_reps):
        fn(market_price_atm, S, K_atm, 1.0, r, OptionType.CALL)  # result discarded: only the elapsed time matters here
    print(f"{name:<10} {(time.perf_counter() - t0) / n_reps * 1e6:>38.2f}")  # total elapsed time, divided by n_reps, converted seconds -> microseconds

fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))  # 2 rows (call/put) x 3 columns (one per solver)
fig.suptitle(f"IV solver |error| across moneyness x maturity, calls (top) and puts (bottom); true sigma={sigma_true}", fontweight="bold")

for row, otype in enumerate(option_types):  # row 0 = calls, row 1 = puts
    grids = {name: np.full((len(maturities), len(moneyness)), np.nan) for name in SOLVERS}  # one 2D error grid per solver, NaN-filled so skipped cells show as blank rather than zero
    for i, T in enumerate(maturities):
        for j, m in enumerate(moneyness):
            K = S * m
            market_price = bs_price(S, K, T, r, sigma_true, otype)  # recomputed here (not reused from the loop above) since this pass needs the full grid layout, not a flat list
            if market_price < 1e-8:
                continue  # left as NaN in the grid, same skip rationale as the first loop above
            for name, solve_fn in SOLVERS.items():
                grids[name][i, j] = abs(solve_fn(market_price, S, K, T, r, otype) - sigma_true)  # this cell's recovered-vs-true error, for this solver

    for col, name in enumerate(SOLVERS):
        ax = axes[row, col]  # this (option type, solver) panel's axes
        err_log = np.log10(np.clip(grids[name], 1e-16, None))  # log10 for a readable color scale across many orders of magnitude; clipped away from exactly 0 to avoid log(0)
        im = ax.imshow(err_log, aspect="auto", origin="lower", cmap="RdYlGn_r", vmin=-12, vmax=0,
                       extent=[moneyness.min(), moneyness.max(), 0, len(maturities)])  # origin="lower": row 0 (shortest maturity) plots at the bottom
        ax.set_yticks(np.arange(len(maturities)) + 0.5)  # tick at the CENTER of each maturity row, not its edge
        ax.set_yticklabels([f"T={t}" for t in maturities])
        ax.set_xlabel("moneyness K/S")
        ax.set_title(f"{name} ({otype.value})")
        fig.colorbar(im, ax=ax, label="log10|error|")  # one colorbar per panel, since vmin/vmax are shared but each panel gets its own for readability

plt.tight_layout()
plt.savefig("experiments/results/iv_solver_benchmark.png", dpi=150)
print("\nSaved experiments/results/iv_solver_benchmark.png")
