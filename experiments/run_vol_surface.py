"""Pulls a live SPY options chain and builds the smile, term structure, and
full volatility surface: the empirical answer to "is Black-Scholes'
constant-volatility assumption true?" If every strike and every maturity
implied the same sigma, every line in the left-hand plot would be flat.
They never are.

Needs a network connection and (for a full-looking smile) trading hours;
outside market hours yfinance still returns the last traded prices, which
is what `data.py`'s `_mid_price` falls back to.

Run: uv run python experiments/run_vol_surface.py [TICKER]
Output: experiments/results/vol_surface_<TICKER>.png
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

import sys  # command-line argument access and sys.exit for the early-exit path below

import matplotlib  # imported separately from pyplot so the backend can be set BEFORE pyplot is imported, below

matplotlib.use("Agg")  # non-interactive, file-only backend: this script never opens a window, just saves a PNG
import matplotlib.pyplot as plt
import numpy as np  # used only for the vol_bounds default flowing through implied_vols; no direct numpy calls in this script

from optionspricer.data import fetch_call_quotes, fetch_spot  # the only two functions in the whole package that touch the network
from optionspricer.implied_vol import JaeckelSolver  # the production-default solver, per BACKGROUND.md's benchmark
from optionspricer.surface import implied_vols, smile_at_maturity, surface_grid, term_structure  # every downstream analysis step this script runs

TICKER = sys.argv[1] if len(sys.argv) > 1 else "SPY"  # optional command-line ticker override, defaulting to SPY
R = 0.045  # approximate short-term risk-free rate; not fetched live, since that's a separate data source

print(f"Fetching {TICKER} spot and options chain...")
S0 = fetch_spot(TICKER)  # one network call: the underlying's current price
quotes = fetch_call_quotes(TICKER, S0)  # a second network call (per expiry, internally): every call quote within the default moneyness band
print(f"{TICKER} spot: {S0:.2f}, quotes in +/-25% moneyness band: {len(quotes)}")

solver = JaeckelSolver()  # one shared solver instance, reused for every quote inverted below
points = implied_vols(quotes, S0, R, solver)  # inverts every quote, dropping no-arbitrage violations and implausible vols (see surface.py)
print(f"Valid IV points after no-arbitrage/sanity filtering: {len(points)} / {len(quotes)}")

if not points:
    print("No valid IV points recovered (market likely closed with no recent trades); nothing to plot.")
    sys.exit(0)  # exits cleanly rather than crashing on empty data further down (e.g. min() on an empty sequence)

maturities = sorted({p.maturity for p in points})  # a set comprehension dedupes, then sorted() gives increasing maturity order
print(f"Maturities represented: {[f'{t:.2f}y' for t in maturities]}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))  # three panels: smile/skew, term structure, interpolated surface
fig.suptitle(f"{TICKER} implied volatility, live chain (spot={S0:.2f})", fontweight="bold")

ax = axes[0]  # panel 1: implied vol vs. strike, one line per maturity, overlaid
for T in maturities[:6]:  # capped at the first 6 maturities, so the legend and plot don't get overcrowded on tickers with many expiries
    strikes, ivs = smile_at_maturity(points, T)
    if len(strikes) > 2:  # skip maturities with too few points to draw a meaningful line
        ax.plot(strikes, ivs * 100, "o-", ms=3, lw=1.2, label=f"T={T:.2f}y")  # *100: plotted as a percentage, not a raw decimal
ax.axvline(S0, color="black", ls="--", lw=1, label=f"spot={S0:.0f}")  # marks today's spot, so the ATM point on each smile is visible at a glance
ax.set_xlabel("strike K")
ax.set_ylabel("implied vol (%)")
ax.set_title("Volatility smile/skew")
ax.legend(fontsize=7)  # small font: there can be up to 6 maturity lines, each needing a legend entry

ax = axes[1]  # panel 2: one (roughly-ATM) implied vol per maturity, showing the term structure's shape
term_T, term_iv = term_structure(points)
ax.plot(term_T, term_iv, "o-", color="purple")
ax.set_xlabel("maturity T (years)")
ax.set_ylabel("ATM-ish implied vol (%)")
ax.set_title("Term structure")

ax = axes[2]  # panel 3: the full interpolated (strike, maturity) -> IV surface, or a placeholder message if there isn't enough data
if len(points) >= 4:  # griddata's cubic interpolation needs at least a handful of scattered points to produce a sane surface
    K, T, IV = surface_grid(points)  # note: T here shadows the maturities variable used above; local to this block only
    cf = ax.contourf(K, T, IV, levels=20, cmap="RdYlGn_r")  # filled contour plot: 20 color bands across the IV range
    ax.axvline(S0, color="white", ls="--", lw=1)  # marks today's spot on the surface too, white for contrast against the colored contours
    fig.colorbar(cf, ax=ax, label="IV (%)")
    ax.set_xlabel("strike K")
    ax.set_ylabel("maturity T (years)")
    ax.set_title("Surface (interpolated)")
else:
    ax.text(0.5, 0.5, "not enough points\nfor a surface", ha="center", va="center", transform=ax.transAxes)  # transform=ax.transAxes: coordinates are fractions of the panel, not data units

plt.tight_layout()
out_path = f"experiments/results/vol_surface_{TICKER}.png"  # filename includes the ticker, so different tickers don't overwrite each other's output
plt.savefig(out_path, dpi=150)
print(f"\nSaved {out_path}")
