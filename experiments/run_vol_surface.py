"""Pulls a live SPY options chain and builds the smile, term structure, and
full volatility surface -- the empirical answer to "is Black-Scholes'
constant-volatility assumption true?" If every strike and every maturity
implied the same sigma, every line in the left-hand plot would be flat.
They never are.

Needs a network connection and (for a full-looking smile) trading hours;
outside market hours yfinance still returns the last traded prices, which
is what `data.py`'s `_mid_price` falls back to.

Run: uv run python experiments/run_vol_surface.py [TICKER]
Output: experiments/results/vol_surface_<TICKER>.png
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optionspricer.data import fetch_call_quotes, fetch_spot
from optionspricer.implied_vol import JaeckelSolver
from optionspricer.surface import implied_vols, smile_at_maturity, surface_grid, term_structure

TICKER = sys.argv[1] if len(sys.argv) > 1 else "SPY"
R = 0.045  # approximate short-term risk-free rate; not fetched live, since that's a separate data source

print(f"Fetching {TICKER} spot and options chain...")
S0 = fetch_spot(TICKER)
quotes = fetch_call_quotes(TICKER, S0)
print(f"{TICKER} spot: {S0:.2f}, quotes in +/-25% moneyness band: {len(quotes)}")

solver = JaeckelSolver()
points = implied_vols(quotes, S0, R, solver)
print(f"Valid IV points after no-arbitrage/sanity filtering: {len(points)} / {len(quotes)}")

if not points:
    print("No valid IV points recovered (market likely closed with no recent trades) -- nothing to plot.")
    sys.exit(0)

maturities = sorted({p.maturity for p in points})
print(f"Maturities represented: {[f'{t:.2f}y' for t in maturities]}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(f"{TICKER} implied volatility, live chain (spot={S0:.2f})", fontweight="bold")

ax = axes[0]
for T in maturities[:6]:
    strikes, ivs = smile_at_maturity(points, T)
    if len(strikes) > 2:
        ax.plot(strikes, ivs * 100, "o-", ms=3, lw=1.2, label=f"T={T:.2f}y")
ax.axvline(S0, color="black", ls="--", lw=1, label=f"spot={S0:.0f}")
ax.set_xlabel("strike K")
ax.set_ylabel("implied vol (%)")
ax.set_title("Volatility smile/skew")
ax.legend(fontsize=7)

ax = axes[1]
term_T, term_iv = term_structure(points)
ax.plot(term_T, term_iv, "o-", color="purple")
ax.set_xlabel("maturity T (years)")
ax.set_ylabel("ATM-ish implied vol (%)")
ax.set_title("Term structure")

ax = axes[2]
if len(points) >= 4:
    K, T, IV = surface_grid(points)
    cf = ax.contourf(K, T, IV, levels=20, cmap="RdYlGn_r")
    ax.axvline(S0, color="white", ls="--", lw=1)
    fig.colorbar(cf, ax=ax, label="IV (%)")
    ax.set_xlabel("strike K")
    ax.set_ylabel("maturity T (years)")
    ax.set_title("Surface (interpolated)")
else:
    ax.text(0.5, 0.5, "not enough points\nfor a surface", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
out_path = f"experiments/results/vol_surface_{TICKER}.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved {out_path}")
