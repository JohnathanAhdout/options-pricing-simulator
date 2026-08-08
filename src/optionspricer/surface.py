"""Building a volatility smile/term-structure/surface out of a list of
option quotes.

Deliberately decoupled from where the quotes come from: everything here
takes a plain list of `Quote` and an `IVSolver`, so it works identically on
live SPY quotes (`data.py`) or on synthetic quotes generated in a test.
That decoupling is also *why* it's testable at all without a network call.
Feed `implied_vols` a batch of quotes priced off a made-up smile
function and it should recover that exact smile, which is what
`tests/test_surface.py` checks.

Why a smile/surface exists at all, in one line (full argument in
BACKGROUND.md): Black-Scholes assumes one constant sigma prices every
strike and maturity on a stock; real market prices don't agree with each
other under that assumption, so "the IV surface" is a picture of exactly
how, and by how much, Black-Scholes is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

from optionspricer.implied_vol.base import IVSolver
from optionspricer.market import OptionType


@dataclass(frozen=True, slots=True)
class Quote:
    strike: float
    maturity: float  # years to expiry
    mid_price: float
    option_type: OptionType


@dataclass(frozen=True, slots=True)
class IVPoint:
    strike: float
    maturity: float
    iv: float


def _no_arbitrage_lower_bound(quote: Quote, spot: float, r: float, q: float) -> float:
    """A European call must be worth at least its discounted intrinsic
    value, max(S*e^{-qT} - K*e^{-rT}, 0), otherwise buying the call,
    exercising, and selling the stock is free money. A quote priced at or
    below this is a stale or broken price, not a signal about volatility."""
    if quote.option_type != OptionType.CALL:
        raise NotImplementedError("no-arbitrage filtering here assumes call quotes; convert puts via parity first")
    return max(spot * np.exp(-q * quote.maturity) - quote.strike * np.exp(-r * quote.maturity), 0.0)


def implied_vols(
    quotes: list[Quote],
    spot: float,
    r: float,
    solver: IVSolver,
    q: float = 0.0,
    vol_bounds: tuple[float, float] = (0.01, 3.0),
) -> list[IVPoint]:
    """Invert every quote for its Black-Scholes implied vol, dropping
    anything that violates the no-arbitrage bound or resolves to an
    implausible vol (almost always a stale/bad quote, not a real price)."""
    points: list[IVPoint] = []
    for quote in quotes:
        if quote.mid_price <= _no_arbitrage_lower_bound(quote, spot, r, q):
            continue
        iv = solver.solve(quote.mid_price, spot, quote.strike, quote.maturity, r, quote.option_type, q)
        if vol_bounds[0] < iv < vol_bounds[1]:
            points.append(IVPoint(strike=quote.strike, maturity=quote.maturity, iv=iv))
    return points


def smile_at_maturity(points: list[IVPoint], maturity: float, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """(strikes, ivs), sorted by strike, for every point at the given
    maturity: one cross-section of the surface, holding T fixed."""
    matched = sorted((p for p in points if abs(p.maturity - maturity) < tol), key=lambda p: p.strike)
    return np.array([p.strike for p in matched]), np.array([p.iv for p in matched])


def term_structure(points: list[IVPoint]) -> tuple[np.ndarray, np.ndarray]:
    """(maturities, ivs), sorted by maturity: one IV per maturity, using
    whichever point in `points` is closest to the median strike at that
    maturity (a proxy for "the ATM point") when more than one is present."""
    by_maturity: dict[float, list[IVPoint]] = {}
    for p in points:
        by_maturity.setdefault(p.maturity, []).append(p)

    maturities, ivs = [], []
    for T, pts in sorted(by_maturity.items()):
        strikes = np.array([p.strike for p in pts])
        median_k = np.median(strikes)
        closest = min(pts, key=lambda p: abs(p.strike - median_k))
        maturities.append(T)
        ivs.append(closest.iv)
    return np.array(maturities), np.array(ivs)


def surface_grid(points: list[IVPoint], n_strikes: int = 60, n_maturities: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the scattered (K, T) -> IV cloud onto a regular grid via
    cubic interpolation, so it can be drawn as a surface/heatmap. Returns
    (K_grid, T_grid, IV_grid), each of shape (n_maturities, n_strikes)."""
    strikes = np.array([p.strike for p in points])
    maturities = np.array([p.maturity for p in points])
    ivs = np.array([p.iv for p in points])

    k_grid = np.linspace(strikes.min(), strikes.max(), n_strikes)
    t_grid = np.linspace(maturities.min(), maturities.max(), n_maturities)
    K, T = np.meshgrid(k_grid, t_grid)
    IV = griddata((strikes, maturities), ivs, (K, T), method="cubic")
    return K, T, IV
