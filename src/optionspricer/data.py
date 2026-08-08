"""The one module in this package allowed to touch the network: pulls a
live options chain via `yfinance` and adapts it into the plain `Quote`
objects `surface.py` consumes. Nothing else in the package imports this
module, and `surface.py` doesn't import it either. That direction of
dependency (data -> surface, never surface -> data) is what keeps every
other test in this repo runnable offline.
"""

from __future__ import annotations

from datetime import datetime

import yfinance as yf

from optionspricer.market import OptionType
from optionspricer.surface import Quote


def _mid_price(row) -> float:
    bid, ask = row["bid"], row["ask"]
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return row["lastPrice"] if row["lastPrice"] > 0 else 0.0


def fetch_spot(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="5d")
    if hist.empty:
        raise RuntimeError(f"no price history for {ticker!r}, check the ticker or your connection")
    return float(hist["Close"].iloc[-1])


def fetch_call_quotes(ticker: str, spot: float, moneyness_bounds: tuple[float, float] = (0.75, 1.25), max_expiries: int | None = None) -> list[Quote]:
    """All call quotes across every listed expiry, restricted to strikes
    within `moneyness_bounds` of spot (far OTM strikes have negligible
    premium, wide relative spreads, and correspondingly noisy IVs)."""
    handle = yf.Ticker(ticker)
    expiries = handle.options[:max_expiries] if max_expiries else handle.options
    today = datetime.today()

    quotes: list[Quote] = []
    for expiry in expiries:
        T = (datetime.strptime(expiry, "%Y-%m-%d") - today).days / 365.0
        if T <= 0:
            continue
        calls = handle.option_chain(expiry).calls
        for _, row in calls.iterrows():
            K = float(row["strike"])
            if not (moneyness_bounds[0] * spot <= K <= moneyness_bounds[1] * spot):
                continue
            price = _mid_price(row)
            if price <= 0:
                continue
            quotes.append(Quote(strike=K, maturity=T, mid_price=price, option_type=OptionType.CALL))
    return quotes
