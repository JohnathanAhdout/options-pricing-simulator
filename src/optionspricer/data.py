"""The one module in this package allowed to touch the network: pulls a
live options chain via `yfinance` and adapts it into the plain `Quote`
objects `surface.py` consumes. Nothing else in the package imports this
module, and `surface.py` doesn't import it either. That direction of
dependency (data -> surface, never surface -> data) is what keeps every
other test in this repo runnable offline.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

from datetime import datetime  # used to compute time-to-expiry from a listed expiry date string

import yfinance as yf  # the only external network client this package ever imports

from optionspricer.market import OptionType
from optionspricer.surface import Quote  # the plain value type this module's job is to produce


def _mid_price(row) -> float:  # row: one pandas Series from yfinance's options-chain DataFrame
    bid, ask = row["bid"], row["ask"]
    if bid > 0 and ask > 0:
        return (bid + ask) / 2  # the standard mid-price: prefer this whenever a real two-sided quote exists
    return row["lastPrice"] if row["lastPrice"] > 0 else 0.0  # fallback for stale/no-quote rows (e.g. outside market hours): last trade, or 0 if even that's missing


def fetch_spot(ticker: str) -> float:  # the underlying's current price, needed to compute moneyness bounds below
    hist = yf.Ticker(ticker).history(period="5d")  # 5 days of history: enough to find a recent close even across a weekend or holiday
    if hist.empty:
        raise RuntimeError(f"no price history for {ticker!r}, check the ticker or your connection")  # fail loudly rather than let a downstream NaN spot silently poison every quote
    return float(hist["Close"].iloc[-1])  # the most recent close in the returned history


def fetch_call_quotes(ticker: str, spot: float, moneyness_bounds: tuple[float, float] = (0.75, 1.25), max_expiries: int | None = None) -> list[Quote]:  # returns Quote objects directly, so callers never touch a raw yfinance DataFrame
    """All call quotes across every listed expiry, restricted to strikes
    within `moneyness_bounds` of spot (far OTM strikes have negligible
    premium, wide relative spreads, and correspondingly noisy IVs)."""
    handle = yf.Ticker(ticker)  # one client object, reused for both the list of expiries and each expiry's option chain below
    expiries = handle.options[:max_expiries] if max_expiries else handle.options  # cap the number of expiries fetched, or use every listed one if max_expiries is None
    today = datetime.today()  # computed once, outside the loop, so every expiry's time-to-maturity is measured from the same instant

    quotes: list[Quote] = []  # accumulator, filled one call quote at a time across every expiry
    for expiry in expiries:
        T = (datetime.strptime(expiry, "%Y-%m-%d") - today).days / 365.0  # calendar days to expiry, converted to years
        if T <= 0:
            continue  # an expiry that's already passed (or expires today) isn't a valid maturity for OptionSpec
        calls = handle.option_chain(expiry).calls  # one network call per expiry, returning a DataFrame of call quotes
        for _, row in calls.iterrows():  # iterrows() yields (index, row) pairs; the index itself is unused here
            K = float(row["strike"])
            if not (moneyness_bounds[0] * spot <= K <= moneyness_bounds[1] * spot):
                continue  # skip strikes outside the moneyness band, per the docstring's rationale
            price = _mid_price(row)
            if price <= 0:
                continue  # a zero price means no usable quote (no bid/ask, no last trade); not worth inverting
            quotes.append(Quote(strike=K, maturity=T, mid_price=price, option_type=OptionType.CALL))  # every quote from this module is a CALL; puts aren't fetched here
    return quotes
