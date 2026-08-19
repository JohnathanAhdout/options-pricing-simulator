"""Delta-hedging simulation and gamma/theta P&L attribution.

This is the mechanism behind "theta decay" and "gamma scalping," made
literal instead of folkloric. A trader who holds an options position and
delta-hedges it, buying or selling just enough stock every rebalance to
keep the *combined* position's delta at zero, is left exposed to exactly
one thing: whether the stock moves *more* or *less* than the volatility
they used to compute their hedge ratios. `simulate_delta_hedge` runs that
hedge mechanically, tick by tick, as a real self-financing trading strategy
(cash account, stock account, accrued interest, the works) and separately
evaluates the closed-form prediction for what the resulting P&L *should*
be, so the two can be checked against each other.

**The theory, briefly** (full derivation with every term explained in
BACKGROUND.md): let Pi(t) be the value of "the option position, plus
enough stock to zero out its delta, plus cash." Ito's lemma applied to the
option leg, combined with the Black-Scholes PDE evaluated at the
*hedging* volatility sigma_h (not the vol that actually realizes), gives

    dPi = 0.5 * Gamma(t) * S(t)^2 * (sigma_r(t)^2 - sigma_h^2) * dt

where sigma_r is *realized* volatility and Gamma(t) is the portfolio's
gamma (already signed: a short straddle's gamma is negative, so its P&L
formula automatically flips relative to a long straddle's, with no extra
bookkeeping needed). Read literally: a delta-hedged position earns money
in proportion to its gamma times the *gap* between what volatility
actually showed up and what volatility was priced in. This is the whole
economic content of "theta decay." Theta is the price (via the BS PDE)
of being long gamma, and whether that price was worth paying is a bet on
realized vs. hedged volatility, nothing more mystical than that.

`simulate_delta_hedge` builds the discrete analogue of the same portfolio
and reports both the mechanically-simulated `final_pnl` and the
closed-form `theoretical_pnl` (the sum, along the *same simulated path*, of
the formula above) so `experiments/run_gamma_theta_pnl.py` can check that
the discrete replication actually converges to the continuous-time theory
as hedging gets more frequent.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

from dataclasses import dataclass, replace  # replace: builds a time-shifted copy of the structure without mutating the original

import numpy as np  # the simulated path, and every array operation in the theoretical-P&L integral

from optionspricer.market import MarketData
from optionspricer.pricing.base import PricingEngine  # the interface supplying Greeks and prices used to hedge and mark
from optionspricer.simulation import gbm_path  # the one function that actually simulates the realized-vol price path
from optionspricer.structures.base import Structure, mark_to_market, portfolio_greeks  # the position being hedged, and the two functions used to value and differentiate it


@dataclass(frozen=True, slots=True)  # same immutability rationale as OptionSpec/MarketData in market.py
class HedgeResult:
    times: np.ndarray  # one timestamp per rebalance, length n_steps + 1
    spot_path: np.ndarray  # the simulated underlying price at every rebalance
    gamma_path: np.ndarray  # the portfolio's gamma at every rebalance, stashed for the theoretical-P&L integral
    pnl_path: np.ndarray  # running mark-to-market P&L at every rebalance
    final_pnl: float  # pnl_path's last entry, pulled out for convenience
    theoretical_pnl: float  # the closed-form prediction, computed along this SAME path


def simulate_delta_hedge(
    structure: Structure,
    S0: float,
    r: float,
    hedge_vol: float,  # the vol used to PRICE and HEDGE: what the trader believes, never fed into the simulated path itself
    realized_vol: float,  # the vol that actually drives the simulated path: the "true," physical volatility of this episode
    maturity: float,
    n_steps: int,
    engine: PricingEngine,
    rng: np.random.Generator,
    q: float = 0.0,
) -> HedgeResult:
    """Delta-hedge `structure` (an options-only position: it's built with
    no stock legs of its own, since the hedge computed here *is* the stock
    leg) from now to `maturity`, rebalancing every dt = maturity / n_steps.

    The underlying is simulated at `realized_vol` (the "true," physical
    volatility of the world), while every hedge ratio and every mark is
    computed at `hedge_vol` (what the trader believes/used to price the
    position). The gap between the two is the entire subject of this
    module. `engine` supplies the Greeks and prices used to compute hedge
    ratios and marks; it does not need to know about `realized_vol` at all,
    since realized volatility only ever drives the simulated path, never
    the hedger's own pricing.
    """
    if structure.stock_legs:  # this function computes its OWN stock leg (the hedge); a structure that already has one would double-count
        raise ValueError(
            "simulate_delta_hedge expects an options-only structure; the "
            "dynamic stock hedge computed here *is* the stock leg"
        )

    dt = maturity / n_steps  # time between rebalances, in years, e.g. maturity=0.5, n_steps=126 gives ~1 trading day
    path = gbm_path(S0, maturity, r, realized_vol, q, n_steps, 1, rng)[0]  # the ONE "true" price path this episode will realize, at realized_vol; [0] takes the single path out of the (1, n_steps+1) batch shape

    def hedge_market(S: float) -> MarketData:  # local closure: builds a MarketData at a given spot, ALWAYS at hedge_vol
        return MarketData(spot=S, rate=r, vol=hedge_vol, dividend_yield=q)  # every pricing/Greeks call uses hedge_vol, NEVER realized_vol

    entry_cost = structure.entry_cost  # the net premium paid (or received, if negative) to put the structure on
    cash = -entry_cost  # paying a debit (entry_cost > 0) spends cash; receiving a credit (entry_cost < 0) adds cash
    shares = 0.0  # stock held for hedging; starts flat, since no hedge trade has happened yet

    times = np.linspace(0.0, maturity, n_steps + 1)  # one timestamp per rebalance, evenly spaced from 0 to maturity
    gamma_path = np.zeros(n_steps)  # preallocated; filled in one entry per rebalance below
    pnl_path = np.zeros(n_steps + 1)  # pnl_path[0] is implicitly 0: entering at model value is a zero-NPV trade by construction

    for i in range(n_steps):
        elapsed = i * dt  # time since inception, at the START of this rebalance: how much each leg's maturity has shrunk by
        S_i = path[i]  # today's simulated spot, at the start of this rebalance

        g = portfolio_greeks(_shift(structure, elapsed), hedge_market(S_i), engine)  # portfolio delta/gamma AS OF right now, priced at hedge_vol
        gamma_path[i] = g.gamma  # stashed for the theoretical-P&L integral below; not used by the hedge mechanics itself
        target_shares = -g.delta  # hold enough stock to offset the option legs' delta exactly. This IS the definition of delta-hedged
        cash -= (target_shares - shares) * S_i  # buy/sell the difference at today's price; a self-financing trade, funded from cash
        shares = target_shares  # update the position AFTER trading, so the next iteration knows what's currently held

        cash *= np.exp(r * dt)  # cash sitting in the account earns (or costs, if negative) the risk-free rate between rebalances

        elapsed_next = (i + 1) * dt  # time since inception at the END of this rebalance, used for the mark below
        S_next = path[i + 1]  # the stock has now moved to its next realized-vol-driven price. This step is where realized vol actually shows up
        # mark_to_market already nets each leg against its entry_price, i.e. it returns
        # raw_option_value - entry_cost; add entry_cost back to get the raw value, since
        # `cash` separately (and only once) accounted for the entry_cost cash flow at t=0
        raw_option_value = mark_to_market(structure, hedge_market(S_next), elapsed_next, engine) + entry_cost
        pnl_path[i + 1] = cash + shares * S_next + raw_option_value  # total book value: cash + stock leg + option leg, marked at hedge_vol

    final_pnl = float(pnl_path[-1])  # the last entry of the running P&L path, pulled out for convenience
    S_grid = path[:-1]  # the spot price AT the start of each rebalance interval, matches where gamma_path[i] was measured
    # the discretized version of the module-docstring formula: sum over every rebalance of
    # 0.5*Gamma_i*S_i^2*(sigma_realized^2 - sigma_hedge^2)*dt, using the SAME simulated path and the
    # SAME gammas the mechanical hedge above actually traded on: a like-for-like theory-vs-simulation check
    theoretical_pnl = float(0.5 * np.sum(gamma_path * S_grid**2 * (realized_vol**2 - hedge_vol**2) * dt))

    return HedgeResult(
        times=times, spot_path=path, gamma_path=gamma_path,
        pnl_path=pnl_path, final_pnl=final_pnl, theoretical_pnl=theoretical_pnl,
    )


def _shift(structure: Structure, elapsed: float) -> Structure:  # called once per rebalance above, to age every leg's maturity forward
    """A copy of `structure` with every leg's remaining time to expiry
    reduced by `elapsed`, floored just above zero so `OptionSpec`'s
    positive-maturity invariant never breaks mid-simulation."""
    shifted_legs = tuple(
        replace(leg, option=replace(leg.option, maturity=max(leg.option.maturity - elapsed, 1e-8)))  # nested replace: a new OptionSpec inside a new OptionLeg, original structure never mutated
        for leg in structure.option_legs
    )
    return replace(structure, option_legs=shifted_legs)  # a new Structure with the aged legs; stock_legs (there are none here) pass through unchanged
