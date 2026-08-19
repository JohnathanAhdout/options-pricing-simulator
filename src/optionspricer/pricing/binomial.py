"""Cox-Ross-Rubinstein (CRR, 1979) binomial tree.

This is the only engine in the package that can price **American**
exercise: at every node, before discounting back, we compare the
continuation value against the value of exercising immediately and keep
the larger of the two. Black-Scholes and the plain Monte Carlo engine have
no way to represent "the holder might act early," because a European
payoff only ever looks at S_T.

The tree also serves as a second, structurally independent way to price
European options: as n_steps -> infinity, the CRR price converges to the
Black-Scholes price (see experiments/run_american_premium.py), which is a
useful sanity check that doesn't depend on either implementation being
"the formula." If a lattice and a closed form agree in the limit, that's
real evidence both are correct, not just internally consistent.

**Why this engine overrides `greeks()` instead of using the base class's
bump-and-reprice default:** a CRR tree's terminal payoff is
max(+/-(S * u^j * d^(N-j) - K), 0), which, for a *fixed* tree (fixed u, d,
N), is a piecewise **linear** function of the root spot S. Bumping S by a
cent rescales every terminal node linearly and essentially never flips
which nodes are in- vs out-of-the-money, so the price is locally exactly
linear in S almost everywhere. Central-difference delta is fine under this
(it's measuring the slope of a line), but gamma, the *curvature*, comes
back as floating-point noise around zero, because there genuinely is no
curvature except at the measure-zero set of spot prices where a terminal
node crosses the strike. This module sidesteps the problem entirely by
reading delta, gamma, and theta directly off the tree's own step-1 and
step-2 node values (a standard technique, e.g. Hull's *Options, Futures,
and Other Derivatives*), which costs nothing extra since those nodes are
already computed during backward induction, and isn't fooled by the
tree's piecewise-linear geometry.
"""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

import numpy as np  # every array (one per tree level) and every vectorized backward-induction step

from optionspricer.market import ExerciseStyle, Greeks, MarketData, OptionSpec, OptionType, PriceResult  # the shared value objects; ExerciseStyle is used only here and in structures
from optionspricer.pricing.base import GreekBumps, PricingEngine  # GreekBumps: reused for vega/rho, which this engine still finite-differences


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,  # defaults to European; pass ExerciseStyle.AMERICAN to turn on early exercise
    q: float = 0.0,
    n_steps: int = 200,
) -> float:  # returns just the scalar price; call price_and_greeks directly if you also want delta/gamma/theta
    """CRR binomial tree price. See `price_and_greeks` for the version that
    also returns delta/gamma/theta at no extra cost."""
    return price_and_greeks(S, K, T, r, sigma, option_type, exercise, q, n_steps)[0]  # discards delta/gamma/theta, keeps only the price (index 0)


def price_and_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    q: float = 0.0,
    n_steps: int = 200,
) -> tuple[float, float, float, float]:  # (price, delta, gamma, theta), all from one backward induction
    """One backward induction, vectorized across each level of the tree
    (one numpy array per step rather than one Python object per node),
    returning (price, delta, gamma, theta).

    Up/down factors u = e^{sigma*sqrt(dt)}, d = 1/u are the CRR choice that
    makes the tree *recombine* (an up-then-down move lands on the same price
    as down-then-up), which is what keeps the node count linear rather than
    exponential in n_steps. p is then whatever risk-neutral up-probability
    makes the tree's one-step expected return match (r - q); see
    BACKGROUND.md for the algebra.

    Delta/gamma/theta are read off the step-1 and step-2 nodes (see module
    docstring for why this beats bumping S directly): with S_u = S*u,
    S_d = S*d at step 1, and S_uu, S_ud (=S), S_dd at step 2,

        delta       = (V_u - V_d) / (S_u - S_d)
        delta_upper = (V_uu - V_ud) / (S_uu - S_ud)
        delta_lower = (V_ud - V_dd) / (S_ud - S_dd)
        gamma       = (delta_upper - delta_lower) / (0.5 * (S_uu - S_dd))
        theta       = (V_ud - V_root) / (2 * dt)   # per year; scaled to per day below

    theta compares the middle step-2 node (same spot S, but 2*dt later) to
    the root, which isolates the effect of time passing from the effect of
    the stock having moved.
    """
    if n_steps < 2:  # step-2 nodes don't exist below this, and gamma/theta below would index out of range
        raise ValueError("n_steps must be >= 2 to read Greeks off the tree")

    dt = T / n_steps  # length of one tree step, in years
    u = np.exp(sigma * np.sqrt(dt))  # CRR up-factor: chosen so log(u) = sigma*sqrt(dt), matching one step's return std dev
    d = 1.0 / u  # CRR down-factor = 1/u. This specific reciprocal choice is what makes the tree recombine
    disc = np.exp(-r * dt)  # one step's discount factor, applied once per level during backward induction
    # risk-neutral up-probability: solves E[S_{t+dt}] = S_t * e^{(r-q)dt} for p, i.e. p*u + (1-p)*d = e^{(r-q)dt}
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):  # a "probability" outside [0,1] would silently poison every price computed from it
        raise ValueError(
            f"risk-neutral probability {p:.4f} outside (0, 1); dt={dt:.4g} is too "
            f"coarse relative to sigma={sigma:.4g}, increase n_steps"
        )

    j = np.arange(n_steps + 1)  # node index 0..N at the terminal level: j up-moves, (N-j) down-moves
    S_terminal = S * u**j * d**(n_steps - j)  # every terminal node's stock price, vectorized over j at once
    if option_type == OptionType.CALL:  # terminal payoff differs by contract type; everything after this is identical for both
        values = np.maximum(S_terminal - K, 0.0)  # terminal payoff at every node, the tree's starting condition
    else:
        values = np.maximum(K - S_terminal, 0.0)

    step1_values = step2_values = None  # populated inside the loop below, once i reaches 1 and 2 respectively
    for i in range(n_steps - 1, -1, -1):  # walk backward from the terminal level to the root
        # one level of backward induction: each node's value is the discounted, probability-weighted
        # average of its two children (values[1:] = "up" children, values[:-1] = "down" children).
        # this single vectorized line replaces what would otherwise be a node-by-node Python loop
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if exercise == ExerciseStyle.AMERICAN:  # skipped entirely for European contracts; this is the ONLY early-exercise-specific code in the module
            j = np.arange(i + 1)  # node indices at level i (i+1 nodes: 0 down-moves..i down-moves)
            S_i = S * u**j * d**(i - j)  # stock price at every node of level i
            intrinsic = np.maximum(S_i - K, 0.0) if option_type == OptionType.CALL else np.maximum(K - S_i, 0.0)
            values = np.maximum(values, intrinsic)  # early-exercise check: hold the continuation value, or exercise now, whichever is worth more
        if i == 2:  # stash the step-2 level for gamma/theta, computed once the backward pass reaches it
            step2_values = values.copy()  # [V_dd, V_ud, V_uu], kept aside for gamma/theta below, not used in the recursion itself
        elif i == 1:  # stash the step-1 level for delta, one level closer to the root
            step1_values = values.copy()  # [V_d, V_u], kept aside for delta below

    price0 = float(values[0])  # after the full backward induction, `values` has exactly one entry left: the root

    # delta = slope of V between the two step-1 nodes. A direct finite difference, but one that uses
    # the tree's own natural discretization instead of an arbitrarily small (and here, misleading) bump
    S_u, S_d = S * u, S * d
    delta = (step1_values[1] - step1_values[0]) / (S_u - S_d)

    # gamma = the change in slope between the upper and lower halves of the step-2 nodes, i.e. a
    # second difference computed from three points that are actually on the tree, not off it
    S_uu, S_ud, S_dd = S * u * u, S, S * d * d
    delta_upper = (step2_values[2] - step2_values[1]) / (S_uu - S_ud)
    delta_lower = (step2_values[1] - step2_values[0]) / (S_ud - S_dd)
    gamma = (delta_upper - delta_lower) / (0.5 * (S_uu - S_dd))

    # theta: the middle step-2 node (S_ud = S, same spot as the root) sits exactly 2*dt later in time
    # with the stock unchanged, so comparing it to the root isolates pure time decay from any price move
    theta = (step2_values[1] - price0) / (2 * dt) / 365.0  # /365: annualized theta -> $ per calendar day

    return price0, float(delta), float(gamma), float(theta)  # cast off numpy scalar types, so callers get plain Python floats


class BinomialEngine(PricingEngine):  # the imperative shell: adapts price_and_greeks() to the common PricingEngine interface
    """European or American, lattice-priced. The only engine here that
    handles early exercise, and the only one with tree-native Greeks."""

    name = "binomial"  # the string every factory/experiment uses to select this engine

    def __init__(self, n_steps: int = 200):  # the tree's resolution; higher n_steps trades speed for convergence to the closed form
        self.n_steps = n_steps

    def price(self, option: OptionSpec, market: MarketData) -> PriceResult:  # note: shadows the module-level price() function above, same pattern as BlackScholesEngine
        p = price(
            market.spot, option.strike, option.maturity, market.rate, market.vol,
            option.option_type, option.exercise, market.dividend_yield, self.n_steps,
        )  # delegates to the module-level function; option.exercise (not a hardcoded European) is what makes American pricing possible here
        return PriceResult(price=p, engine=self.name)  # exact given n_steps, so stderr stays None like BlackScholesEngine's

    def greeks(self, option: OptionSpec, market: MarketData, bumps: GreekBumps | None = None) -> Greeks:  # overrides the default finite-difference greeks() for delta/gamma/theta, but NOT vega/rho
        b = bumps or GreekBumps()  # only used below, for the vega/rho bumps that still rely on finite differences
        _, delta, gamma, theta = price_and_greeks(
            market.spot, option.strike, option.maturity, market.rate, market.vol,
            option.option_type, option.exercise, market.dividend_yield, self.n_steps,
        )  # delta/gamma/theta come from ONE tree build, read off its nodes; the price itself is discarded here (leading _)

        vol_dn = max(market.vol - b.d_vol, 1e-8)  # same floor as PricingEngine.greeks(), so a near-zero-vol option never gets bumped negative
        vol_up = market.vol + b.d_vol
        p_up_v = price_and_greeks(market.spot, option.strike, option.maturity, market.rate, vol_up, option.option_type, option.exercise, market.dividend_yield, self.n_steps)[0]  # rebuilds a WHOLE second tree just to read off price0
        p_dn_v = price_and_greeks(market.spot, option.strike, option.maturity, market.rate, vol_dn, option.option_type, option.exercise, market.dividend_yield, self.n_steps)[0]  # and a third tree here, since vega has no tree-native shortcut
        vega = (p_up_v - p_dn_v) / (vol_up - vol_dn) / 100.0  # same central-difference pattern as PricingEngine's default, scaled to $ per vol point

        p_up_r = price_and_greeks(market.spot, option.strike, option.maturity, market.rate + b.d_rate, market.vol, option.option_type, option.exercise, market.dividend_yield, self.n_steps)[0]  # a fourth tree, rate bumped up
        p_dn_r = price_and_greeks(market.spot, option.strike, option.maturity, market.rate - b.d_rate, market.vol, option.option_type, option.exercise, market.dividend_yield, self.n_steps)[0]  # and a fifth, rate bumped down
        rho = (p_up_r - p_dn_r) / (2 * b.d_rate) / 100.0  # same central-difference pattern as delta in PricingEngine's default

        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)  # delta/gamma/theta tree-native; vega/rho finite-differenced, all bundled into one result
