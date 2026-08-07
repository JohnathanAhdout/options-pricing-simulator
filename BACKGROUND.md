# Background

Full derivations for everything the code in `src/` implements: where each formula comes
from, why every term in it is there, and what breaks if you drop one. The main
[README](README.md) tells you *what* was built and *what the experiments found*; this
file is the *why*, worked out from first principles, so nothing in the code is a formula
copied without understanding it.

## Table of contents
- [Black-Scholes: the closed form](#black-scholes-the-closed-form)
- [The Greeks, dissected](#the-greeks-dissected)
- [Monte Carlo pricing](#monte-carlo-pricing)
- [The binomial tree](#the-binomial-tree)
- [Implied volatility: existence, uniqueness, and three ways to find it](#implied-volatility-existence-uniqueness-and-three-ways-to-find-it)
- [The volatility smile and surface](#the-volatility-smile-and-surface)
- [Multi-leg structures](#multi-leg-structures)
- [Delta-hedging and the gamma/theta P&L identity](#delta-hedging-and-the-gammatheta-pl-identity)
- [Hidden Markov Models for regime detection](#hidden-markov-models-for-regime-detection)

---

## Black-Scholes: the closed form

A stock price is modeled as geometric Brownian motion (GBM):

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

$S_t$ is the price, $\mu$ the real-world drift, $\sigma$ the volatility (held constant
here — the assumption the rest of this document spends its time poking holes in), and
$W_t$ a standard Brownian motion: continuous, independent increments, $W_t - W_s \sim
N(0, t-s)$. $dW_t$ is where the randomness comes from. Over an instant $dt$, the stock
moves by a deterministic drift $\mu S_t\, dt$ plus a random shock $\sigma S_t\, dW_t$.

Black, Scholes, and Merton's key insight, in 1973, was that you can build a portfolio out
of the option and $\Delta$ shares of stock whose value, over an infinitesimal instant, has
no random term at all: the option's sensitivity to $S$ exactly cancels the stock's own
randomness. A portfolio with zero risk has to earn exactly the risk-free rate $r$,
otherwise there's a riskless arbitrage — borrow at $r$, buy the portfolio, pocket the
difference. Working through that no-arbitrage condition, via Itô's lemma applied to
$V(S,t)$, the unknown option value, produces the **Black-Scholes PDE**:

$$\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = rV$$

Notice that $\mu$, the stock's real-world drift, doesn't appear anywhere in this equation.
That's the famous, counterintuitive part: an option's price doesn't depend on how bullish
or bearish the world actually is, only on $\sigma$, $r$, and the contract terms.
Equivalently, under the PDE above the stock behaves as if its drift were $r$, not $\mu$.
This fictitious world is the **risk-neutral measure**, and it's exactly what
`simulation.gbm_terminal` and `gbm_path` simulate: drift $r - q$, never $\mu$. Solving the
PDE with a European call's boundary condition $V(S,T) = \max(S-K, 0)$ gives the closed
form implemented in `pricing/black_scholes.py`:

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S/K) + \left(r - q + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}} \qquad d_2 = d_1 - \sigma\sqrt{T}$$

where $N(\cdot)$ is the standard normal CDF and $q$ is a continuous dividend yield (set
$q=0$ to recover the textbook no-dividend formula).

So what's actually inside $d_1$? $\ln(S/K)$ is log-moneyness: zero exactly at the money,
positive if $S > K$. $\left(r - q + \frac{\sigma^2}{2}\right)T$ is the risk-neutral drift
of $\ln S_t$ over $[0,T]$, and the sign is worth staring at: it's $+\sigma^2/2$, not
$-\sigma^2/2$. That looks backwards next to the simulation formula further down, which
uses $-\sigma^2/2$. The reason is that $d_1$ isn't estimating $E[\ln S_T]$; it's a
specific quantity that falls out of solving the pricing integral, and its derivation —
integrating the lognormal density against the payoff — produces a $+\sigma^2/2$ term via a
change of variables. Both signs are correct in their own context. Conflating them is one
of the easiest sign-error bugs to make in a from-scratch Black-Scholes implementation,
which is why it's called out explicitly in `black_scholes.py`. Dividing by
$\sigma\sqrt{T}$, the standard deviation of $\ln S_T$, turns the whole numerator into a
$z$-score, which is what makes plugging it into a normal CDF meaningful at all.

The price formula itself splits into two readable pieces. $S e^{-qT} N(d_1)$ is the
present value of the stock you receive, conditional on finishing in the money,
probability-weighted. It's tempting to read $N(d_1)$ as $P(S_T > K)$, but that's actually
$N(d_2)$; $N(d_1)$ is a risk-neutral, stock-numéraire-weighted probability that falls out
of the same integral. $K e^{-rT} N(d_2)$ is the present value of the strike you pay on
exercise, weighted by $N(d_2)$, which really is $P^Q(S_T > K)$ under the risk-neutral
measure — the option only gets exercised, and the strike only actually changes hands, in
that event. The put formula, $P = K e^{-rT} N(-d_2) - S e^{-q T} N(-d_1)$, is the exact
mirror image, using $N(-x) = 1-N(x)$, and is algebraically identical to computing the call
and applying **put-call parity**:

$$C - P = S e^{-qT} - K e^{-rT}$$

which holds by a static replication argument independent of Black-Scholes entirely: a
long call plus a short put has the same payoff as a forward contract, so it has to have
the same price. `tests/test_black_scholes.py::test_put_call_parity` checks this identity
holds to machine precision regardless of $\sigma$, since it's true for any pricing model,
not just this one.

## The Greeks, dissected

The Greeks are the partial derivatives of $V$ with respect to each input. `analytic_greeks`
in `black_scholes.py` differentiates the closed form directly; every other engine gets them
via finite differences (see `pricing/base.py`) or, for the binomial tree, by reading them
off the lattice itself (see below).

**Delta**, $\Delta = \partial V/\partial S$: for a call, $\Delta = e^{-qT}N(d_1)$. This is
approximately the risk-neutral probability of finishing in the money. The exact
probability is $N(d_2)$, not $N(d_1)$, but the two are close, and $N(d_1)$ is what
actually falls out of differentiating the formula. It isn't obvious that the $N(d_1)$ and
$N(d_2)$ terms in $\partial C/\partial S$ collapse this cleanly, but they do, once you
expand $\partial N(d_1)/\partial S$ and $\partial N(d_2)/\partial S$ and use
$S\phi(d_1) = Ke^{-(r-q)T}\phi(d_2)$ — a non-obvious identity that follows from completing
the square in the $d_1$/$d_2$ definitions. Practically, delta tells you how many shares of
stock replicate one option, which is exactly what `hedging.py`'s
`target_shares = -g.delta` uses it for.

**Gamma**, $\Gamma = \partial^2 V/\partial S^2 = \partial \Delta/\partial S$:

$$\Gamma = \frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}$$

where $\phi$ is the standard normal density, not the CDF. Gamma is identical for a call
and a put at the same strike and maturity, a direct consequence of put-call parity: $C -
P$ is linear in $S$ (it equals $Se^{-qT} - Ke^{-rT}$ exactly), so its second derivative is
exactly zero, forcing $\Gamma_{\text{call}} = \Gamma_{\text{put}}$.
`tests/test_black_scholes.py::test_gamma_positive_and_equal_for_call_and_put` checks this
directly. Gamma peaks at the money and collapses toward zero both deep in and deep out of
the money. That collapse is the whole reason the binomial tree needs a tree-native Greeks
method (see below), and the whole reason Newton's method for implied vol can blow up (see
the IV section).

**Theta**, $\Theta = \partial V/\partial t$ (with respect to calendar time passing, i.e.
$-\partial V/\partial T$): for a call,

$$\Theta = -\frac{S e^{-qT}\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1)$$

Three terms, three separate effects:
1. **"Gamma rent,"** $-\frac{Se^{-qT}\phi(d_1)\sigma}{2\sqrt{T}}$: always negative, and the
   direct cost of owning convexity. It's the same expression as gamma itself, up to a
   scale factor. A long-gamma position bleeds this every day it doesn't realize a big
   enough move — the mechanism made precise in the
   [gamma/theta P&L section](#delta-hedging-and-the-gammatheta-pl-identity) below.
2. **Financing the strike,** $-rKe^{-rT}N(d_2)$: as time passes, the present value of the
   strike you'd eventually pay rises (the discount factor $e^{-rT}$ approaches 1), which
   makes the call slightly less valuable. Another drag on theta.
3. **Missed dividends,** $+qSe^{-qT}N(d_1)$: only nonzero with $q>0$. The call holder
   isn't collecting the dividends the stock is paying out, and as time passes there's less
   of that opportunity cost left to miss — a small positive contribution to theta.

The put's theta is the same first term, with the second and third terms sign-flipped: the
put holder receives $K$ on exercise, so a rising discount factor helps rather than hurts,
and there's no missed-dividend cost when short the implicit stock exposure.

**Vega**, $\nu = \partial V/\partial \sigma$, scaled to \$-per-1-vol-point:

$$\nu = \frac{Se^{-qT}\sqrt{T}\,\phi(d_1)}{100}$$

Identical for calls and puts, same parity argument as gamma. Always positive: more
uncertainty about where $S_T$ ends up can only help an option holder, since a payoff of
$\max(\cdot, 0)$ only benefits from extra spread in the outcome, never loses from it.

**Rho**, $\rho = \partial V/\partial r$, scaled to \$-per-1%-rate-move: $\rho_{\text{call}}
= \frac{KTe^{-rT}N(d_2)}{100}$, positive because a higher rate raises the forward price
$Se^{(r-q)T}$, making calls (which profit from $S$ being high) more valuable; the put's
rho is the negative mirror image.

## Monte Carlo pricing

Simulate the risk-neutral terminal distribution directly and average discounted payoffs.
`simulation.gbm_terminal` draws

$$S_T = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\,Z\right], \qquad Z \sim N(0,1)$$

Why $-\sigma^2/2$ here but $+\sigma^2/2$ in $d_1$? Because this formula answers a
different question: what is $S_T$, given that $\ln S_T$ is normally distributed with a
particular mean and variance under the risk-neutral measure? Itô's lemma applied to
$\ln S_t$ under GBM gives $d(\ln S_t) = (r - q - \sigma^2/2)\,dt + \sigma\,dW_t$. The
$-\sigma^2/2$ term is the **Itô correction**: because $\ln$ is concave, $E[\ln S_T] \neq
\ln E[S_T]$, and the correction is exactly what's needed so that $E[S_T] = S_0
e^{(r-q)T}$ comes out right despite the concavity — Jensen's inequality biting in exactly
this way. `tests/test_simulation.py::test_gbm_terminal_mean_matches_risk_neutral_drift`
checks $E[S_T]$ against $S_0 e^{(r-q)T}$ directly.

The Monte Carlo price is a sample mean of $M$ i.i.d. discounted payoffs, and two classical
results govern how it behaves. The **Law of Large Numbers** says it converges to the true
expectation, the Black-Scholes price, as $M\to\infty$. The **Central Limit Theorem** says
the standard error of that sample mean shrinks as $\sigma_{\text{payoff}}/\sqrt{M}$: a
$1/\sqrt{M}$ rate, meaning quadrupling the path count only halves the error.
`experiments/run_mc_convergence.py` checks this rate empirically — see the README for the
fitted exponent.

`MonteCarloEngine` reseeds its RNG from the same integer seed on every call to `price()`.
That makes `price()` a pure, reproducible function of its arguments, and as a free side
effect it means the five bumped `price()` calls inside the base class's finite-difference
`greeks()` all share the same underlying $Z$ draws. That turns what would otherwise be the
difference of two independent, noisy estimates into the difference of two highly
correlated ones, which is dramatically less noisy. It's the standard variance-reduction
technique of **common random numbers**, and here it falls out for free from choosing to
reseed deterministically, rather than being something bolted on separately.

## The binomial tree

The Cox-Ross-Rubinstein (CRR, 1979) construction replaces continuous-time GBM with a
discrete lattice: over each small step $dt$, the stock either moves up by a factor $u$ or
down by a factor $d$.

CRR's specific choice is $u = e^{\sigma\sqrt{dt}}$, $d = 1/u$. The $\sigma\sqrt{dt}$ in the
exponent matches one step's log-return standard deviation under GBM, so as $dt \to 0$
(more steps), the tree's discrete distribution converges to the continuous lognormal one.
The choice $d = 1/u$ specifically, rather than some other down-factor, is what makes the
tree **recombine**: an up-move followed by a down-move lands on exactly
$S \cdot u \cdot d = S \cdot u/u = S$, the same node a down-then-up move reaches. Without
that, the number of distinct nodes would grow as $2^N$ instead of $N+1$ at level $N$.
Recombination is the entire reason a binomial tree is computationally tractable at all.

Exactly as in continuous time, pricing under the tree uses a risk-neutral probability $p$,
chosen so that the tree's one-step expected return matches the risk-neutral drift:

$$p \cdot u + (1-p) \cdot d = e^{(r-q)\,dt} \implies p = \frac{e^{(r-q)dt} - d}{u - d}$$

If $dt$ is too large relative to $\sigma$, this can fall outside $(0,1)$. That's a genuine
numerical failure mode, and `binomial.py` checks for it explicitly and raises, since a
"probability" outside $[0,1]$ would make every downstream price meaningless.

Terminal payoffs are computed at every leaf node, and each level is built from the one
after it: $V_i = e^{-r\,dt}\left[p\,V_{i+1, \text{up}} + (1-p)\,V_{i+1,\text{down}}\right]$,
the discounted, probability-weighted average of the two possible next values. For American
exercise, at every node the code additionally takes
$\max(\text{continuation value}, \text{intrinsic value})$: the holder gets to choose, at
every point in the tree, between holding and exercising now, and a rational holder always
takes whichever is worth more. This single `np.maximum` is the entire mechanism behind
early exercise — see the README's American-premium experiment for what it's worth in
dollars.

As $N \to \infty$, more and smaller steps, the CRR price converges to the Black-Scholes
price for European exercise. `experiments/run_american_premium.py` checks this directly,
and it's meaningful precisely because the tree and the closed form are structurally
different derivations — a discrete lattice versus solving a PDE — that happen to agree in
the limit. That's much stronger evidence both are implemented correctly than either one
being merely self-consistent. The empirical convergence rate for a plain CRR tree is
$O(1/N)$, not the faster $O(1/N^2)$ some smoothed variants achieve, and it's often
oscillatory — see the README for the measured convergence data.

Here's a subtlety that trips up a lot of from-scratch binomial implementations. For fixed
$u$, $d$, $N$, the tree's price as a function of the root spot $S$ is
$\sum_j w_j \max(\pm(S u^j d^{N-j} - K), 0)$ for fixed weights $w_j$: a sum of terms each
of which is **linear in $S$** wherever the $\max(\cdot,0)$ doesn't flip sign. A small bump
to $S$ almost never crosses one of those sign-flip boundaries, so locally the tree's price
is exactly linear in $S$, with zero curvature. That means a naive central-difference gamma
computes $(p_{\text{up}} - 2p_0 + p_{\text{down}})/h^2$, where the numerator is genuinely —
not approximately — zero up to floating-point rounding, and dividing that by a tiny $h^2$
amplifies the rounding noise into a number that looks like a real, wrong answer.
`binomial.py` sidesteps this by reading delta and gamma directly off the tree's own step-1
and step-2 node values instead of bumping $S$ at all. Those nodes are already computed
during the ordinary backward induction, so it costs nothing extra, and it isn't fooled by
the tree's piecewise-linear geometry.
`tests/test_binomial.py::test_tree_native_gamma_matches_black_scholes` is a regression
test for exactly this failure mode.

## Implied volatility: existence, uniqueness, and three ways to find it

Given a market price $C_{\text{mkt}}$, define $f(\sigma) = C_{\text{BS}}(\sigma) -
C_{\text{mkt}}$ and look for $\sigma$ such that $f(\sigma) = 0$. Before picking a method,
it's worth proving this is actually solvable.

Existence comes first. As $\sigma \to 0$, the Black-Scholes call price approaches its
discounted intrinsic value, $\max(Se^{-qT} - Ke^{-rT}, 0)$: at zero volatility the stock is
deterministic, so the option is worth exactly its (possibly zero) certain payoff. As
$\sigma \to \infty$, the call price approaches $Se^{-qT}$: infinite volatility means the
option is almost certainly exercised, so it converges to the value of just owning the
stock. Since $C_{\text{BS}}(\sigma)$ is continuous and sweeps from near zero to $Se^{-qT}$
as $\sigma$ ranges over $(0,\infty)$, the **Intermediate Value Theorem** guarantees a root
exists for any market price strictly between those two bounds — any price consistent with
no-arbitrage.

Uniqueness follows almost as easily. Vega, $\partial C/\partial \sigma$, is positive
everywhere (see the Greeks section — it's a product of manifestly positive terms), so
$C_{\text{BS}}(\sigma)$ is strictly increasing, and a strictly increasing continuous
function can cross any horizontal line at most once. Existence plus uniqueness together
mean there's exactly one implied volatility for any valid market price.
`test_implied_vol.py::test_root_is_unique_...` checks the monotonicity directly.

**Newton-Raphson** is the fastest of the three when it works:
$\sigma_{n+1} = \sigma_n - f(\sigma_n)/f'(\sigma_n)$, where $f'$ is vega. It converges
quadratically near the root — each step roughly squares the number of correct digits —
because Newton's method is really just repeatedly replacing $f$ with its local linear
(tangent-line) approximation and solving that exactly. How good an approximation a
straight line is to $f$ depends on $f$'s curvature, and that curvature error shrinks
quadratically as the step size shrinks. Its failure mode sits exactly where that
tangent-line approximation stops being a good idea: near-zero vega, deep ITM/OTM or near
expiry (see the gamma-collapse discussion above), where a small pricing error implies a
division by a tiny number and hence a huge, often self-worsening step. The new $\sigma$
can land somewhere with even smaller vega than before, compounding over iterations into a
genuinely enormous final value — see the README for a measured example with a recovered
$\sigma$ in the hundreds of thousands.

**Brent's method**, from 1973, takes the opposite approach: it maintains a bracket $[a,b]$
with $f(a)$ and $f(b)$ of opposite sign, guaranteeing the root is trapped inside. That
alone, via repeated bisection, would already guarantee convergence, just slowly —
linearly, one more correct bit per step. Brent's method speeds this up by first attempting
a faster step: the secant method (a straight line through the two most recent points), or,
once three points are available, inverse quadratic interpolation (fitting the unique
parabola through the last three $(\sigma, f(\sigma))$ points in the "$\sigma$ as a function
of $f$" direction, then jumping to where that parabola crosses zero). It falls back to a
safe bisection step whenever the fast step would land outside a safe region of the
bracket, or isn't making enough progress. The result is bisection's guarantee combined
with superlinear speed in practice — up to the golden-ratio rate, roughly 1.62 extra
correct digits per step, for the secant-only case. Because it never leaves its bracket,
Brent's method can't blow up the way Newton can. It isn't magic, though: in a region where
$f$ is genuinely almost flat over the entire bracket (see the discussion of deep-ITM,
near-expiry contracts below), Brent can converge to a point that satisfies the tolerance
$|f(\sigma)| < \text{tol}$ while still being numerically far from the "true" generating
$\sigma$, because many different $\sigma$ values produce prices within tolerance of each
other there. That's a real, measured failure mode too — see the README's IV solver
benchmark.

**Jaeckel's method** (in the style of his 2014 "Let's Be Rational" construction) stacks
two ideas on top of each other.

The first is to normalize the problem. Rewrite the price in terms of the forward
$F = Se^{(r-q)T}$, log-forward-moneyness $x = \ln(F/K)$, and total variance
$\theta = \sigma\sqrt{T}$. Every Black-Scholes price collapses to a function of just
$(x, \theta)$, two numbers instead of five, via the normalized price
$\beta = C/(e^{-rT}\sqrt{FK})$. That collapse makes it possible to invert the small- and
large-$|x|$ asymptotics of $\beta(\theta)$ in closed form and get a genuinely good
starting guess for $\theta$ — see `_initial_theta` in `jaeckel.py` for the three regimes
(ATM, OTM, ITM) and the quadratic each one solves — regardless of how far in or out of the
money the option is.

The second is to iterate with Halley's method instead of Newton's. Halley's update uses
the price's curvature in $\sigma$, the second derivative $f''$, on top of its slope
($f'$ = vega):

$$\sigma_{n+1} = \sigma_n - \frac{f(\sigma_n)/f'(\sigma_n)}{1 - \dfrac{f(\sigma_n)f''(\sigma_n)}{2f'(\sigma_n)^2}}$$

When $f$ is locally linear ($f''=0$), the denominator is exactly 1 and this reduces to
plain Newton. But whenever $f$ curves, the denominator bends the step size to compensate,
which is what gives cubic convergence — each step roughly triples the number of correct
digits — and, just as importantly, damps the step automatically near small vega instead of
blowing up the way plain Newton does.

If Halley still fails to converge within its iteration budget, `jaeckel.py` falls back to
`BrentSolver` as a guaranteed, if slower, backstop.

## The volatility smile and surface

Black-Scholes assumes one constant $\sigma$ prices every strike and maturity on a given
underlying. If that were literally true, inverting the market price of every listed option
on the same stock for its implied volatility would produce the exact same number every
time. It never does. Real implied volatilities vary systematically with strike (the
**smile/skew**) and maturity (the **term structure**), and `surface.py` /
`experiments/run_vol_surface.py` build exactly this picture from live data.

Why does the skew exist at all, at least for equities? Two overlapping explanations. The
first is **fat tails**: real return distributions have more extreme-move probability mass
than a lognormal predicts (higher kurtosis), so far-OTM options, which pay off precisely
in those tail events, are systematically underpriced by a constant-vol model, and the
market corrects for it by implying a higher $\sigma$ there. The second is skewness, or
crash-insurance demand: equity markets fall much harder than they rise, so low-strike puts
(crash insurance) see persistent buying pressure that constant-vol Black-Scholes doesn't
anticipate, inflating their implied vol relative to high-strike calls.

There's a second axis to this: plotting ATM implied vol against maturity. It can slope up
(near-term calm, more long-run uncertainty priced in), slope down (near-term stress
expected to resolve), or hump in the middle (a specific known event, like an earnings
date, sitting inside one maturity bucket but not others). None of these shapes are
possible under a model with one global $\sigma$. Their mere existence is direct market
evidence against the constant-volatility assumption, and it's the empirical motivation for
both the regime-detection work later in this document and for more sophisticated models —
local volatility, stochastic volatility — that this project doesn't implement but that
exist precisely to explain this picture.

One more piece of bookkeeping before quotes can be inverted at all: a call must be worth
at least its discounted intrinsic value, $\max(Se^{-qT} - Ke^{-rT}, 0)$, otherwise you
could buy the call, exercise it, sell the stock, and pocket a riskless profit. `surface.py`
drops any quote at or below this floor before attempting to invert it, since a quote that
low is stale or broken data, not a genuine signal about volatility.

## Multi-leg structures

A `Structure` is a tuple of `OptionLeg`/`StockLeg` objects, each with a signed `quantity`
(positive = long, negative = short) and the price it was entered at. Every named strategy
(straddle, spread, covered call, ...) is the *same* type; only which legs are in it
differs.

With that sign convention, `quantity * (value - entry_price)` gives the correct P&L for
both long and short legs without a branch. For a long leg ($\text{quantity}=+1$), it's the
ordinary "what you got minus what you paid." For a short leg ($\text{quantity}=-1$), it
becomes $-(\text{value} - \text{entry\_price}) = \text{entry\_price} - \text{value}$: you
keep the credit you received and owe whatever the position is worth now, which is exactly
right.

Portfolio-level Greeks come almost for free. Because differentiation is linear, the Greeks
of a portfolio of legs are just the quantity-weighted sum of each leg's own Greeks, with no
separate formula needed. A short straddle (short one call plus short one put, same strike)
therefore has negative gamma and positive theta, mirror images of the long straddle's
positive gamma and negative theta. That's exactly the "sell gamma, collect theta" position
the hedging section below is built around, and it falls straight out of summing two
negated single-option Greeks — nothing about it is special-cased anywhere in the code.

## Delta-hedging and the gamma/theta P&L identity

This is the derivation behind `hedging.py` and the flagship experiments: the mechanism
that makes "theta decay" and "gamma scalping" precise statements instead of trading-desk
folklore.

Here's the setup. A trader holds a `Structure` — any combination of option legs; the sign
convention above means this covers long or short positions with no extra casework — and
continuously **delta-hedges** it: holds $-\Delta_{\text{portfolio}}$ shares of stock, so
the combined position (options plus hedge) has zero delta at every instant. Let

$$\Pi_t = V_t - \Delta_t S_t$$

be the value of that combined position: option value minus the hedge's cost basis. Cash
from trading the hedge and financing at $r$ is tracked separately and shown to cancel
below.

$V(S,t)$ is a twice-differentiable function of the stock price and time, and $S$ follows
GBM with the actual, realized volatility $\sigma_r$, not necessarily the $\sigma_h$ the
trader priced and hedges with. Applying Itô's lemma to the option leg gives:

$$dV = \frac{\partial V}{\partial t}dt + \frac{\partial V}{\partial S}dS + \frac{1}{2}\frac{\partial^2 V}{\partial S^2}(dS)^2 = \Theta\,dt + \Delta\,dS + \frac{1}{2}\Gamma\,\sigma_r^2 S^2\,dt$$

using the Itô multiplication rule $(dS)^2 = \sigma_r^2 S^2\,dt$, Brownian motion's
quadratic variation. This is the step where the realized volatility of the actual world
enters the equation, regardless of what volatility was used to compute $\Theta$ and
$\Gamma$ themselves.

Now subtract the hedge. The change in the hedged portfolio, excluding financing (cash
earning $r$ on the stock trade proceeds, handled separately and exactly offset below), is:

$$d\Pi = dV - \Delta\,dS = \Theta\,dt + \frac{1}{2}\Gamma\sigma_r^2 S^2\,dt$$

The $\Delta\,dS$ terms cancel exactly, which is the entire point of delta-hedging: it
removes first-order exposure to the stock's direction, leaving only the second-order
(convexity) and time-decay terms.

One more substitution closes the loop. $\Theta$ here was computed using the hedging
volatility $\sigma_h$, whatever vol the option was marked or sold at, which by definition
satisfies the Black-Scholes PDE at $\sigma_h$:

$$\Theta + rS\Delta + \frac{1}{2}\Gamma\sigma_h^2 S^2 = rV \implies \Theta = rV - rS\Delta - \frac{1}{2}\Gamma\sigma_h^2S^2$$

The $rV - rS\Delta$ terms are exactly the financing income and cost of holding
$V - \Delta S = \Pi$ in cash at the risk-free rate. They cancel against the interest
actually credited to the hedge's cash account in `hedging.py`'s `cash *= exp(r*dt)` step.
What's left, after that cancellation, is the entire economic content of delta-hedging P&L:

$$\boxed{d\Pi_{\text{net}} = \frac{1}{2}\Gamma\,S^2\left(\sigma_r^2 - \sigma_h^2\right)dt}$$

Read literally: a delta-hedged position's P&L, at every instant, is proportional to its
gamma times the squared-volatility gap between what actually happened and what was priced
in. A long-gamma position ($\Gamma>0$, bought options) profits when $\sigma_r > \sigma_h$
— the world was wilder than priced, so the big moves the position was long gamma for
actually happened, more than compensating for the theta paid — and loses when
$\sigma_r < \sigma_h$, having paid for convexity that never showed up, bleeding theta
uncompensated. A short-gamma position is the exact mirror: it profits from a calmer world
than priced, collecting theta faster than it loses on the occasional big move, and loses
in a wilder one. Since a `Structure`'s own leg quantities already carry the correct sign
of $\Gamma$ (a short straddle sums to negative gamma automatically, per the previous
section), this single formula, with no extra casework for long versus short, is exactly
what `hedging.py`'s `theoretical_pnl` computes and what
`experiments/run_gamma_theta_pnl.py` validates against a real, mechanically simulated
self-financing hedge — see the README for how closely the two agree.

So what does this actually say about "theta decay"? Theta isn't an independent force. It's
the price, fixed by the no-arbitrage PDE, of being long gamma. Whether that price was
worth paying is entirely a bet on realized versus hedged volatility — there's no other
source of P&L in a delta-hedged options position under these assumptions (no jumps,
continuous rebalancing, no transaction costs). Discretizing the hedge, rebalancing at
finite intervals instead of continuously, introduces additional variance around this
formula: hedging error whose standard deviation shrinks as rebalancing gets more frequent.
Critically, it doesn't bias the mean away from the formula above. Both properties are
checked directly in `tests/test_hedging.py` and in the frequency sweep in
`experiments/run_gamma_theta_pnl.py`.

## Hidden Markov Models for regime detection

Returns are drawn from one of $K$ (here, 2) Gaussian "regimes," and the active regime
follows its own Markov chain that's never directly observed, only inferred from its
fingerprint on the returns. The parameters are $\pi_i = P(\text{state}_0 = i)$,
$A_{ij} = P(\text{state}_{t+1}=j \mid \text{state}_t = i)$ (the transition matrix), and
$(\mu_i, \sigma_i)$, the emission distribution of a return while in state $i$.

Define $\hat\alpha_t(i) = P(\text{state}_t = i \mid \text{returns}_{0:t})$, the filtered,
strictly causal probability of being in state $i$ having seen only data up to and
including $t$. It has a clean recursion:

$$c_t\,\hat\alpha_t(j) = b_j(x_t) \sum_i \hat\alpha_{t-1}(i)\,A_{ij}$$

where $b_j(x_t) = N(x_t; \mu_j, \sigma_j^2)$ is the emission density and $c_t$ is chosen so
$\hat\alpha_t$ sums to 1. This $c_t$ isn't just a normalization convenience. It equals
$P(x_t \mid x_{0:t-1})$, the one-step-ahead predictive probability of the observation
actually seen, which means $\prod_t c_t = P(x_{0:T})$, the total data likelihood. So
$\sum_t \log c_t$ recovers the model's log-likelihood for free, as a byproduct of the same
recursion that computes the filtered probabilities. `filtered_state_probs` in `regime.py`
is exactly $\hat\alpha$, and it's what a live strategy is allowed to condition on, since
nothing in its recursion ever looks past $t$.

A second recursion, run backward from $T-1$ down to $0$, computes $\hat\beta_t(i)$: a
measure, consistently rescaled using the same $c_t$'s from the forward pass, of how
probable all the future observations are given state $t = i$. Combined:

$$\gamma_t(i) = \hat\alpha_t(i)\,\hat\beta_t(i) = P(\text{state}_t = i \mid \text{ALL the data})$$

exactly, with no further renormalization needed — a consequence of how the forward and
backward scaling factors are constructed to cancel against each other. This is the best
possible hindsight estimate of the regime at $t$, valid only for offline analysis of a
completed series (`smoothed_state_probs`), never for a live decision, since it uses
information from after $t$.

Baum-Welch is the EM algorithm applied to this model. An E-step runs forward-backward
under the current parameter guess to get $\gamma_t(i)$ and the pairwise
$\xi_t(i,j) = P(\text{state}_t=i, \text{state}_{t+1}=j \mid \text{all data})$; an M-step
then re-estimates every parameter as a closed-form, $\gamma$/$\xi$-weighted MLE:

$$\pi_i \leftarrow \gamma_0(i) \qquad A_{ij} \leftarrow \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)} \qquad \mu_i \leftarrow \frac{\sum_t \gamma_t(i)\,x_t}{\sum_t \gamma_t(i)} \qquad \sigma_i^2 \leftarrow \frac{\sum_t \gamma_t(i)(x_t - \mu_i)^2}{\sum_t \gamma_t(i)}$$

Each M-step is a weighted version of the ordinary Gaussian MLE, where the weight on
observation $t$ for state $i$ is how much of the soft posterior mass at $t$ belongs to
state $i$. EM's core theoretical guarantee is that the observed-data log-likelihood never
decreases across iterations, and
`tests/test_regime.py::test_log_likelihood_is_monotonically_non_decreasing` checks this
directly against a from-scratch, independent, unscaled log-sum-exp forward pass.

That test is also how a real scaling bug got caught and fixed during development. An
earlier version of the forward/backward pass applied a per-timestep numerical
stabilization to the emission probabilities that correctly canceled out of $\hat\alpha$,
$\hat\beta$, $\gamma$, and $\xi$ themselves, but not out of the log-likelihood computed
from $\sum_t \log c_t$. The actual EM parameter updates were correct the entire time, but
the log-likelihood trace used to monitor convergence was silently wrong and
non-monotonic. It's the kind of bug that's easy to miss, because the parameters still
converge to sensible values — only checking the log-likelihood trace against an
independent implementation caught it.

Initialization matters more than it might seem. Starting every state with identical
parameters is a fixed point of EM: perfect symmetry between states never breaks on its
own. `regime.fit` seeds different states with visibly different standard deviations,
spread around the data's overall std, to give EM something asymmetric to sort out from the
first iteration.

Viterbi answers a different question from smoothing entirely: not "what's the most likely
state at each $t$ independently," but "what's the most likely single path through all
states, taken as a whole." It's solved by dynamic programming in log-space. `delta[t,j]`
tracks the log-probability of the best path ending in state $j$ at time $t$, `psi[t,j]`
remembers which state at $t-1$ that best path came from, and a final backward pass through
the `psi` backpointers recovers the whole path from just its endpoint.

One design choice is worth explaining directly: why does the strategy in
`experiments/run_regime_gamma_scalping.py` refit per episode on a rolling window, instead
of just using one global fit? A live trading decision can only use data available at the
time, and a fixed global fit implicitly uses the entire history — including the future,
relative to any earlier decision point — to set its parameters. That would be lookahead
bias. Refitting a fresh 2-state EM model on only the trailing window at each decision
point is slower and noisier (a handful of parameters fit from a few dozen to a couple
hundred observations is a genuinely small-sample problem — see the README for how
classification accuracy scales with window length), but it's the honest, causal version of
the question "could a real strategy have known this at the time."
