# Background

Full derivations for everything the code in `src/` implements: where each formula comes
from, why every term in it is there, and what breaks if you drop one. The main
[README](README.md) tells you *what* was built and *what the experiments found*. This
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

Here's the plain version first. A stock price wiggles around randomly, and on average it
drifts up by the risk-free rate. Black-Scholes is the formula for what an option on that
stock is worth, and the surprising part is that the formula doesn't care whether you
personally think the stock is going up or down. It only cares how much the stock wiggles.
That's it. Everything below is just making that idea precise.

A stock price is assumed to follow geometric Brownian motion (GBM):

$$dS_t = \mu S_t \ dt + \sigma S_t \ dW_t$$

$S_t$ is the price, $\mu$ is the real-world drift (how fast the stock tends to grow on
average), and $\sigma$ is the volatility, held constant here. That constant-volatility
assumption is the thing this whole document eventually pokes holes in, so keep it in the
back of your mind. $W_t$ is a standard Brownian motion: think of it as the mathematical
idealization of a random walk that never stops jittering, with independent, normally
distributed steps, $W_t - W_s \sim N(0, t-s)$. $dW_t$ is where all the randomness comes
from. Over a tiny instant $dt$, the stock moves by a predictable drift, $\mu S_t\ dt$,
plus a random shock, $\sigma S_t\ dW_t$.

Black, Scholes, and Merton's insight, in 1973, was that you can build a portfolio out of the option and $\Delta$ shares of stock
whose value, over an infinitesimally small instant, has no randomness left in it at all.
The stock's own wiggle and the option's sensitivity to that wiggle cancel out exactly. A
portfolio with zero risk has to earn exactly the risk-free rate $r$. If it earned more,
you could borrow money for free, buy the portfolio, and pocket the difference: a genuine
money machine, which markets don't allow to exist for long. Working through that
no-arbitrage condition, using Itô's lemma applied to $V(S,t)$, the unknown option value,
produces the **Black-Scholes PDE**:

$$\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = rV$$

Look closely and notice what's missing: $\mu$, the stock's real-world drift, is nowhere
in this equation. That's the famous, counterintuitive part. An option's price doesn't
depend on how bullish or bearish the world actually is, only on $\sigma$, $r$, and the
contract terms. Equivalently, under this PDE the stock behaves as if its drift were $r$
instead of $\mu$. This fictitious, drift-adjusted world is called the **risk-neutral
measure**, and it's exactly what `simulation.gbm_terminal` and `gbm_path` simulate:
drift $r - q$, never $\mu$. Solving the PDE with a European call's boundary condition,
$V(S,T) = \max(S-K, 0)$, gives the closed form implemented in `pricing/black_scholes.py`:

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S/K) + \left(r - q + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}} \qquad d_2 = d_1 - \sigma\sqrt{T}$$

$N(\cdot)$ is the standard normal CDF, and $q$ is a continuous dividend yield (set $q=0$
to get the textbook no-dividend formula back).

So what's actually inside $d_1$? $\ln(S/K)$ is log-moneyness: zero exactly at the money,
positive if $S > K$. $\left(r - q + \frac{\sigma^2}{2}\right)T$ is the risk-neutral drift
of $\ln S_t$ over $[0,T]$. The sign is worth staring at: it's $+\sigma^2/2$, not
$-\sigma^2/2$. That looks backwards next to the simulation formula further down, which
uses $-\sigma^2/2$, and it's a genuinely easy place to introduce a bug if you're building
this from scratch. The reason for the flip is that $d_1$ isn't estimating $E[\ln S_T]$.
It's a specific quantity that falls out of solving the pricing integral, and its
derivation, integrating the lognormal density against the payoff, produces a
$+\sigma^2/2$ term through a change of variables. Both signs are correct in their own
context; they answer different questions. Conflating them is one of the easiest sign
errors to make in a from-scratch Black-Scholes implementation, which is why it gets
called out explicitly in `black_scholes.py`. Dividing by $\sigma\sqrt{T}$, the standard
deviation of $\ln S_T$, turns the whole numerator into a $z$-score, which is what makes
plugging it into a normal CDF meaningful in the first place.

The price formula itself splits into two readable pieces. $S e^{-qT} N(d_1)$ is the
present value of the stock you receive, conditional on finishing in the money,
probability-weighted. It's tempting to read $N(d_1)$ as $P(S_T > K)$, but that's actually
$N(d_2)$. $N(d_1)$ is a slightly different, risk-neutral, stock-numéraire-weighted
probability that falls out of the same integral. $K e^{-rT} N(d_2)$ is the present value
of the strike you pay on exercise, weighted by $N(d_2)$, which really is $P^Q(S_T > K)$
under the risk-neutral measure: the option only gets exercised, and the strike only
actually changes hands, in that event. The put formula,
$P = K e^{-rT} N(-d_2) - S e^{-q T} N(-d_1)$, is the exact mirror image, using
$N(-x) = 1-N(x)$, and it's algebraically identical to computing the call and applying
**put-call parity**:

$$C - P = S e^{-qT} - K e^{-rT}$$

This holds through a static replication argument that's completely independent of
Black-Scholes. A long call plus a short put has the same payoff as a forward contract, so
it has to have the same price, full stop. `tests/test_black_scholes.py::test_put_call_parity`
checks this identity holds to machine precision regardless of $\sigma$, since it's true
for any pricing model, not just this one.

## The Greeks, dissected

The Greeks are just the partial derivatives of $V$ with respect to each input: how much
does the option's price move if the stock moves, if volatility moves, if a day passes,
if rates move. `analytic_greeks` in `black_scholes.py` differentiates the closed form
directly. Every other engine gets them via finite differences (see `pricing/base.py`) or,
for the binomial tree, by reading them straight off the lattice (see below).

**Delta**, $\Delta = \partial V/\partial S$: for a call, $\Delta = e^{-qT}N(d_1)$. In
plain terms, delta answers "if the stock moves a dollar, how many cents does my option
move?" It's also approximately the risk-neutral probability of finishing in the money.
The exact probability is $N(d_2)$, not $N(d_1)$, but the two are close, and $N(d_1)$ is
what actually falls out of differentiating the formula. It isn't obvious that the
$N(d_1)$ and $N(d_2)$ terms in $\partial C/\partial S$ collapse this cleanly, but they
do, once you expand $\partial N(d_1)/\partial S$ and $\partial N(d_2)/\partial S$ and use
$S\phi(d_1) = Ke^{-(r-q)T}\phi(d_2)$, a non-obvious identity that follows from completing
the square in the $d_1$/$d_2$ definitions. Practically, delta tells you how many shares
of stock replicate one option, which is exactly what `hedging.py`'s
`target_shares = -g.delta` uses it for.

**Gamma**, $\Gamma = \partial^2 V/\partial S^2 = \partial \Delta/\partial S$: if delta is
"how much the option moves," gamma is "how much delta itself moves." It's the
sensitivity of the sensitivity, or how fast your hedge ratio goes stale as the stock
moves.

$$\Gamma = \frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}$$

Here $\phi$ is the standard normal density, not the CDF. Gamma is identical for a call
and a put at the same strike and maturity, a direct consequence of put-call parity: $C -
P$ is linear in $S$ (it equals $Se^{-qT} - Ke^{-rT}$ exactly), so its second derivative is
exactly zero, forcing $\Gamma_ {\text{call}} = \Gamma_ {\text{put}}$.
`tests/test_black_scholes.py::test_gamma_positive_and_equal_for_call_and_put` checks this
directly. Gamma peaks at the money and drops toward zero both deep in and deep out of the
money. That drop-off is the whole reason the binomial tree needs a tree-native Greeks
method (see below), and the whole reason Newton's method for implied vol can blow up (see
the IV section).

**Theta**, $\Theta = \partial V/\partial t$ (with respect to calendar time passing, i.e.
$-\partial V/\partial T$): theta is rent. It's what you lose (usually) every single day
just from time ticking forward, holding everything else fixed. For a call,

$$\Theta = -\frac{S e^{-qT}\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1)$$

Three terms, three separate effects:
1. **"Gamma rent,"** $-\frac{Se^{-qT}\phi(d_1)\sigma}{2\sqrt{T}}$: always negative, and the
   direct cost of owning convexity. It's the same expression as gamma itself, up to a
   scale factor. A long-gamma position bleeds this every day it doesn't realize a big
   enough move, a mechanism made precise in the
   [gamma/theta P&L section](#delta-hedging-and-the-gammatheta-pl-identity) below.
2. **Financing the strike,** $-rKe^{-rT}N(d_2)$: as time passes, the present value of the
   strike you'd eventually pay rises (the discount factor $e^{-rT}$ approaches 1), which
   makes the call slightly less valuable. Another small drag on theta.
3. **Missed dividends,** $+qSe^{-qT}N(d_1)$: only nonzero with $q>0$. The call holder
   isn't collecting the dividends the stock is paying out, and as time passes there's
   less of that opportunity cost left to miss, which is a small positive contribution to
   theta.

The put's theta is the same first term, with the second and third terms sign-flipped: the
put holder receives $K$ on exercise, so a rising discount factor helps rather than hurts,
and there's no missed-dividend cost when short the implicit stock exposure.

**Vega**, $\nu = \partial V/\partial \sigma$, scaled to \$-per-1-vol-point: vega answers
"if everyone got a little more nervous about how much the stock might move, how much more
is my option worth?"

$$\nu = \frac{Se^{-qT}\sqrt{T}\ \phi(d_1)}{100}$$

Vega is identical for calls and puts, by the same parity argument as gamma. It's always
positive: more uncertainty about where $S_T$ ends up can only help an option holder,
since a payoff of $\max(\cdot, 0)$ only benefits from extra spread in the outcome, and
never loses from it.

**Rho**, $\rho = \partial V/\partial r$, scaled to \$-per-1%-rate-move:
$\rho_ {\text{call}} = \frac{KTe^{-rT}N(d_2)}{100}$. This is positive because a higher
rate raises the forward price $Se^{(r-q)T}$, making calls, which profit from $S$ being
high, more valuable. The put's rho is the negative mirror image.

## Monte Carlo pricing

The idea here is almost cheating: instead of solving for the price with algebra, just
simulate a huge number of possible futures for the stock, see what the option is worth in
each one, average the payoffs, and discount back to today. It's brute force, but it works
for almost any payoff you can simulate, including ones with no closed form at all.
`simulation.gbm_terminal` draws

$$S_T = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\ Z\right], \qquad Z \sim N(0,1)$$

Why $-\sigma^2/2$ here but $+\sigma^2/2$ in $d_1$? Because this formula answers a
different question: what is $S_T$, given that $\ln S_T$ is normally distributed with a
particular mean and variance under the risk-neutral measure? Itô's lemma applied to
$\ln S_t$ under GBM gives $d(\ln S_t) = (r - q - \sigma^2/2)\ dt + \sigma\ dW_t$. The
$-\sigma^2/2$ term is the **Itô correction**. Because $\ln$ is concave, $E[\ln S_T] \neq
\ln E[S_T]$, and the correction is exactly what's needed so that $E[S_T] = S_0 e^{(r-q)T}$
comes out right despite the concavity. This is Jensen's inequality biting in exactly this
way. `tests/test_simulation.py::test_gbm_terminal_mean_matches_risk_neutral_drift` checks
$E[S_T]$ against $S_0 e^{(r-q)T}$ directly.

The Monte Carlo price is a sample mean of $M$ independent, identically distributed
discounted payoffs, and two classical results govern how well it behaves. The **Law of
Large Numbers** says it converges to the true expectation, the Black-Scholes price, as
$M\to\infty$. The **Central Limit Theorem** says the standard error of that sample mean
shrinks as $\sigma_ {\text{payoff}}/\sqrt{M}$, a $1/\sqrt{M}$ rate. In plain terms: to cut
your error in half, you need four times as many simulated paths, not two.
`experiments/run_mc_convergence.py` checks this rate empirically. See the README for the
fitted exponent.

`MonteCarloEngine` reseeds its RNG from the same integer seed on every call to `price()`.
That makes `price()` a pure, reproducible function of its arguments, and as a free side
effect it means the five bumped `price()` calls inside the base class's finite-difference
`greeks()` all share the same underlying $Z$ draws. That turns what would otherwise be
the difference of two independent, noisy estimates into the difference of two highly
correlated ones, which is dramatically less noisy. It's the standard variance-reduction
technique of **common random numbers**, and here it falls out for free from choosing to
reseed deterministically, rather than being bolted on as a separate step.

## The binomial tree

Here's the plain version: chop time into a bunch of tiny steps. At each step, let the
stock move up by a fixed factor or down by a fixed factor, nothing in between. Walk
backward from the last step to the first, working out what the option must be worth at
every point on that grid. That's the entire idea, and it's what the Cox-Ross-Rubinstein
(CRR, 1979) construction formalizes.

CRR's specific choice is $u = e^{\sigma\sqrt{dt}}$, $d = 1/u$. The $\sigma\sqrt{dt}$ in
the exponent matches one step's log-return standard deviation under GBM, so as
$dt \to 0$ (more, smaller steps), the tree's discrete distribution converges to the
continuous lognormal one. The choice $d = 1/u$ specifically, rather than some other
down-factor, is what makes the tree **recombine**: an up-move followed by a down-move
lands on exactly $S \cdot u \cdot d = S \cdot u/u = S$, the same node a down-then-up move
reaches. Without that, the number of distinct nodes would grow as $2^N$ instead of $N+1$
at level $N$. Recombination is the entire reason a binomial tree is computationally
practical at all.

Exactly as in continuous time, pricing under the tree uses a risk-neutral probability
$p$, chosen so that the tree's one-step expected return matches the risk-neutral drift:

$$p \cdot u + (1-p) \cdot d = e^{(r-q)\ dt} \implies p = \frac{e^{(r-q)dt} - d}{u - d}$$

If $dt$ is too large relative to $\sigma$, this can fall outside $(0,1)$. That's a real
numerical failure mode, and `binomial.py` checks for it explicitly and raises, since a
"probability" outside $[0,1]$ would make every downstream price meaningless.

Terminal payoffs are computed at every leaf node, and each level is built from the one
after it: $V_i = e^{-r\ dt}\left[p\ V_ {i+1, \text{up}} + (1-p)\ V_ {i+1,\text{down}}\right]$,
the discounted, probability-weighted average of the two possible next values. For
American exercise, at every node the code also takes
$\max(\text{continuation value}, \text{intrinsic value})$. The holder gets to choose, at
every point in the tree, between holding and exercising right now, and a rational holder
always takes whichever is worth more. That single `np.maximum` is the entire mechanism
behind early exercise. See the README's American-premium experiment for what it's worth
in dollars.

As $N \to \infty$, more and smaller steps, the CRR price converges to the Black-Scholes
price for European exercise. `experiments/run_american_premium.py` checks this directly,
and it's meaningful precisely because the tree and the closed form are structurally
different derivations, a discrete lattice versus solving a PDE, that happen to agree in
the limit. That agreement is much stronger evidence both are implemented correctly than
either one merely being consistent with itself. The empirical convergence rate for a
plain CRR tree is $O(1/N)$, not the faster $O(1/N^2)$ some smoothed variants achieve, and
it's often oscillatory. See the README for the measured convergence data.

Here's a subtlety that trips up a lot of from-scratch binomial implementations. For fixed
$u$, $d$, $N$, the tree's price as a function of the root spot $S$ is
$\sum_j w_j \max(\pm(S u^j d^{N-j} - K), 0)$ for fixed weights $w_j$: a sum of terms each
of which is **linear in $S$** wherever the $\max(\cdot,0)$ doesn't flip sign. A small bump
to $S$ almost never crosses one of those sign-flip boundaries, so locally the tree's
price is exactly linear in $S$, with zero curvature. That means a naive central-difference
gamma computes $(p_ {\text{up}} - 2p_0 + p_ {\text{down}})/h^2$, where the numerator is
genuinely, not approximately, zero up to floating-point rounding. Dividing that near-zero
number by a tiny $h^2$ amplifies the rounding noise into a number that looks like a real
answer but isn't. `binomial.py` sidesteps this by reading delta and gamma directly off the
tree's own step-1 and step-2 node values instead of bumping $S$ at all. Those nodes are
already computed during the ordinary backward induction, so it costs nothing extra, and
it isn't fooled by the tree's piecewise-linear geometry.
`tests/test_binomial.py::test_tree_native_gamma_matches_black_scholes` is a regression
test for exactly this failure mode.

## Implied volatility: existence, uniqueness, and three ways to find it

Pricing asks "given a volatility, what's the price?" Implied volatility asks the opposite
question: "given the price the market is actually charging, what volatility does that
price imply?" You're inverting the formula. Given a market price $C_ {\text{mkt}}$, define
$f(\sigma) = C_ {\text{BS}}(\sigma) - C_ {\text{mkt}}$ and look for $\sigma$ such that
$f(\sigma) = 0$. Before picking a method, it's worth proving this is even solvable.

Existence comes first. As $\sigma \to 0$, the Black-Scholes call price approaches its
discounted intrinsic value, $\max(Se^{-qT} - Ke^{-rT}, 0)$. At zero volatility the stock
is deterministic, so the option is worth exactly its (possibly zero) certain payoff. As
$\sigma \to \infty$, the call price approaches $Se^{-qT}$. Infinite volatility means the
option is almost certainly exercised, so it converges to the value of just owning the
stock. Since $C_ {\text{BS}}(\sigma)$ is continuous and sweeps from near zero to $Se^{-qT}$
as $\sigma$ ranges over $(0,\infty)$, the **Intermediate Value Theorem** guarantees a root
exists for any market price strictly between those two bounds, which is any price
consistent with no-arbitrage.

Uniqueness follows almost as easily. Vega, $\partial C/\partial \sigma$, is positive
everywhere (see the Greeks section; it's a product of manifestly positive terms), so
$C_ {\text{BS}}(\sigma)$ is strictly increasing, and a strictly increasing continuous
function can cross any horizontal line at most once. Existence plus uniqueness together
mean there's exactly one implied volatility for any valid market price.
`test_implied_vol.py::test_root_is_unique_...` checks the monotonicity directly.

**Newton-Raphson** is the fastest of the three when it works. Guess a sigma, check how
far off the resulting price is, use the slope (vega) to jump straight toward the answer,
and repeat: $\sigma_ {n+1} = \sigma_n - f(\sigma_n)/f'(\sigma_n)$, where $f'$ is vega. It
converges quadratically near the root: each step roughly squares the number of correct
digits. That's because Newton's method is really just repeatedly replacing $f$ with its
local, straight-line approximation and solving that exactly. How good a straight line is
as an approximation to $f$ depends on $f$'s curvature, and that curvature error shrinks
quadratically as the step size shrinks. Its failure mode sits exactly where that
straight-line approximation stops making sense: near-zero vega, deep in or out of the
money, or near expiry (see the gamma drop-off discussed above). There, a small pricing
error implies a division by a tiny number, and hence a huge, often self-worsening step.
The new $\sigma$ can land somewhere with even smaller vega than before, compounding over
iterations into a genuinely enormous final value. See the README for a measured example
with a recovered $\sigma$ in the hundreds of thousands.

**Brent's method**, from 1973, takes the opposite approach. Instead of guessing and
jumping, it keeps the true answer trapped: it maintains a bracket $[a,b]$ with $f(a)$ and
$f(b)$ of opposite sign, so the root can never escape. That alone, through repeated
bisection, would already guarantee convergence, just slowly (linearly, one more correct
bit per step). Brent's method speeds this up by first attempting a faster step: the
secant method (a straight line through the two most recent points), or, once three points
are available, inverse quadratic interpolation (fitting the unique parabola through the
last three $(\sigma, f(\sigma))$ points in the "$\sigma$ as a function of $f$" direction,
then jumping to where that parabola crosses zero). It falls back to a safe bisection step
whenever the fast step would land outside a safe region of the bracket, or isn't
converging quickly enough. The result is bisection's guarantee with superlinear speed in
practice, up to the golden-ratio rate, roughly 1.62 extra correct digits per step, for
the secant-only case. Because it never leaves its bracket, Brent's method can't blow up
the way Newton can. It isn't magic, though. In a region where $f$ is genuinely almost flat
over the entire bracket (see the discussion of deep-ITM, near-expiry contracts below),
Brent can converge to a point that satisfies the tolerance $|f(\sigma)| < \text{tol}$
while still being numerically far from the "true" generating $\sigma$, because many
different $\sigma$ values produce prices within tolerance of each other there. That's a
real, measured failure mode too. See the README's IV solver benchmark.

**Jaeckel's method**, in the style of his 2014 "Let's Be Rational" construction, stacks
two ideas on top of each other.

The first is to normalize the problem so the starting guess is already close to correct.
Rewrite the price in terms of the forward $F = Se^{(r-q)T}$, log-forward-moneyness
$x = \ln(F/K)$, and total variance $\theta = \sigma\sqrt{T}$. Every Black-Scholes price
collapses to a function of just $(x, \theta)$, two numbers instead of five, through the
normalized price $\beta = C/(e^{-rT}\sqrt{FK})$. That collapse makes it possible to
invert the small- and large-$|x|$ asymptotics of $\beta(\theta)$ in closed form and get a
genuinely good starting guess for $\theta$. See `_initial_theta` in `jaeckel.py` for the
three regimes (at, out, and in the money) and the quadratic each one solves, and it works
regardless of how far in or out of the money the option is.

The second is to iterate with Halley's method instead of Newton's. Halley's update uses
the price's curvature in $\sigma$, the second derivative $f''$, on top of its slope
($f'$ = vega):

$$\sigma_ {n+1} = \sigma_n - \frac{f(\sigma_n)/f'(\sigma_n)}{1 - \dfrac{f(\sigma_n)f''(\sigma_n)}{2f'(\sigma_n)^2}}$$

When $f$ is locally a straight line ($f''=0$), the denominator is exactly 1 and this
reduces to plain Newton. But whenever $f$ curves, the denominator bends the step size to
compensate, which is what gives cubic convergence (each step roughly triples the number
of correct digits) and, just as important, damps the step automatically near small vega
instead of blowing up the way plain Newton does.

If Halley still fails to converge within its iteration budget, `jaeckel.py` falls back to
`BrentSolver` as a guaranteed, if slower, backstop.

## The volatility smile and surface

Black-Scholes assumes one constant $\sigma$ prices every strike and maturity on a given
underlying. If that were literally true, inverting the market price of every listed
option on the same stock for its implied volatility would produce the exact same number
every time. It never does. Real implied volatilities vary systematically with strike (the
**smile/skew**) and maturity (the **term structure**), and `surface.py` and
`experiments/run_vol_surface.py` build exactly this picture from live data.

Why does the skew exist at all, at least for equities? Two overlapping explanations. The
first is **fat tails**: real return distributions have more extreme-move probability mass
than a lognormal predicts, so far-OTM options, which pay off precisely in those tail
events, are systematically underpriced by a constant-vol model. The market corrects for
that by implying a higher $\sigma$ there. The second is crash-insurance demand: equity
markets fall much harder than they rise, so low-strike puts, which act as crash
insurance, see persistent buying pressure that a constant-vol model doesn't anticipate,
inflating their implied vol relative to high-strike calls.

There's a second axis to this: plotting at-the-money implied vol against maturity. It can
slope up (near-term calm, more long-run uncertainty priced in), slope down (near-term
stress expected to resolve), or hump in the middle (a specific known event, like an
earnings date, sitting inside one maturity bucket but not others). None of these shapes
are possible under a model with one global $\sigma$. Their mere existence is direct
market evidence against the constant-volatility assumption, and it's the empirical
motivation for both the regime-detection work later in this document and for more
sophisticated models, like local volatility or stochastic volatility, that this project
doesn't implement but that exist precisely to explain this picture.

One more piece of bookkeeping before quotes can be inverted at all: a call must be worth
at least its discounted intrinsic value, $\max(Se^{-qT} - Ke^{-rT}, 0)$. Otherwise you
could buy the call, exercise it, sell the stock, and pocket a riskless profit. `surface.py`
drops any quote at or below this floor before attempting to invert it, since a quote that
low is stale or broken data, not a genuine signal about volatility.

## Multi-leg structures

A `Structure` is a tuple of `OptionLeg`/`StockLeg` objects, each with a signed `quantity`
(positive means long, negative means short) and the price it was entered at. Every named
strategy (straddle, spread, covered call, and so on) is the *same* type. Only which legs
are in it differs.

With that sign convention, `quantity * (value - entry_price)` gives the correct P&L for
both long and short legs without needing a branch. For a long leg
($\text{quantity}=+1$), it's the ordinary "what you got minus what you paid." For a short
leg ($\text{quantity}=-1$), it becomes `-(value - entry_price) = entry_price - value`:
you keep the credit you received and owe whatever the position is worth now, which is
exactly right.

Portfolio-level Greeks come almost for free. Because differentiation is linear, the
Greeks of a portfolio of legs are just the quantity-weighted sum of each leg's own
Greeks, with no separate formula needed. A short straddle (short one call plus short one
put, same strike) therefore has negative gamma and positive theta, mirror images of the
long straddle's positive gamma and negative theta. That's exactly the "sell gamma,
collect theta" position the hedging section below is built around, and it falls straight
out of summing two negated single-option Greeks. Nothing about it is special-cased
anywhere in the code.

## Delta-hedging and the gamma/theta P&L identity

Here's the plain-English version, before any of the algebra: if you delta-hedge an
option, you've turned it into a bet on whether the stock moves around *more* or *less*
than the volatility you priced it at. Move around more than expected, and you make money.
Move around less, and you lose. "Theta decay" is just the price tag on that bet, and this
section derives the exact dollar formula for it, rather than treating it as trading-desk
folklore.

Here's the setup. A trader holds a `Structure`, any combination of option legs (the sign
convention above means this covers long or short positions with no extra casework), and
continuously **delta-hedges** it: holds $-\Delta_ {\text{portfolio}}$ shares of stock, so
the combined position (options plus hedge) has zero delta at every instant. Let

$$\Pi_t = V_t - \Delta_t S_t$$

be the value of that combined position: option value minus the hedge's cost basis. Cash
from trading the hedge, and the interest it earns at rate $r$, is tracked separately and
shown to cancel below.

$V(S,t)$ is a twice-differentiable function of the stock price and time, and $S$ follows
GBM with the actual, realized volatility $\sigma_r$, not necessarily the $\sigma_h$ the
trader priced and hedges with. Applying Itô's lemma to the option leg gives:

$$dV = \frac{\partial V}{\partial t}dt + \frac{\partial V}{\partial S}dS + \frac{1}{2}\frac{\partial^2 V}{\partial S^2}(dS)^2 = \Theta\ dt + \Delta\ dS + \frac{1}{2}\Gamma\ \sigma_r^2 S^2\ dt$$

using the Itô multiplication rule $(dS)^2 = \sigma_r^2 S^2\ dt$, Brownian motion's
quadratic variation. This is the step where the realized volatility of the actual world
enters the equation, regardless of what volatility was used to compute $\Theta$ and
$\Gamma$ themselves.

Now subtract the hedge. The change in the hedged portfolio, excluding financing (cash
earning $r$ on the stock trade proceeds, handled separately and exactly offset below), is:

$$d\Pi = dV - \Delta\ dS = \Theta\ dt + \frac{1}{2}\Gamma\sigma_r^2 S^2\ dt$$

The $\Delta\ dS$ terms cancel exactly, which is the entire point of delta-hedging: it
removes first-order exposure to the stock's direction, leaving only the second-order
(convexity) and time-decay terms.

One more substitution closes the loop. $\Theta$ here was computed using the hedging
volatility $\sigma_h$, whatever vol the option was marked or sold at, which by definition
satisfies the Black-Scholes PDE at $\sigma_h$:

$$\Theta + rS\Delta + \frac{1}{2}\Gamma\sigma_h^2 S^2 = rV \implies \Theta = rV - rS\Delta - \frac{1}{2}\Gamma\sigma_h^2S^2$$

The $rV - rS\Delta$ terms are exactly the financing income and cost of holding
$V - \Delta S = \Pi$ in cash at the risk-free rate. They cancel against the interest
actually credited to the hedge's cash account in `hedging.py`'s `cash *= exp(r*dt)` step.
What's left, after that cancellation, is the entire economic content of delta-hedging
P&L:

$$\boxed{d\Pi_ {\text{net}} = \frac{1}{2}\Gamma\ S^2\left(\sigma_r^2 - \sigma_h^2\right)dt}$$

Read this literally and it says exactly what the plain-English version at the top of this
section promised: a delta-hedged position's P&L, at every instant, is proportional to its
gamma times the squared-volatility gap between what actually happened and what was priced
in. A long-gamma position ($\Gamma>0$, bought options) profits when $\sigma_r > \sigma_h$.
The world was wilder than priced, so the big moves the position was long gamma for
actually happened, more than paying for the theta spent. It loses when
$\sigma_r < \sigma_h$, having paid for convexity that never showed up, bleeding theta
with nothing to show for it. A short-gamma position is the exact mirror: it profits from
a calmer world than priced, collecting theta faster than it loses on the occasional big
move, and loses in a wilder one. Since a `Structure`'s own leg quantities already carry
the correct sign of $\Gamma$ (a short straddle sums to negative gamma automatically, per
the previous section), this single formula, with no extra casework for long versus short,
is exactly what `hedging.py`'s `theoretical_pnl` computes and what
`experiments/run_gamma_theta_pnl.py` validates against a real, mechanically simulated,
self-financing hedge. See the README for how closely the two agree.

So what does this actually say about "theta decay"? Theta isn't some independent force
acting on your position. It's the price, fixed by the no-arbitrage PDE, of being long
gamma. Whether that price was worth paying comes down entirely to a bet on realized
versus hedged volatility. There's no other source of P&L in a delta-hedged options
position under these assumptions (no jumps, continuous rebalancing, no transaction
costs). Discretizing the hedge, rebalancing at finite intervals instead of continuously,
introduces additional variance around this formula: hedging error whose standard
deviation shrinks as rebalancing gets more frequent. Importantly, it doesn't bias the
mean away from the formula above. Both properties are checked directly in
`tests/test_hedging.py` and in the frequency sweep in
`experiments/run_gamma_theta_pnl.py`.

## Hidden Markov Models for regime detection

Here's the intuition first. Imagine you can't directly see whether the market is calm or
turbulent right now. All you get to watch is the day-to-day returns. A Hidden Markov
Model is a tool that looks at that sequence of returns and figures out, probabilistically,
which of a small number of hidden "moods" the market is likely in, and how those moods
tend to persist or switch over time.

Formally: returns are drawn from one of $K$ (here, 2) Gaussian regimes, and the active
regime follows its own Markov chain that's never directly observed, only inferred from
its fingerprint on the returns. The parameters are $\pi_i = P(\text{state}_ 0 = i)$,
$A_ {ij} = P(\text{state}_ {t+1}=j \mid \text{state}_ t = i)$ (the transition matrix), and
$(\mu_i, \sigma_i)$, the emission distribution of a return while in state $i$.

Define $\hat\alpha_t(i) = P(\text{state}_ t = i \mid \text{returns}_ {0:t})$, the
filtered, strictly causal probability of being in state $i$ having seen only data up to
and including $t$. It has a clean recursion:

$$c_t\ \hat\alpha_t(j) = b_j(x_t) \sum_i \hat\alpha_ {t-1}(i)\ A_ {ij}$$

where $b_j(x_t) = N(x_t; \mu_j, \sigma_j^2)$ is the emission density and $c_t$ is chosen
so $\hat\alpha_t$ sums to 1. This $c_t$ isn't just a normalization convenience. It equals
$P(x_t \mid x_ {0:t-1})$, the one-step-ahead predictive probability of the observation
actually seen, which means $\prod_t c_t = P(x_ {0:T})$, the total data likelihood. So
$\sum_t \log c_t$ recovers the model's log-likelihood for free, as a byproduct of the
same recursion that computes the filtered probabilities. `filtered_state_probs` in
`regime.py` is exactly $\hat\alpha$, and it's what a live strategy is allowed to condition
on, since nothing in its recursion ever looks past $t$.

A second recursion, run backward from $T-1$ down to $0$, computes $\hat\beta_t(i)$: a
measure, consistently rescaled using the same $c_t$'s from the forward pass, of how
probable all the future observations are given state $t = i$. Combined:

$$\gamma_t(i) = \hat\alpha_t(i)\ \hat\beta_t(i) = P(\text{state}_ t = i \mid \text{ALL the data})$$

exactly, with no further renormalization needed, a consequence of how the forward and
backward scaling factors are constructed to cancel against each other. This is the best
possible hindsight estimate of the regime at $t$, valid only for offline analysis of a
completed series (`smoothed_state_probs`), never for a live decision, since it uses
information from after $t$.

Baum-Welch is the EM algorithm applied to this model. An E-step runs forward-backward
under the current parameter guess to get $\gamma_t(i)$ and the pairwise
$\xi_t(i,j) = P(\text{state}_ t=i, \text{state}_ {t+1}=j \mid \text{all data})$. An
M-step then re-estimates every parameter as a closed-form, $\gamma$/$\xi$-weighted MLE:

$$\pi_i \leftarrow \gamma_0(i) \qquad A_ {ij} \leftarrow \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)} \qquad \mu_i \leftarrow \frac{\sum_t \gamma_t(i)\ x_t}{\sum_t \gamma_t(i)} \qquad \sigma_i^2 \leftarrow \frac{\sum_t \gamma_t(i)(x_t - \mu_i)^2}{\sum_t \gamma_t(i)}$$

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
converge to sensible values. Only checking the log-likelihood trace against an
independent implementation caught it.

Initialization matters more than it might seem. Starting every state with identical
parameters is a fixed point of EM: perfect symmetry between states never breaks on its
own. `regime.fit` seeds different states with visibly different standard deviations,
spread around the data's overall standard deviation, to give EM something asymmetric to
sort out from the first iteration.

Viterbi answers a different question from smoothing entirely: not "what's the most likely
state at each $t$ independently," but "what's the most likely single path through all
states, taken as a whole." It's solved by dynamic programming in log-space. `delta[t,j]`
tracks the log-probability of the best path ending in state $j$ at time $t$, `psi[t,j]`
remembers which state at $t-1$ that best path came from, and a final backward pass
through the `psi` backpointers recovers the whole path from just its endpoint.

One design choice is worth explaining directly: why does the strategy in
`experiments/run_regime_gamma_scalping.py` refit per episode on a rolling window, instead
of just using one global fit? A live trading decision can only use data available at the
time, and a fixed global fit implicitly uses the entire history, including the future
relative to any earlier decision point, to set its parameters. That would be lookahead
bias, a form of cheating. Refitting a fresh 2-state EM model on only the trailing window
at each decision point is slower and noisier (a handful of parameters fit from a few
dozen to a couple hundred observations is a genuinely small-sample problem; see the
README for how classification accuracy scales with window length), but it's the honest,
causal version of the question "could a real strategy have known this at the time."
