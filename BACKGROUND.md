# Background

Derivations for every formula `src/` implements: where each one comes from, what each
term does, and what breaks if you drop it. [README.md](README.md) covers results; this
covers mechanism.

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

A stock price follows geometric Brownian motion:

$$dS_t = \mu S_t\ dt + \sigma S_t\ dW_t$$

`dW_t` is a normal draw scaled by `sqrt(dt)`, not `dt`. Over one step of size `dt`, the
random shock has standard deviation `sigma * sqrt(dt)`. Every simulation in
`simulation.py` encodes this directly:

```python
Z = rng.standard_normal(n_paths)
return S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
```

`Z` is the standard normal draw, `sigma * np.sqrt(T)` is the scaling. Drop the square
root and the standard deviation of the shock scales with time instead of its square
root, which silently corrupts every downstream Greek.

Black, Scholes, and Merton's 1973 result: build a portfolio of the option plus $\Delta$
shares of stock, and its value over an infinitesimal step has no random term left in it.
The option's sensitivity to $S$ cancels the stock's own randomness exactly. A zero-risk
portfolio has to earn the risk-free rate $r$, or there's a riskless arbitrage: borrow at
$r$, buy the portfolio, pocket the difference. Applying Ito's lemma to $V(S,t)$ under that
constraint produces the Black-Scholes PDE:

$$\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = rV$$

$\mu$, the stock's real-world drift, does not appear. An option's price depends only on
$\sigma$, $r$, and the contract terms, not on whether you think the stock is going up or
down. Under this PDE the stock behaves as if its drift were $r$: the risk-neutral
measure. `simulation.gbm_terminal` and `gbm_path` simulate exactly this drift (`r - q`),
never $\mu$.

Solving the PDE with a European call's boundary condition $V(S,T) = \max(S-K, 0)$ gives:

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S/K) + \left(r - q + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}} \qquad d_2 = d_1 - \sigma\sqrt{T}$$

`black_scholes.py` implements `d1` directly:

```python
def d1(S, K, T, r, sigma, q=0.0):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
```

The sign on `0.5 * sigma**2` is positive. `gbm_terminal`, above, uses a negative sign for
a similar-looking term. Both are correct: `d1`'s drift term comes from integrating the
lognormal density against the payoff, a different derivation than "what's the expected
log-price," and the sign flips as a consequence. Swapping the two is the single most
common bug in a from-scratch Black-Scholes implementation, which is why `black_scholes.py`
calls it out at the point of use. `np.log(S / K)` is log-moneyness: zero at the money,
positive if $S > K$. Dividing by `sigma * np.sqrt(T)`, the standard deviation of
log-return over `[0, T]`, converts the numerator into a z-score, which is what makes
`norm.cdf(d1)` meaningful at all.

`price()`:

```python
D1, D2 = d1(S, K, T, r, sigma, q), d2(S, K, T, r, sigma, q)
disc_r = np.exp(-r * T)
disc_q = np.exp(-q * T)
if option_type == OptionType.CALL:
    return S * disc_q * norm.cdf(D1) - K * disc_r * norm.cdf(D2)
return K * disc_r * norm.cdf(-D2) - S * disc_q * norm.cdf(-D1)
```

`S * disc_q * norm.cdf(D1)` is the present value of the stock received, conditional on
finishing in the money. `norm.cdf(D1)` is not $P(S_T > K)$; that's `norm.cdf(D2)`.
`D1`'s term is a risk-neutral, stock-numeraire-weighted probability that falls out of the
same integral. `K * disc_r * norm.cdf(D2)` is the present value of the strike paid on
exercise, weighted by the actual exercise probability under the risk-neutral measure.

The put branch is the mirror image via `norm.cdf(-x) = 1 - norm.cdf(x)`, and it's
algebraically identical to computing the call and applying put-call parity:

$$C - P = S e^{-qT} - K e^{-rT}$$

This holds independent of Black-Scholes: a long call plus a short put has the same payoff
as a forward contract, so it has the same price. `tests/test_black_scholes.py::test_put_call_parity`
checks the identity to machine precision for any $\sigma$, since it holds for any pricing
model, not just this one.

## The Greeks, dissected

Partial derivatives of `price` with respect to each input. `analytic_greeks`
differentiates the closed form directly:

```python
pdf_d1 = norm.pdf(D1)
gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
vega = S * disc_q * np.sqrt(T) * pdf_d1 / 100.0
```

**Delta** ($\Delta = e^{-qT}N(d_1)$ for a call): dollars the option moves per dollar the
stock moves. Approximately, not exactly, the risk-neutral probability of finishing in the
money; the exact probability is `N(d2)`, not `N(d1)`. The two collapse to this form via
$S\phi(d_1) = Ke^{-(r-q)T}\phi(d_2)$, which follows from completing the square in the
$d_1$/$d_2$ definitions. `hedging.py` uses it directly: `target_shares = -g.delta`.

**Gamma** ($\Gamma = \partial\Delta/\partial S$): how fast delta itself moves. Identical
for calls and puts at the same strike and maturity, because $C - P = Se^{-qT} - Ke^{-rT}$
is linear in $S$, so its second derivative is exactly zero. Peaks at the money, collapses
toward zero deep in or out of the money. That collapse is why the binomial tree needs
tree-native Greeks (see below), and why Newton's method for implied vol can diverge (see
the IV section).

**Theta** ($\Theta = \partial V/\partial t$, calendar time): three terms, summed:

```python
theta = (
    -S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
    - r * K * disc_r * norm.cdf(D2)
    + q * S * disc_q * norm.cdf(D1)
) / 365.0
```

- `-S * disc_q * pdf_d1 * sigma / (2 * sqrt(T))`: "gamma rent," always negative, the cost
  of carrying convexity. Same expression as gamma up to a scale factor.
- `- r * K * disc_r * norm.cdf(D2)`: financing cost. As $T$ falls, `disc_r` rises toward
  1, so the present value of the strike rises, making the call slightly less valuable.
- `+ q * S * disc_q * norm.cdf(D1)`: missed dividends. Nonzero only when `q > 0`.

Dividing by 365 converts annual theta to dollars per calendar day. The put's theta flips
the sign on the last two terms: the put holder receives $K$ on exercise, so a rising
discount factor helps rather than hurts.

**Vega** ($\nu = \partial V/\partial\sigma$, per 1-vol-point): identical for calls and
puts, same parity argument as gamma. Always positive: more uncertainty about $S_T$ can
only help an option holder, since $\max(\cdot, 0)$ benefits from extra spread and never
loses from it.

**Rho** ($\rho = \partial V/\partial r$, per 1%): $\rho_ {\text{call}} =
KTe^{-rT}N(d_2)/100$. Positive because a higher rate raises the forward price, making
calls more valuable. The put's rho is the negative mirror.

## Monte Carlo pricing

Simulate the risk-neutral terminal distribution, average discounted payoffs:

```python
def gbm_terminal(S0, T, r, sigma, q, n_paths, rng):
    Z = rng.standard_normal(n_paths)
    return S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
```

The `-0.5 * sigma**2` term (opposite sign from `d1`) is the Ito correction. Because `log`
is concave, $E[\ln S_T] \neq \ln E[S_T]$, and this term is exactly what makes
$E[S_T] = S_0 e^{(r-q)T}$ come out right despite the concavity.
`tests/test_simulation.py::test_gbm_terminal_mean_matches_risk_neutral_drift` checks this
directly.

`monte_carlo.price` averages discounted payoffs over `Z`:

```python
ST = gbm_terminal(S, T, r, sigma, q, n_paths, rng)
payoffs = np.maximum(ST - K, 0.0) if option_type == OptionType.CALL else np.maximum(K - ST, 0.0)
disc = np.exp(-r * T)
return float(disc * payoffs.mean()), float(disc * payoffs.std(ddof=1) / np.sqrt(n_paths))
```

Two guarantees apply. Law of Large Numbers: the mean converges to the true Black-Scholes
price as `n_paths` grows. Central Limit Theorem: the standard error, the second return
value above, shrinks as `1/sqrt(n_paths)`. Quadrupling `n_paths` halves the error;
`experiments/run_mc_convergence.py` fits the exponent and checks it against -0.5.

`MonteCarloEngine` reseeds its RNG from `self.seed` on every `price()` call, not once per
engine lifetime. That makes `price()` pure: same inputs, bit-identical output. It also
means the five bumped calls inside the base class's finite-difference `greeks()` draw the
same underlying `Z`, so the differences that go into delta/gamma/vega are differences in
the deterministic bump, not sampling noise. That's the standard common-random-numbers
variance-reduction trick, free from choosing to reseed deterministically rather than
bolted on separately.

## The binomial tree

Chop time into `n_steps` intervals. At each step the stock moves up by factor `u` or down
by factor `d`. Walk backward from maturity to the root, computing the option value at
every node.

```python
u = np.exp(sigma * np.sqrt(dt))
d = 1.0 / u
p = (np.exp((r - q) * dt) - d) / (u - d)
```

`u = e^{\sigma\sqrt{dt}}` matches one step's log-return standard deviation under GBM: as
`dt -> 0`, the tree's distribution converges to the continuous lognormal one. `d = 1/u`
specifically, not some other down-factor, makes the tree recombine: an up-then-down move
lands on `S * u * d = S`, the same price a down-then-up move reaches. Without that, node
count grows as `2^N` instead of `N+1`. `p` is the risk-neutral up-probability, solved so
the tree's one-step expected return matches `e^{(r-q)dt}`. `binomial.py` raises if `p`
falls outside `(0, 1)`, which happens when `dt` is too large relative to `sigma`.

Backward induction:

```python
for i in range(n_steps - 1, -1, -1):
    values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
    if exercise == ExerciseStyle.AMERICAN:
        S_i = S * u**j * d**(i - j)
        intrinsic = np.maximum(S_i - K, 0.0) if option_type == OptionType.CALL else np.maximum(K - S_i, 0.0)
        values = np.maximum(values, intrinsic)
```

`values[1:]` and `values[:-1]` are a node's "up" and "down" children; the discounted,
probability-weighted average replaces a node-by-node loop with one vectorized line per
level. The American branch is the entire early-exercise mechanism: at every node, take
the larger of continuation value and intrinsic value. A European call or put never
touches that branch.

As `n_steps` grows, the CRR price converges to Black-Scholes for European exercise
(`experiments/run_american_premium.py` checks this). The convergence rate is `O(1/N)`,
not the faster `O(1/N^2)` some smoothed variants achieve, and it's often oscillatory.

Reading Greeks off this tree by bumping `S` and re-differencing doesn't work. For fixed
`u`, `d`, `N`, the tree's price as a function of root spot `S` is piecewise linear: a
small bump to `S` almost never flips which terminal nodes are in the money, so the price
is locally exactly linear in `S`. Central-difference delta is fine, since it measures a
slope. Gamma, the curvature, comes back as floating-point noise around zero, since there's
no real curvature except at a measure-zero set of spot prices. `binomial.py` reads delta
and gamma directly off the step-1 and step-2 nodes instead:

```python
S_u, S_d = S * u, S * d
delta = (step1_values[1] - step1_values[0]) / (S_u - S_d)

S_uu, S_ud, S_dd = S * u * u, S, S * d * d
delta_upper = (step2_values[2] - step2_values[1]) / (S_uu - S_ud)
delta_lower = (step2_values[1] - step2_values[0]) / (S_ud - S_dd)
gamma = (delta_upper - delta_lower) / (0.5 * (S_uu - S_dd))
```

Those nodes are already computed during backward induction, so this costs nothing extra,
and it isn't fooled by the piecewise-linear geometry.
`tests/test_binomial.py::test_tree_native_gamma_matches_black_scholes` regression-tests
exactly this failure mode. Theta compares the middle step-2 node (same spot `S`, `2*dt`
later) against the root, isolating pure time decay from any price move:

```python
theta = (step2_values[1] - price0) / (2 * dt) / 365.0
```

## Implied volatility: existence, uniqueness, and three ways to find it

Given a market price, find the $\sigma$ that reproduces it: solve
$f(\sigma) = C_ {\text{BS}}(\sigma) - C_ {\text{mkt}} = 0$.

Existence: as $\sigma \to 0$, the call price approaches discounted intrinsic value (a
deterministic stock has a deterministic, possibly-zero payoff). As $\sigma \to \infty$,
the price approaches $Se^{-qT}$ (near-certain exercise, so the option converges to owning
the stock). $C_ {\text{BS}}(\sigma)$ is continuous and sweeps between those two bounds, so
the Intermediate Value Theorem guarantees a root for any market price between them.

Uniqueness: vega is positive everywhere, so $C_ {\text{BS}}(\sigma)$ is strictly
increasing, and a strictly increasing continuous function crosses any horizontal line at
most once.

**Newton-Raphson**:

```python
for _ in range(max_iter):
    model_price = bs_price(S, K, T, r, sigma, option_type, q)
    diff = model_price - market_price
    if abs(diff) < tol:
        return sigma
    vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1(S, K, T, r, sigma, q))
    if abs(vega) < 1e-12:
        break
    sigma -= diff / vega
    sigma = max(sigma, 1e-8)
```

`sigma -= diff / vega` is the whole algorithm: replace $f$ with its tangent line at the
current guess, solve that exactly, repeat. Converges quadratically near the root (each
step roughly squares the number of correct digits) because tangent-line error shrinks
quadratically as step size shrinks. Fails exactly where the tangent-line approximation
stops holding: near-zero vega, deep ITM/OTM or near expiry. There, `diff / vega` divides
by a near-zero number, producing a step large enough to land somewhere with even smaller
vega than before. The failure compounds: one benchmark run recovers $\sigma =
841{,}630.998$ against a true value of $0.35$ (full grid in the README). The
`abs(vega) < 1e-12` check catches the case before it compounds; it doesn't prevent it.

**Brent's method**: no starting guess, no divergence risk. Maintains a bracket `[a, b]`
with `f(a)` and `f(b)` of opposite sign, so the root stays trapped inside. Bisection alone
would already converge, just slowly (linearly, one correct bit per step). Brent's method
first tries a faster step (secant, or inverse quadratic interpolation once three points
exist) and falls back to bisection whenever the fast step would land outside a safe region
of the bracket or isn't converging fast enough. Guaranteed convergence, superlinear speed
in practice (up to the golden-ratio rate, roughly 1.62 extra correct digits per step). Can
still land on a numerically wrong $\sigma$ when $f$ is nearly flat across the whole
bracket (deep ITM, near expiry): many different $\sigma$ values produce prices within
tolerance of each other there, so `abs(f(sigma)) < tol` passes without the recovered
$\sigma$ being close to the true one.

**Jaeckel's method**: normalize, then iterate with curvature. Rewrite the price in terms
of the forward $F = Se^{(r-q)T}$, log-forward-moneyness $x = \ln(F/K)$, and total variance
$\theta = \sigma\sqrt{T}$:

```python
F = S * np.exp((r - q) * T)
beta = market_price / (discount * np.sqrt(F * K))
x = np.log(F / K)
```

`beta` collapses every Black-Scholes price to a function of just `(x, theta)`, which makes
it possible to invert the small- and large-`|x|` asymptotics in closed form for a
genuinely good starting `theta` (`_initial_theta`, three branches for ATM/OTM/ITM). Then
iterate with Halley's method instead of Newton's:

```python
denom = 1.0 - (diff * vega2) / (2.0 * vega**2)
sigma -= (diff / vega) / denom if abs(denom) > 1e-12 else diff / vega
```

`vega2` is $f''$, the price's curvature in $\sigma$. When $f$ is locally linear
(`vega2 = 0`), `denom = 1` and this reduces to plain Newton. When $f$ curves, `denom`
bends the step size to compensate, giving cubic convergence (each step roughly triples
correct digits) and damping the step automatically near small vega, exactly where plain
Newton blows up. If Halley still fails to converge, `jaeckel.py` falls back to
`brent_solve`.

## The volatility smile and surface

Black-Scholes assumes one $\sigma$ prices every strike and maturity. Invert every listed
option's market price for its implied vol and, if the model were right, every one would
come back the same number. It doesn't. Plot implied vol against strike and it's a curve,
not a flat line: the **smile/skew**. Plot against maturity: the **term structure**.

Two explanations for the equity skew specifically. Fat tails: real return distributions
have more extreme-move probability mass than a lognormal predicts, so far-OTM options are
systematically underpriced by a constant-vol model, and the market corrects by implying a
higher $\sigma$ there. Crash-insurance demand: equities fall harder than they rise, so
low-strike puts see persistent buying pressure a constant-vol model doesn't anticipate.

Before inverting a quote, `surface.py` checks it clears the no-arbitrage floor:

```python
return max(spot * np.exp(-q * quote.maturity) - quote.strike * np.exp(-r * quote.maturity), 0.0)
```

A call worth less than this is stale or broken data: buy it, exercise, sell the stock, and
the arbitrage is riskless. Quotes at or below the floor get dropped before `implied_vols`
calls a solver at all.

## Multi-leg structures

A `Structure` is a tuple of legs, each with a signed `quantity` (positive long, negative
short) and the price it was entered at. Every named strategy (straddle, spread, covered
call) is the same type; only the legs differ.

```python
total = total + leg.quantity * (_intrinsic(leg.option, S_T) - leg.entry_price)
```

`quantity * (payoff - entry_price)` gives correct P&L for both directions without a
branch. Long ($\text{quantity}=+1$): payoff minus premium paid. Short
($\text{quantity}=-1$): `-(payoff - entry_price) = entry_price - payoff`, i.e. keep the
credit, owe the payoff if it finishes in the money.

Portfolio Greeks are the quantity-weighted sum of each leg's Greeks, since
differentiation is linear:

```python
for leg in structure.option_legs:
    g = engine.greeks(leg.option, market)
    delta += leg.quantity * g.delta
    gamma += leg.quantity * g.gamma
```

A short straddle (short one call, short one put, same strike) sums to negative gamma and
positive theta automatically: mirror images of the long straddle, with no special-casing
anywhere in the code. `hedging.py` reads `portfolio_greeks(...).delta` to know how many
shares offset a structure.

## Delta-hedging and the gamma/theta P&L identity

A trader holds a `Structure` and delta-hedges it: holds $-\Delta_ {\text{portfolio}}$
shares of stock so the combined position has zero delta at every instant. Let
$\Pi_t = V_t - \Delta_t S_t$.

$V(S,t)$ follows Ito's lemma under the actual realized volatility $\sigma_r$, not the
hedging volatility $\sigma_h$ used to compute $\Delta$ and $\Gamma$:

$$dV = \Theta\ dt + \Delta\ dS + \frac{1}{2}\Gamma\ \sigma_r^2 S^2\ dt$$

Subtract the hedge: the $\Delta\ dS$ terms cancel exactly, which is the entire point of
delta-hedging.

$$d\Pi = \Theta\ dt + \frac{1}{2}\Gamma\sigma_r^2 S^2\ dt$$

$\Theta$ was computed at $\sigma_h$, so it satisfies the Black-Scholes PDE at $\sigma_h$:
$\Theta = rV - rS\Delta - \frac{1}{2}\Gamma\sigma_h^2S^2$. The $rV - rS\Delta$ terms are
the financing income and cost of holding $\Pi$ in cash at $r$; they cancel against
`hedging.py`'s `cash *= exp(r*dt)` step. What's left:

$$\boxed{d\Pi_ {\text{net}} = \frac{1}{2}\Gamma\ S^2\left(\sigma_r^2 - \sigma_h^2\right)dt}$$

A delta-hedged position's P&L is proportional to gamma times the squared-volatility gap
between what actually happened and what was priced in. Long gamma ($\Gamma>0$) profits
when $\sigma_r > \sigma_h$; short gamma profits when $\sigma_r < \sigma_h$. Theta is not
an independent force: it's the price, fixed by the PDE, of being long gamma.

`simulate_delta_hedge` implements the discrete version of this loop:

```python
g = portfolio_greeks(_shift(structure, elapsed), hedge_market(S_i), engine)
target_shares = -g.delta
cash -= (target_shares - shares) * S_i
shares = target_shares
cash *= np.exp(r * dt)
```

`target_shares = -g.delta` is the definition of delta-hedged, recomputed at every
rebalance. `cash -= (target_shares - shares) * S_i` is the self-financing trade: buy or
sell the difference at today's price, funded from cash, nothing injected or withdrawn.
`cash *= np.exp(r * dt)` accrues interest on whatever's left between rebalances. The
theoretical P&L is computed along the identical simulated path, using the identical
gammas the mechanical loop traded on:

```python
theoretical_pnl = float(0.5 * np.sum(gamma_path * S_grid**2 * (realized_vol**2 - hedge_vol**2) * dt))
```

Same path, same gammas, two independent computations of P&L that should agree.
`experiments/run_gamma_theta_pnl.py` confirms they do, within discretization noise that
shrinks as rebalancing gets more frequent (numbers in the README).

## Hidden Markov Models for regime detection

Returns are drawn from one of $K$ (here, 2) Gaussian regimes; the active regime follows a
Markov chain that's never directly observed. Parameters: $\pi_i = P(\text{state}_ 0=i)$,
transition matrix $A_ {ij} = P(\text{state}_ {t+1}=j \mid \text{state}_ t=i)$, emission means
and standard deviations per state.

Forward pass, filtering causally:

```python
alpha_hat[0] = params.startprob * B[0]
c[0] = alpha_hat[0].sum()
alpha_hat[0] /= c[0]

for t in range(1, n_obs):
    alpha_hat[t] = B[t] * (alpha_hat[t - 1] @ params.transmat)
    c[t] = alpha_hat[t].sum()
    alpha_hat[t] /= c[t]
```

`alpha_hat[t, i]` is $P(\text{state}_ t=i \mid \text{returns}_ {0:t})$, renormalized to sum
to 1 at every step to prevent underflow. `c[t]` is not just a normalizer: it equals
$P(x_t \mid x_ {0:t-1})$, so `sum(log(c))` recovers the model's log-likelihood as a
byproduct of the same loop. `filtered_state_probs` returns exactly this array, and it's
the only regime estimate a live strategy is allowed to condition on, since nothing in the
loop looks past `t`.

Baum-Welch alternates an E-step (forward-backward, producing `gamma` and pairwise `xi`)
with an M-step, a closed-form re-estimation:

```python
new_startprob = gamma[0].copy()
new_transmat = xi_sum / gamma[:-1].sum(axis=0, keepdims=True).T
new_means = (gamma * returns[:, None]).sum(axis=0) / weights
new_stdevs = np.sqrt(np.maximum(new_var, 1e-10))
```

Each update is a weighted MLE, where the weight on observation `t` for state `i` is how
much of the soft posterior mass at `t` belongs to state `i`. EM guarantees the
log-likelihood never decreases across iterations;
`tests/test_regime.py::test_log_likelihood_is_monotonically_non_decreasing` checks this
against an independent, unscaled log-sum-exp implementation. That test caught a real bug
during development: an earlier version applied a numerical stabilization to emission
probabilities that canceled correctly out of `alpha_hat`, `beta_hat`, and `gamma`, but not
out of the log-likelihood computed from `sum(log(c))`. Parameters converged to sensible
values the entire time; the log-likelihood trace used to monitor convergence was silently
wrong.

Starting every state with identical parameters is a fixed point of EM (perfect symmetry
never breaks on its own), so `fit()` seeds states with different standard deviations,
spread around the data's overall standard deviation.

Viterbi answers a different question: not the most likely state at each `t`
independently, but the most likely single path through all states.

```python
scores = delta[t - 1][:, None] + log_transmat
psi[t] = np.argmax(scores, axis=0)
delta[t] = scores[psi[t], np.arange(n_states)] + logB[t]
```

`delta[t, j]` is the log-probability of the best path ending in state `j` at time `t`.
`psi[t, j]` is the backpointer: which state at `t-1` that best path came from. A final
backward pass through `psi` recovers the whole path from just its endpoint.

`experiments/run_regime_gamma_scalping.py` refits on a trailing window before each
decision instead of using one global fit. A fixed global fit implicitly uses the entire
history, including data from after any earlier decision point, which is lookahead bias.
Refitting on only the trailing window is slower and noisier (small-sample EM on a few
dozen to a couple hundred observations), but it's the causal version of the question a
real strategy has to answer: could this have been known at the time.
