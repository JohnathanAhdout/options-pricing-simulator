# Options Pricing & Simulator

A quantitative finance toolkit built from first principles, implementing the core models used in institutional derivatives trading and academic research.

## What This Covers

**Black-Scholes Closed-Form Pricing**  
Full implementation of the BS model for European calls and puts, including all five Greeks (delta, gamma, theta, vega, rho) derived analytically from the pricing formula using partial derivatives.

**Monte Carlo Option Pricing**  
Independent price estimation via Geometric Brownian Motion path simulation (M=100,000 paths). Converges to the closed-form BS price by the Law of Large Numbers — used to cross-validate the analytical model without relying on a closed-form solution.

**Implied Volatility via Newton-Raphson**  
Numerical root-finding to invert the BS formula for σ given observed market prices. Since σ is embedded inside two normal CDFs and has no algebraic inverse, the solver uses vega (∂C/∂σ) as the derivative at each iteration, guaranteeing convergence given the strict monotonicity of BS price in volatility.

**Volatility Surface Construction from Live Market Data**  
Pulls real SPY options chain data via `yfinance`, computes IV for each (K, T) pair, and interpolates onto a uniform grid using `scipy.griddata` to produce a 3D volatility surface. Demonstrates the empirical failure of the constant-volatility assumption — the volatility smile/skew is visible in live data.

**Greek Surfaces Over (S, T) Space**  
Visualizes how delta, gamma, theta, and vega evolve simultaneously as a function of spot price and time to expiry, showing the nonlinear, time-decaying nature of options exposure.

**Multi-Leg Strategy Payoff Diagrams**  
Constructs and prices common options strategies (long call, long put, straddle, covered call, bull call spread) as portfolios of BS-priced legs. Generates payoff-at-expiry diagrams to visualize bounded/unbounded risk profiles.

**Mark-to-Market P&L Simulation**  
Simulates 300 GBM stock price paths over 126 trading days (~6 months) and computes the mark-to-market value of each strategy at every time step using live BS re-pricing. Produces a full distribution of P&L trajectories before expiry.

## Stack

`Python` · `NumPy` · `SciPy` · `yfinance` · `Matplotlib`

## Key Concepts Demonstrated

- Stochastic calculus (GBM, risk-neutral measure)
- Numerical methods (Newton-Raphson, Monte Carlo simulation)
- Financial derivatives pricing theory
- Market microstructure (IV skew as a real-world BS violation)
- Portfolio-level Greeks and risk management
