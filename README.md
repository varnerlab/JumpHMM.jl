# JumpHMM.jl

[![CI](https://github.com/varnerlab/JumpHMM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/varnerlab/JumpHMM.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/varnerlab/JumpHMM.jl/actions/workflows/documentation.yml/badge.svg)](https://varnerlab.github.io/JumpHMM.jl/dev/)

`JumpHMM.jl` is a Julia package implementing a Hybrid Hidden Markov Model with Poisson jump-diffusion for generating synthetic equity time series. The package implements the method described in:

> Alswaidan A. and Varner J. *Hybrid Hidden Markov Model for Modeling Equity Excess Growth Rate Dynamics: A Discrete-State Approach with Jump-Diffusion*. Cornell University.

## Documentation
Full documentation is available at: [https://varnerlab.github.io/JumpHMM.jl/dev/](https://varnerlab.github.io/JumpHMM.jl/dev/)

## Installation
`JumpHMM.jl` can be installed from the Julia REPL:
```julia
using Pkg
Pkg.add(url="https://github.com/varnerlab/JumpHMM.jl.git")
```

## Single-Asset Example
```julia
using JumpHMM

# Load your daily closing prices
prices = [...]  # Vector{Float64}

# Fit the base model (Laplace partition + transition matrix + Student-t emissions)
model = fit(JumpHiddenMarkovModel, prices; rf=0.05, N=100, ν=5.0)

# Tune jump parameters (ϵ, λ) via grid search
model = tune(model, prices;
    ϵ_range=range(1e-4, 2.5e-2, length=20),
    λ_range=range(10, 160, length=16),
    n_paths=200, w_κ=0.20)

# Simulate 1000 synthetic paths of 252 trading days
result = simulate(model, 252; n_paths=1000, seed=1234)

# Validate against empirical data
report = validate(model, prices; n_paths=1000)
println("KS pass rate: $(report.ks_pass_rate)")
println("Mean ACF-MAE: $(report.mean_acf_mae)")

# Decode hidden states from new observations
G = excess_growth_rates(new_prices; rf=0.05)
states = decode(model, G)
```

## Multi-Asset Portfolio Example
Three interchangeable dependence models are available behind `AbstractDependenceModel`:

| Model | Dependence | Tail Dependence | Best For |
|-------|-----------|-----------------|----------|
| `StudentTCopula` | Full n×n | Yes (via ν) | Realistic portfolios |
| `GaussianCopula` | Full n×n | None | Simple baseline |
| `SingleIndexModel` | Through 1 factor | Inherited from market | 100+ asset universes |
| `HybridSingleIndexModel` | 1 factor + HMM marginals + copula | Yes (from marginals + copula) | Variance-preserving synthetic data |

```julia
using JumpHMM

tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
price_matrix = [...]  # (n_obs × 5) matrix

# Copula approach (default: Student-t copula)
portfolio = fit(PortfolioModel, tickers, price_matrix;
    dependence=StudentTCopula, rf=0.05, N=100)
portfolio = tune(portfolio, price_matrix)
result = simulate(portfolio, 252; n_paths=1000)

# Single-Index Model approach (for large universes)
portfolio = fit(PortfolioModel, tickers, price_matrix;
    dependence=SingleIndexModel, market="SPY", rf=0.05)

# Hybrid SIM approach (variance-corrected, preserves HMM marginals)
portfolio = fit(PortfolioModel, tickers, price_matrix;
    dependence=HybridSingleIndexModel, market="SPY", rf=0.05)
```

The interface is identical after `fit` — `tune`, `simulate`, and `validate` work the same regardless of the dependence model.

## How It Works
The model discretizes continuous excess growth rates into `N` states (default 100) using equal-probability quantile bins of a fitted Laplace distribution. A transition matrix is estimated by frequency counting, and per-state Student-t(ν=5) emission distributions capture the observation density. The key innovation is a Poisson jump mechanism: with probability `ϵ` per time step, the chain is forced into tail states for `Poisson(λ)` consecutive steps, producing the persistent volatility clustering observed in real markets.

Only 4 scalar parameters require explicit optimization: `μ` and `b` (Laplace location/scale, fit via MLE), `ϵ` (jump probability), and `λ` (mean jump duration, tuned via grid search).

## Disclaimer and Risks
__This content is offered solely for training and informational purposes__. No offer or solicitation to buy or sell securities or derivative products or any investment or trading advice or strategy is made, given, or endorsed by the teaching team.

__Trading involves risk__. Carefully review your financial situation before investing in securities, futures contracts, options, or commodity interests. Past performance, whether actual or indicated by historical tests of strategies, is no guarantee of future performance or success. Trading is generally inappropriate for someone with limited resources, investment or trading experience, or a low-risk tolerance. Only risk capital that is not required for living expenses.

__You are fully responsible for any investment or trading decisions you make__. Such decisions should be based solely on evaluating your financial circumstances, investment or trading objectives, risk tolerance, and liquidity needs.
