# Getting Started

## Quick Start: Single Asset

```julia
using JumpHMM

# Load your price data (Vector{Float64} of daily closing prices)
prices = [...]

# Step 1: Fit the base model (partition + transitions + emissions)
model = fit(JumpHiddenMarkovModel, prices; rf=0.05, N=100, ν=5.0)

# Step 2: Tune jump parameters via grid search
model = tune(model, prices;
    ϵ_range=range(1e-4, 2.5e-2, length=20),
    λ_range=range(10, 160, length=16),
    n_paths=200, w_κ=0.20)

# Step 3: Simulate 1000 synthetic paths of 252 trading days
result = simulate(model, 252; n_paths=1000, seed=1234)

# Step 4: Validate
report = validate(model, prices; n_paths=1000)
println("KS pass rate: $(report.ks_pass_rate)")
println("Mean ACF-MAE: $(report.mean_acf_mae)")
```

## Quick Start: Multi-Asset Portfolio

```julia
using JumpHMM

tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
price_matrix = [...]  # (n_obs × 5) matrix

# Fit marginals + Student-t copula (default)
portfolio = fit(PortfolioModel, tickers, price_matrix;
    dependence=StudentTCopula, rf=0.05, N=100)

# Tune and simulate
portfolio = tune(portfolio, price_matrix)
port_result = simulate(portfolio, 252; n_paths=1000)

# Validate each asset
reports = validate(portfolio, price_matrix)
for (ticker, report) in reports
    println("$ticker: KS=$(report.ks_pass_rate), ACF-MAE=$(report.mean_acf_mae)")
end
```

## How It Works

The model is defined by five components:

1. **Laplace Partition**: Continuous observations are discretized into `N` states (default 100) using equal-probability quantile bins of a fitted Laplace distribution
2. **Transition Matrix**: Estimated by direct frequency counting of observed state-to-state transitions (no Baum-Welch/EM)
3. **Emissions**: Per-state location-scale Student-t distributions with `ν=5` degrees of freedom
4. **Stationary Distribution**: Computed via matrix exponentiation `T^50`
5. **Jump Mechanism**: With probability `ϵ` per step, the chain is forced into tail states for `Poisson(λ)` consecutive steps, producing volatility clustering

The `fit` → `tune` two-step workflow separates the deterministic model fitting (fast) from the stochastic jump parameter optimization (expensive grid search).
