# JumpHMM.jl Feature Request: Residual Bootstrap for SingleIndexModel

## Problem

`SingleIndexModel.sample_dependence()` uses Gaussian noise for residuals:

```julia
returns[t, i] = sim.α[i] + sim.β[i] * G_market[t] + sim.σ_ε[i] * randn()
```

This produces 0% KS pass rates for non-market assets because empirical SIM residuals are heavily fat-tailed (kurtosis 10+ for NVDA), but `randn()` produces kurtosis 0. The Gaussian assumption destroys all tail structure.

## Evidence (from Copula-Comparison.jl)

NVDA residual diagnostics:
- Empirical residual kurtosis: **10.33**
- SIM assumes: N(0, 4.897) with kurtosis **0**
- Simulated NVDA kurtosis: 2.38 vs observed 5.02
- KS pass rate: **0.0%** (all 1,000 paths fail)

The old paper code used residual bootstrap and achieved 58.4% mean KS across 424 assets.

## Proposed Fix

Store empirical residuals in `SingleIndexModel` and resample them during simulation:

```julia
struct SingleIndexModel <: AbstractDependenceModel
    α::Vector{Float64}
    β::Vector{Float64}
    σ_ε::Vector{Float64}
    residuals::Matrix{Float64}      # NEW: (T, n_assets) empirical residuals
    market_model::JumpHiddenMarkovModel
end
```

During fitting (in `fit(SingleIndexModel, ...)`), compute and store:
```julia
residuals[:, i] = G_i .- α[i] .- β[i] .* G_market
```

During simulation, resample with replacement instead of `randn()`:
```julia
function sample_dependence(sim::SingleIndexModel, n::Int)
    market_result = simulate(sim.market_model, n; n_paths=1)
    G_market = market_result.paths[1].observations
    T_train = size(sim.residuals, 1)

    n_assets = length(sim.α)
    returns = Matrix{Float64}(undef, n, n_assets)
    for i in 1:n_assets
        idx = rand(1:T_train, n)  # bootstrap indices
        for t in 1:n
            returns[t, i] = sim.α[i] + sim.β[i] * G_market[t] + sim.residuals[idx[t], i]
        end
    end
    return returns
end
```

This preserves:
- Fat tails in residual distribution (kurtosis 10+)
- Cross-asset residual correlation (if using shared bootstrap indices)
- Exact σ_ε (by construction)

## Optional Enhancement

Add a `residual_method` parameter to control behavior:
- `:bootstrap` (default) — resample empirical residuals
- `:gaussian` — current behavior (N(0, σ_ε))
- `:studentt` — fit Student-t to residuals per asset

## Files to Modify

- `src/Types.jl` — add `residuals` field to `SingleIndexModel`
- `src/SIM.jl` — update `fit()` to store residuals, update `sample_dependence()` to resample
