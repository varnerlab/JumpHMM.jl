# JumpHMM.jl — HybridSingleIndexModel Update Plan

## Context

The eCornell AI Finance course implements a "hybrid SIM" construction in `eCornellAIFinance.Compute.jl` (lines 954–1171) that combines the Single Index Model with JumpHMM marginals via a variance-correction trick. This logic is general-purpose but currently locked inside the course package. We want to upstream it into JumpHMM.jl so it's reusable, and potentially write a short arXiv note about it.

The existing `SingleIndexModel` in JumpHMM.jl (`src/SIM.jl`) is the naive version — no variance correction, no R²-preserving branch, no β clipping. We keep it as-is and add a new type alongside it.

## Design Decisions (confirmed with user)

1. **New type** `HybridSingleIndexModel <: AbstractDependenceModel` (not extending existing `SingleIndexModel`)
2. **Copula field** is `AbstractDependenceModel` — users can plug in `StudentTCopula`, `GaussianCopula`, `VineCopula`
3. **End-to-end `fit`** from raw prices (fits market HMM, per-ticker marginals, OLS, copula on OLS residuals)
4. **Integrates with `PortfolioModel`** via `dependence=HybridSingleIndexModel`
5. **r2_preserve_threshold and idiosyncratic floor** are fields on the type (set at fit time)
6. **Returns growth rates only** — no price conversion (matches existing `sample_dependence` contract)
7. **Copula fitted on OLS residuals** (not raw returns) — theoretically correct, avoids double-counting market correlation
8. **Existing `SingleIndexModel` untouched** — no breaking changes

## Files to Modify (all in JumpHMM.jl repo)

### 1. `src/Types.jl` — Add `HybridSingleIndexModel` struct

Insert after `SingleIndexModel` (line ~231):

```julia
struct HybridSingleIndexModel <: AbstractDependenceModel
    α::Vector{Float64}                        # per-ticker intercepts
    β::Vector{Float64}                        # per-ticker calibrated betas
    r²::Vector{Float64}                       # per-ticker R² from OLS
    σ_market::Float64                         # market growth-rate std
    marginals::Dict{String,JumpHiddenMarkovModel}  # per-ticker HMM marginals
    copula::AbstractDependenceModel           # fitted on OLS residuals
    market_model::JumpHiddenMarkovModel       # fitted market HMM
    tickers::Vector{String}                   # ordered ticker list (matches α, β, r² indexing)
    r2_preserve_threshold::Float64            # R² cutoff for R²-preserving branch (default 0.80)
    idiosyncratic_floor::Float64              # minimum idiosyncratic variance share (default 0.10)
end
```

### 2. `src/SIM.jl` — Add `fit` and `sample_dependence` for the new type

**`fit(::Type{HybridSingleIndexModel}, ...)`**

Signature:
```julia
function fit(::Type{HybridSingleIndexModel},
             tickers::Vector{String},
             prices::AbstractMatrix{<:Real},
             market_ticker::String;
             copula_type::Type{<:AbstractDependenceModel}=StudentTCopula,
             rf::Float64=0.0, N::Int=100, ν::Float64=5.0,
             dt::Float64=1/252, min_obs::Int=2,
             r2_preserve_threshold::Float64=0.80,
             idiosyncratic_floor::Float64=0.10)
```

Steps:
1. Identify market column from `tickers` / `market_ticker`
2. Compute excess growth rates via `excess_growth_rates(prices; rf, dt)`
3. Fit market HMM: `fit(JumpHiddenMarkovModel, prices[:, market_col]; rf, N, ν, dt, min_obs)`
4. Fit per-ticker HMM marginals (excluding market): `fit(JumpHiddenMarkovModel, prices[:, j]; ...)` for each non-market ticker
5. OLS regression per non-market ticker against market returns → α, β, σ_ε, r², residual vectors
6. Compute `σ_market = std(market_returns)`
7. Fit copula on the residual matrix: `fit(copula_type, residual_matrix)`
8. Return `HybridSingleIndexModel(α, β, r², σ_market, marginals, copula, market_model, asset_tickers, r2_preserve_threshold, idiosyncratic_floor)`

**`sample_dependence(model::HybridSingleIndexModel, n)`**

This is the core hybrid recipe. Returns `(n × n_assets)` matrix of growth rates.

Steps (per call):
1. Simulate one market path: `simulate(model.market_model, n; n_paths=1)` → `G_market` (trim first obs)
2. Sample copula uniforms: `sample_dependence(model.copula, n)` → `U` matrix
3. For each ticker `k`:
   a. Simulate HMM marginal: `simulate(model.marginals[ticker], n; n_paths=1)` → `obs_j` (trim first obs)
   b. Compute `σ²_HMM = var(obs_j)`
   c. Look up `α_k, β_k, r²_k`
   d. **Branch on r²_k**:
      - If `r²_k ≥ r2_preserve_threshold` → **R²-preserving branch**:
        - `σ²_ε_target = β_k² · σ_market² · (1 - r²_k) / r²_k`  (0 when r²≈1)
        - `ε_scaled = sqrt(σ²_ε_target / σ²_HMM) .* obs_j`  (zeros if either is 0)
      - Else → **Marginal-preserving branch** with β clipping:
        - `ρ = β_k² · σ_market² / σ²_HMM`
        - If `ρ > 1 - f`: clip β, set `s² = f`
        - Else: `s² = 1 - ρ`
        - `ε_scaled = sqrt(s²) .* obs_j`
   e. **Copula rank-reorder**: `sorted_eps = sort(ε_scaled)`, reorder by `ordinalrank(U[:, k])`
   f. **Compose**: `g_k = α_k .+ β_k .* G_market .+ ε_reordered`
4. Return `(T_eff × n_assets)` matrix

### 3. `src/Portfolio.jl` — Integrate with `PortfolioModel`

**`fit(PortfolioModel, ...; dependence=HybridSingleIndexModel)`** (add `elseif` branch ~line 32):
- Requires `market` kwarg (same as `SingleIndexModel` branch)
- Separate market column from asset columns
- Call `fit(HybridSingleIndexModel, tickers, prices, market; copula_type, ...)`
- Build `PortfolioModel` with the hybrid as dependence, marginals from the hybrid model, market_ticker set

**`simulate(portfolio, ...)`** (add `elseif` branch ~line 158):
- Similar to the existing `SingleIndexModel` branch
- Call `sample_dependence(dep, n_steps)` per path
- Pack into `SimulationPath` (states/jumps filled with sentinel values, same pattern as naive SIM)

**`tune(portfolio, ...)`** (add handling ~line 91):
- Tune market model inside the `HybridSingleIndexModel`
- Tune each per-ticker marginal
- Return new `PortfolioModel` with updated hybrid model

### 4. `src/JumpHMM.jl` — Add export

Add `HybridSingleIndexModel` to the exports (line 62):
```julia
export GaussianCopula, StudentTCopula, SingleIndexModel, HybridSingleIndexModel
```

### 5. `test/test_sim.jl` — Add tests

New `@testset "Hybrid Single-Index Model"` with:

1. **fit and sample** — synthetic market + 3 assets with known β, verify fit returns correct fields, `sample_dependence` returns correct shape
2. **Variance preservation (marginal-preserving branch)** — for low-R² assets, verify `var(g_i) ≈ σ²_HMM` within tolerance
3. **R² recovery (R²-preserving branch)** — for high-R² assets, regress simulated g_i on g_m, verify recovered R² ≈ target R²
4. **β recovery** — regress simulated g_i on g_m, verify recovered β ≈ calibrated β
5. **β clipping** — construct a case where `ρ > 0.90`, verify β gets clipped and idiosyncratic floor holds
6. **SPY degenerate limit** — market ticker regressed against itself: R²=1, ε≡0, g = α + β·g_m deterministically
7. **Copula pluggability** — fit with `GaussianCopula` instead of default `StudentTCopula`, verify it works
8. **PortfolioModel integration** — `fit(PortfolioModel, ...; dependence=HybridSingleIndexModel, market="MKT")` → `simulate` → correct output shape

## Implementation Order

1. Types.jl (struct definition)
2. SIM.jl (`fit` + `sample_dependence`)
3. Portfolio.jl (integration branches)
4. JumpHMM.jl (export)
5. Tests

## Verification

```bash
cd /path/to/JumpHMM.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

All existing tests must continue to pass (no breaking changes). New tests must cover the 8 scenarios above.

## What's NOT in scope

- Price-path construction / `MyBacktestScenario` (stays in eCornell)
- Starting prices, Δt-based price recursion
- Changes to existing `SingleIndexModel`
- README / docs updates (separate PR)
- arXiv paper (separate effort)

## Reference: The Variance Correction Math

See `eCornell-AI-finance-lectures/code/scripts/hybrid.md` for the full derivation. Key identities:

**Marginal-preserving branch** (low R²):
- Scale factor: `s² = 1 - ρ` where `ρ = β²σ²_m / σ²_gen`
- β clipping when `ρ > 1 - f`: `β_eff = sign(β) · sqrt((1-f) · σ²_gen / σ²_m)`, `s² = f`

**R²-preserving branch** (high R²):
- Target residual variance: `σ²_ε_target = β² · σ²_m · (1 - R²) / R²`
- SPY limit (R²=1): `σ²_ε_target = 0` → deterministic `g = α + β·g_m`
