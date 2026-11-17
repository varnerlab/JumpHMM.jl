# JumpHMM.jl — Package Specification (v1)

**Date:** 2026-03-25
**Authors:** Jeffrey Varner, Abdulrahman Alswaidan
**Paper:** *Hybrid Hidden Markov Model for Modeling Equity Excess Growth Rate Dynamics: A Discrete-State Approach with Jump-Diffusion*
**Paper repo:** https://github.com/varnerlab/HMM-w-jumps-paper.git
**Reference impl:** https://github.com/varnerlab/VLQuantitativeFinancePackage.jl.git

---

## 1. Overview

`JumpHMM.jl` is a self-contained Julia package implementing a Hybrid Hidden Markov Model with Poisson jump-diffusion for generating synthetic equity time series. The package reproduces three canonical stylized facts of financial returns: heavy-tailed distributions, negligible linear autocorrelation, and persistent volatility clustering.

**Design principles:**
- All structs are immutable — fitting/tuning return new instances
- Self-contained — all finance helpers built in, no external dependencies on VLQuantitativeFinancePackage.jl
- Single-asset and multi-asset (via copulas, not Single-Index Model)

---

## 2. Package Structure

```
JumpHMM.jl/
├── src/
│   ├── JumpHMM.jl          # module definition, includes, exports
│   ├── Types.jl             # all abstract and concrete types
│   ├── Partition.jl         # Laplace fitting, quantile state discretization
│   ├── Transition.jl        # frequency-counting transition matrix estimation
│   ├── Emission.jl          # per-state Student-t emission fitting
│   ├── Simulate.jl          # forward simulation with jump mechanism
│   ├── Decode.jl            # Viterbi decoding, forward filtering
│   ├── Tune.jl              # grid search for (ϵ, λ)
│   ├── Validate.jl          # KS, AD, ACF, kurtosis, Wasserstein, Hellinger
│   ├── Copula.jl            # Gaussian and Student-t copula fitting/sampling
│   ├── SIM.jl               # Single-Index Model fitting/sampling
│   ├── Portfolio.jl         # multi-asset model construction and simulation
│   └── Finance.jl           # excess growth rates, price reconstruction
├── test/
│   ├── runtests.jl
│   ├── test_partition.jl
│   ├── test_transition.jl
│   ├── test_emission.jl
│   ├── test_simulate.jl
│   ├── test_decode.jl
│   ├── test_tune.jl
│   ├── test_validate.jl
│   ├── test_copula.jl
│   ├── test_sim.jl
│   ├── test_portfolio.jl
│   └── test_finance.jl
├── Project.toml
└── LICENSE
```

---

## 3. Dependencies

```toml
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
HypothesisTests = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
```

---

## 4. Type System

### 4.1 Abstract Types

```julia
"""
Supertype for all Markov models in the package.
"""
abstract type AbstractMarkovModel end

"""
Supertype for copula-based dependence models.
"""
abstract type AbstractDependenceModel end

"""
Supertype for validation report types.
"""
abstract type AbstractValidationResult end
```

### 4.2 State Partition

```julia
"""
    LaplacePartition

Defines the discretization of continuous observations into N states using
equal-probability quantile bins of a fitted Laplace distribution.

Quantile boundaries: Q_k = F_L⁻¹(k/N; μ, b) for k = 0, 1, ..., N.
The first boundary is -Inf and the last is +Inf to capture all tails.

## Fields
- `μ::Float64` — Laplace location parameter (MLE from data)
- `b::Float64` — Laplace scale parameter (MLE from data)
- `N::Int` — number of discrete states
- `boundaries::Vector{Float64}` — length (N+1) quantile boundaries,
   where boundaries[1] = -Inf and boundaries[end] = +Inf
"""
struct LaplacePartition
    μ::Float64
    b::Float64
    N::Int
    boundaries::Vector{Float64}

    function LaplacePartition(μ::Float64, b::Float64, N::Int)
        cutoffs = range(0.0, 1.0, length=N+1) |> collect
        d = Laplace(μ, b)
        boundaries = [quantile(d, p) for p in cutoffs]
        boundaries[1] = -Inf
        boundaries[end] = Inf
        return new(μ, b, N, boundaries)
    end
end
```

### 4.3 Emission Distributions

```julia
"""
    StudentTEmission

Per-state emission distribution: location-scale Student-t.
    G_t | S_t = k  ~  μ_k + σ_k × t_ν

When a state has fewer than `min_obs` observations (default 2),
the emission is fitted from the global dataset and `is_fallback = true`.

## Fields
- `μ::Float64` — location (sample mean of observations assigned to this state)
- `σ::Float64` — scale (sample std of observations assigned to this state)
- `ν::Float64` — degrees of freedom (default 5.0)
- `n_obs::Int` — number of observations used to fit this emission
- `is_fallback::Bool` — true if fitted from global data due to insufficient observations
"""
struct StudentTEmission
    μ::Float64
    σ::Float64
    ν::Float64
    n_obs::Int
    is_fallback::Bool
end
```

### 4.4 Jump Parameters

```julia
"""
    JumpParameters

Controls the Poisson-driven jump mechanism that forces the Markov chain
into tail states for extended durations.

At each time step:
- With probability (1 - ϵ): normal Markovian transition
- With probability ϵ: a jump is triggered
  - Duration K ~ Poisson(λ) consecutive steps
  - Each step lands in bottom N_tail states (prob p_neg) or top N_tail states (prob 1-p_neg)
  - After K steps, normal dynamics resume

## Fields
- `ϵ::Float64` — jump probability per time step (paper optimal: 1e-4 for SPY)
- `λ::Float64` — mean jump duration in steps (paper optimal: 100 for SPY)
- `p_neg::Float64` — probability of jumping to negative/bottom tail (default 0.52)
- `N_tail::Int` — number of tail states on each side (default 5)
"""
struct JumpParameters
    ϵ::Float64
    λ::Float64
    p_neg::Float64
    N_tail::Int

    function JumpParameters(ϵ::Float64, λ::Float64;
                            p_neg::Float64=0.52, N_tail::Int=5)
        @assert 0.0 ≤ ϵ ≤ 1.0 "ϵ must be in [0, 1]"
        @assert λ > 0.0 "λ must be positive"
        @assert 0.0 ≤ p_neg ≤ 1.0 "p_neg must be in [0, 1]"
        @assert N_tail ≥ 1 "N_tail must be ≥ 1"
        return new(ϵ, λ, p_neg, N_tail)
    end
end
```

### 4.5 The Model

```julia
"""
    JumpHiddenMarkovModel <: AbstractMarkovModel

A fitted single-asset Jump-HMM. Fully immutable — all fitting and tuning
operations return a new instance.

The model encapsulates:
1. A Laplace-based state partition (N discrete states from quantile bins)
2. An N×N row-stochastic transition matrix (estimated by frequency counting)
3. Per-state Student-t emission distributions
4. A stationary distribution π̄
5. Jump parameters (ϵ, λ, p_neg, N_tail)

## Fields
- `partition::LaplacePartition` — state discretization
- `transition::Matrix{Float64}` — N×N row-stochastic transition matrix
- `emissions::Vector{StudentTEmission}` — length-N per-state emission distributions
- `stationary::Vector{Float64}` — length-N stationary distribution π̄ = (T^50)[1,:]
- `jump::JumpParameters` — jump mechanism parameters
- `ν::Float64` — Student-t degrees of freedom used for all emissions
- `rf::Float64` — risk-free rate used during fitting
- `dt::Float64` — time step used during fitting (default 1/252)
"""
struct JumpHiddenMarkovModel <: AbstractMarkovModel
    partition::LaplacePartition
    transition::Matrix{Float64}
    emissions::Vector{StudentTEmission}
    stationary::Vector{Float64}
    jump::JumpParameters
    ν::Float64
    rf::Float64
    dt::Float64
end
```

### 4.6 Simulation Output Types

```julia
"""
    SimulationPath

A single simulated path from the Jump-HMM.

## Fields
- `states::Vector{Int}` — hidden state index at each time step
- `observations::Vector{Float64}` — sampled excess growth rates
- `jumps::Vector{Bool}` — true at time steps where a jump was active
"""
struct SimulationPath
    states::Vector{Int}
    observations::Vector{Float64}
    jumps::Vector{Bool}
end

"""
    SimulationResult

Collection of simulated paths from a single model.

## Fields
- `paths::Vector{SimulationPath}` — individual path results
"""
struct SimulationResult
    paths::Vector{SimulationPath}
end
```

### 4.7 Validation Types

```julia
"""
    PathTestResult

Statistical test results for a single simulated path compared
against the empirical distribution.

## Fields
- `ks_pvalue::Float64` — Kolmogorov-Smirnov two-sample p-value
- `ad_pvalue::Float64` — Anderson-Darling k-sample p-value
- `wasserstein::Float64` — Wasserstein-1 (earth mover's) distance
- `hellinger::Float64` — Hellinger distance
- `acf_mae::Float64` — mean absolute error of |returns| autocorrelation
- `kurtosis_sim::Float64` — excess kurtosis of simulated path
- `kurtosis_emp::Float64` — excess kurtosis of empirical data
"""
struct PathTestResult
    ks_pvalue::Float64
    ad_pvalue::Float64
    wasserstein::Float64
    hellinger::Float64
    acf_mae::Float64
    kurtosis_sim::Float64
    kurtosis_emp::Float64
end

"""
    ValidationReport <: AbstractValidationResult

Aggregate validation report across many Monte Carlo paths.

## Fields
- `path_results::Vector{PathTestResult}` — per-path test results
- `ks_pass_rate::Float64` — fraction of paths with KS p-value > α
- `ad_pass_rate::Float64` — fraction of paths with AD p-value > α
- `mean_acf_mae::Float64` — mean ACF-MAE across paths
- `mean_wasserstein::Float64` — mean Wasserstein-1 distance
- `mean_hellinger::Float64` — mean Hellinger distance
- `mean_kurtosis::Float64` — mean excess kurtosis across paths
- `α::Float64` — significance level used for pass/fail (default 0.05)
"""
struct ValidationReport <: AbstractValidationResult
    path_results::Vector{PathTestResult}
    ks_pass_rate::Float64
    ad_pass_rate::Float64
    mean_acf_mae::Float64
    mean_wasserstein::Float64
    mean_hellinger::Float64
    mean_kurtosis::Float64
    α::Float64
end
```

### 4.8 Dependence Models

Three interchangeable dependence models, all subtypes of `AbstractDependenceModel`.
The user selects one at fit time; all downstream operations (`simulate`, `validate`)
dispatch on the abstract type, so the interface is identical regardless of choice.

| Model | Dependence | Tail Dependence | Scale | Use Case |
|-------|-----------|-----------------|-------|----------|
| `GaussianCopula` | Full n×n | None | Moderate | Simple baseline |
| `StudentTCopula` | Full n×n | Yes (via ν) | Moderate | Realistic portfolios |
| `SingleIndexModel` | Through 1 factor | Inherited from market HMM | Large | 100+ asset universes |

```julia
"""
    GaussianCopula <: AbstractDependenceModel

Gaussian copula for multi-asset dependence, defined by a correlation matrix.
Captures linear dependence but underestimates tail dependence.

A copula is a pure mathematical dependence structure — it has no knowledge
of asset identifiers. Column ordering is managed by PortfolioModel.

## Fields
- `Σ::Matrix{Float64}` — n × n correlation matrix
"""
struct GaussianCopula <: AbstractDependenceModel
    Σ::Matrix{Float64}
end

"""
    StudentTCopula <: AbstractDependenceModel

Student-t copula for multi-asset dependence, with tail dependence
parameter. More realistic than Gaussian for equity portfolios.

A copula is a pure mathematical dependence structure — it has no knowledge
of asset identifiers. Column ordering is managed by PortfolioModel.

## Fields
- `Σ::Matrix{Float64}` — n × n correlation matrix
- `ν::Float64` — degrees of freedom (controls tail dependence)
"""
struct StudentTCopula <: AbstractDependenceModel
    Σ::Matrix{Float64}
    ν::Float64
end

"""
    SingleIndexModel <: AbstractDependenceModel

Single-Index Model dependence: all cross-asset correlation flows through
one market factor. Each asset i has:
    G_i = αᵢ + βᵢ × G_market + εᵢ

Scales well to large universes (2 parameters per asset vs n×n/2 for copulas)
but cannot capture sector-level or tail dependence beyond what the market
factor provides. Reproduces the paper's multi-asset methodology (Section 4).

## Fields
- `α::Vector{Float64}` — per-asset intercepts (length n_assets)
- `β::Vector{Float64}` — per-asset market betas (length n_assets)
- `σ_ε::Vector{Float64}` — per-asset residual standard deviations (length n_assets)
- `market_model::JumpHiddenMarkovModel` — fitted HMM for the market index
"""
struct SingleIndexModel <: AbstractDependenceModel
    α::Vector{Float64}
    β::Vector{Float64}
    σ_ε::Vector{Float64}
    market_model::JumpHiddenMarkovModel
end
```

### 4.9 Portfolio Types

```julia
"""
    PortfolioModel

Multi-asset model: marginal Jump-HMMs coupled by a dependence model.

The dependence model can be a copula (GaussianCopula, StudentTCopula)
or a factor model (SingleIndexModel). The choice affects how cross-asset
correlation is captured, but the simulation and validation interface is
identical regardless.

When using SingleIndexModel: the `marginals` dict does NOT include the
market index — the market model lives inside the SingleIndexModel itself.
The `marginals` dict contains only the non-market assets, keyed by ticker.

## Fields
- `tickers::Vector{String}` — asset identifiers (column order for dependence model)
- `marginals::Dict{String, JumpHiddenMarkovModel}` — per-asset fitted models
- `dependence::AbstractDependenceModel` — dependence structure
"""
struct PortfolioModel
    tickers::Vector{String}
    marginals::Dict{String, JumpHiddenMarkovModel}
    dependence::AbstractDependenceModel
end

"""
    PortfolioSimulationResult

Correlated synthetic paths across multiple assets.

## Fields
- `tickers::Vector{String}` — asset identifiers
- `results::Dict{String, SimulationResult}` — per-asset simulation results
"""
struct PortfolioSimulationResult
    tickers::Vector{String}
    results::Dict{String, SimulationResult}
end
```

---

## 5. Public API

### 5.1 Finance Helpers (`Finance.jl`)

```julia
"""
    excess_growth_rates(prices; rf=0.0, dt=1/252) → Vector{Float64}

Compute annualized excess log-growth rates from a price vector:
    G_t = (1/dt) × ln(P_t / P_{t-1}) - rf

Returns a vector of length (length(prices) - 1).
"""
function excess_growth_rates(prices::AbstractVector{<:Real};
                             rf::Float64=0.0, dt::Float64=1/252)
end

"""
    excess_growth_rates(prices; rf=0.0, dt=1/252) → Matrix{Float64}

Compute excess growth rates from a price matrix (n_obs × n_assets).
Returns a matrix of size (n_obs - 1) × n_assets.
"""
function excess_growth_rates(prices::AbstractMatrix{<:Real};
                             rf::Float64=0.0, dt::Float64=1/252)
end

"""
    excess_growth_rates(data, tickers; rf=0.0, dt=1/252, price_col=:close) → Matrix{Float64}

Compute excess growth rates from a Dict{String, DataFrame} of price data.
Each DataFrame must have a column named `price_col` (default :close).
Returns a matrix of size (n_obs - 1) × length(tickers).
"""
function excess_growth_rates(data::Dict{String, DataFrame},
                             tickers::Vector{String};
                             rf::Float64=0.0, dt::Float64=1/252,
                             price_col::Symbol=:close)
end

"""
    prices_from_growth_rates(G, P0; rf=0.0, dt=1/252) → Vector{Float64}

Reconstruct a price path from excess growth rates and an initial price:
    P_t = P_{t-1} × exp((G_t + rf) × dt)

Returns a vector of length (length(G) + 1) starting with P0.
"""
function prices_from_growth_rates(G::AbstractVector{Float64}, P0::Float64;
                                  rf::Float64=0.0, dt::Float64=1/252)
end
```

### 5.2 State Partition (`Partition.jl`)

```julia
"""
    fit(LaplacePartition, observations; N=100) → LaplacePartition

Fit a Laplace distribution via MLE to the observations, then construct
N equal-probability quantile bins.
"""
function fit(::Type{LaplacePartition}, observations::AbstractVector{Float64};
             N::Int=100)
end

"""
    assign_states(partition, observations) → Vector{Int}

Map continuous observations to discrete state indices.

Convention: observation x is assigned to state k where
    boundaries[k] ≤ x < boundaries[k+1]
The last state (k = N) captures boundaries[N] ≤ x (inclusive upper tail).
"""
function assign_states(partition::LaplacePartition,
                       observations::AbstractVector{Float64})
end
```

### 5.3 Transition Matrix (`Transition.jl`)

```julia
"""
    estimate_transition(states, N) → Matrix{Float64}

Estimate an N×N row-stochastic transition matrix by direct frequency
counting of consecutive state pairs.

If a row has zero counts (state never observed as source), it is filled
with a uniform distribution (1/N) as a fallback.
"""
function estimate_transition(states::AbstractVector{Int}, N::Int)
end

"""
    stationary_distribution(T; power=50) → Vector{Float64}

Compute the stationary distribution π̄ by matrix exponentiation:
    π̄ = (T^power)[1, :]
"""
function stationary_distribution(T::Matrix{Float64}; power::Int=50)
end
```

### 5.4 Emission Fitting (`Emission.jl`)

```julia
"""
    fit_emissions(states, observations, N; ν=5.0, min_obs=2) → Vector{StudentTEmission}

Fit per-state Student-t emission distributions.

For each state k:
- Collect all observations assigned to state k
- If n_obs ≥ min_obs: fit μ_k = mean(obs), σ_k = std(obs)
- If n_obs < min_obs: fallback to global μ = mean(all_obs), σ = std(all_obs),
  and mark is_fallback = true

Arguments:
- `states` — encoded state sequence (same length as observations)
- `observations` — continuous excess growth rates
- `N` — total number of states
- `ν` — Student-t degrees of freedom (default 5.0)
- `min_obs` — minimum observations per state before fallback (default 2)
"""
function fit_emissions(states::AbstractVector{Int},
                       observations::AbstractVector{Float64}, N::Int;
                       ν::Float64=5.0, min_obs::Int=2)
end

"""
    sample_emission(emission) → Float64

Draw a single sample from a StudentTEmission:
    x = μ + σ × rand(TDist(ν))
"""
function sample_emission(e::StudentTEmission)
end
```

### 5.5 Model Fitting (`JumpHMM.jl` / dispatched)

```julia
"""
    fit(JumpHiddenMarkovModel, prices; kwargs...) → JumpHiddenMarkovModel

Fit a Jump-HMM from a price time series. Performs the full pipeline:
1. Compute excess growth rates from prices
2. Fit Laplace distribution → build LaplacePartition with N states
3. Assign observations to states
4. Estimate transition matrix by frequency counting
5. Fit per-state Student-t emissions (with fallback for sparse states)
6. Compute stationary distribution via T^50
7. Initialize jump parameters with ϵ=0, λ=1 (no jumps until tuned)

The returned model has no active jump mechanism. Call `tune` to optimize
the jump parameters.

## Keyword Arguments
- `rf::Float64 = 0.0` — annualized risk-free rate
- `N::Int = 100` — number of discrete states
- `ν::Float64 = 5.0` — Student-t degrees of freedom
- `dt::Float64 = 1/252` — time step (daily data by default)
- `min_obs::Int = 2` — minimum observations per state before fallback
"""
function fit(::Type{JumpHiddenMarkovModel}, prices::AbstractVector{<:Real};
             rf::Float64=0.0, N::Int=100, ν::Float64=5.0,
             dt::Float64=1/252, min_obs::Int=2)
end
```

### 5.6 Jump Tuning (`Tune.jl`)

```julia
"""
    tune(model, prices; kwargs...) → JumpHiddenMarkovModel

Grid search over (ϵ, λ) to find jump parameters that minimize a composite
objective balancing distributional and temporal fidelity:

    J = Σ(acf_obs - acf_sim)² + w_κ × (κ_obs - κ_sim)²

where acf is the autocorrelation of |returns| and κ is excess kurtosis.

Returns a NEW model with optimized JumpParameters. The partition, transition
matrix, and emissions are carried forward unchanged.

Only paths containing at least one jump event contribute to the objective
(matching the paper's methodology).

## Keyword Arguments
- `ϵ_range` — values of ϵ to search (default: range(1e-4, 2.5e-2, length=20))
- `λ_range` — values of λ to search (default: range(10, 160, length=16))
- `n_paths::Int = 200` — number of paths to simulate per grid point
- `n_steps::Int = 0` — simulation length per path (default: length of training data)
- `w_κ::Float64 = 0.20` — kurtosis penalty weight in objective
- `p_neg::Float64 = 0.52` — negative tail bias for jumps
- `N_tail::Int = 5` — number of tail states on each side
- `acf_lags::Int = 25` — number of autocorrelation lags to compare
- `seed::Union{Int, Nothing} = nothing` — random seed for reproducibility
"""
function tune(model::JumpHiddenMarkovModel, prices::AbstractVector{<:Real};
              kwargs...)
end
```

### 5.7 Simulation (`Simulate.jl`)

```julia
"""
    simulate(model, n_steps; kwargs...) → SimulationResult

Generate synthetic paths from a fitted Jump-HMM.

Algorithm (per path):
1. Draw starting state from the stationary distribution (or use `start`)
2. At each time step t:
   a. With probability ϵ: trigger a jump
      - Draw K ~ Poisson(λ) consecutive jump steps
      - Each jump step: land in bottom N_tail states (prob p_neg)
        or top N_tail states (prob 1-p_neg), uniformly within the tail
      - Mark these steps as jump=true
   b. With probability (1-ϵ): normal transition via row of T
      - Mark step as jump=false
3. For each state in the chain, sample an observation from the
   corresponding StudentTEmission

Returns a SimulationResult containing all paths.

## Keyword Arguments
- `n_paths::Int = 1000` — number of paths to generate
- `start::Union{Int, Symbol} = :stationary` — starting state index,
  or :stationary to sample from π̄
- `seed::Union{Int, Nothing} = nothing` — random seed
"""
function simulate(model::JumpHiddenMarkovModel, n_steps::Int;
                  n_paths::Int=1000,
                  start::Union{Int, Symbol}=:stationary,
                  seed::Union{Int, Nothing}=nothing)
end
```

### 5.8 Decoding (`Decode.jl`)

```julia
"""
    decode(model, observations) → Vector{Int}

Viterbi algorithm: find the most likely state sequence given observed
excess growth rates.

Emission likelihood at state k for observation x:
    p(x | k) = (1/σ_k) × pdf(TDist(ν), (x - μ_k) / σ_k)

Transition probabilities come from the model's transition matrix.
Starting probabilities come from the stationary distribution.

Computation is done in log-space for numerical stability.
"""
function decode(model::JumpHiddenMarkovModel,
                observations::AbstractVector{Float64})
end

"""
    forward_filter(model, observations) → Matrix{Float64}

Forward algorithm: compute filtered state probabilities
    P(S_t = k | G_{1:t}) for each time step.

Returns an (n_steps × N) matrix where each row sums to 1.

Note: Named `forward_filter` (not `filter`) to avoid shadowing Base.filter.
"""
function forward_filter(model::JumpHiddenMarkovModel,
                        observations::AbstractVector{Float64})
end

"""
    log_likelihood(model, observations) → Float64

Compute the log-likelihood of an observation sequence under the model,
using the forward algorithm.
"""
function log_likelihood(model::JumpHiddenMarkovModel,
                        observations::AbstractVector{Float64})
end
```

### 5.9 Validation (`Validate.jl`)

```julia
"""
    validate(model, prices; kwargs...) → ValidationReport

Statistical validation of the model against empirical data.

Procedure:
1. Compute empirical excess growth rates from prices
2. Simulate n_paths synthetic paths of the same length
3. For each path, compute:
   - KS two-sample test p-value (vs empirical)
   - Anderson-Darling k-sample test p-value
   - Wasserstein-1 distance
   - Hellinger distance
   - ACF-MAE: mean |acf_obs(|G|) - acf_sim(|G|)| over specified lags
   - Excess kurtosis
4. Aggregate into pass rates and means

## Keyword Arguments
- `n_paths::Int = 1000` — number of Monte Carlo paths
- `α::Float64 = 0.05` — significance level for KS/AD pass/fail
- `acf_lags::Int = 25` — number of lags for ACF comparison
- `rf::Float64 = model.rf` — risk-free rate (defaults to model's fitted value)
- `seed::Union{Int, Nothing} = nothing`
"""
function validate(model::JumpHiddenMarkovModel,
                  prices::AbstractVector{<:Real}; kwargs...)
end
```

### 5.10 Dependence Model Fitting (`Copula.jl`, `SIM.jl`)

```julia
"""
    fit(GaussianCopula, returns) → GaussianCopula

Fit a Gaussian copula from an (n_obs × n_assets) matrix of excess growth rates.
Estimates the Pearson correlation matrix of the probability-integral-transformed
(PIT) marginals. Column ordering is the caller's responsibility — the copula
is a pure dependence structure with no knowledge of asset identifiers.
"""
function fit(::Type{GaussianCopula}, returns::AbstractMatrix{Float64})
end

"""
    fit(StudentTCopula, returns) → StudentTCopula

Fit a Student-t copula from an (n_obs × n_assets) matrix of excess growth rates.
Estimates both the correlation matrix and degrees of freedom ν via maximum
likelihood on the PIT marginals. Column ordering is the caller's responsibility.
"""
function fit(::Type{StudentTCopula}, returns::AbstractMatrix{Float64})
end

"""
    fit(SingleIndexModel, returns, market_returns; kwargs...) → SingleIndexModel

Fit a Single-Index Model from asset returns and a market factor.
For each asset i, estimates αᵢ, βᵢ, σ_εᵢ via OLS regression:
    G_i = αᵢ + βᵢ × G_market + εᵢ

Also fits a JumpHiddenMarkovModel to the market returns.

## Keyword Arguments
- `rf::Float64 = 0.0` — risk-free rate for market model fitting
- `N::Int = 100` — states for market model
- `ν::Float64 = 5.0` — Student-t df for market model
"""
function fit(::Type{SingleIndexModel}, returns::AbstractMatrix{Float64},
             market_returns::AbstractVector{Float64}; kwargs...)
end

"""
    sample_dependence(dep, n_samples) → Matrix{Float64}

Draw n_samples from the dependence model.

For copulas: returns an (n_samples × n_assets) matrix of uniform marginals
in [0, 1] with the specified dependence structure.

For SingleIndexModel: returns an (n_samples × n_assets) matrix of
asset returns generated via G_i = αᵢ + βᵢ × G_market_sim + εᵢ,
where G_market_sim comes from simulating the market HMM.
"""
function sample_dependence(dep::AbstractDependenceModel, n_samples::Int)
end
```

### 5.11 Portfolio (`Portfolio.jl`)

```julia
"""
    fit(PortfolioModel, tickers, prices; kwargs...) → PortfolioModel

Fit a multi-asset model. The dependence keyword selects the strategy:

- `StudentTCopula` (default): fit independent marginal HMMs per asset,
  then fit a Student-t copula from the joint returns
- `GaussianCopula`: same, but with a Gaussian copula
- `SingleIndexModel`: fit a market HMM, then regress each asset on the
  market factor to get (α, β, σ_ε) per asset. Requires `market` keyword.

## Keyword Arguments
- `dependence::Type{<:AbstractDependenceModel} = StudentTCopula`
- `market::String = ""` — market ticker (required for SingleIndexModel,
  must be a column in the prices matrix)
- `rf::Float64 = 0.0`
- `N::Int = 100`
- `ν::Float64 = 5.0`
- `dt::Float64 = 1/252`
"""
function fit(::Type{PortfolioModel}, tickers::Vector{String},
             prices::AbstractMatrix{<:Real};
             dependence::Type{<:AbstractDependenceModel}=StudentTCopula,
             market::String="",
             rf::Float64=0.0, N::Int=100, ν::Float64=5.0,
             dt::Float64=1/252)
end

"""
    tune(portfolio, prices; kwargs...) → PortfolioModel

Tune jump parameters for each marginal model independently.
Returns a new PortfolioModel with tuned marginals and the same dependence model.

For SingleIndexModel: also tunes the market model's jump parameters.
"""
function tune(portfolio::PortfolioModel, prices::AbstractMatrix{<:Real};
              kwargs...)
end

"""
    simulate(portfolio, n_steps; kwargs...) → PortfolioSimulationResult

Generate correlated synthetic paths across all assets.

Algorithm depends on the dependence model:

**Copula (Gaussian or Student-t):**
1. Simulate each marginal HMM independently (state chains + emissions)
2. Use the copula to reorder/correlate the observation-level samples
   across assets, preserving each marginal's dynamics

**SingleIndexModel:**
1. Simulate the market HMM to get a market return path
2. For each asset i: G_i = αᵢ + βᵢ × G_market + εᵢ where εᵢ ~ N(0, σ_εᵢ)

## Keyword Arguments
- `n_paths::Int = 1000`
- `seed::Union{Int, Nothing} = nothing`
"""
function simulate(portfolio::PortfolioModel, n_steps::Int; kwargs...)
end

"""
    validate(portfolio, prices; kwargs...) → Dict{String, ValidationReport}

Validate each marginal model against its empirical data.
Returns a dict mapping ticker → ValidationReport.
"""
function validate(portfolio::PortfolioModel,
                  prices::AbstractMatrix{<:Real}; kwargs...)
end
```

---

## 6. End-to-End Usage Examples

### 6.1 Single Asset

```julia
using JumpHMM

# Load price data (user provides this)
prices = [...]  # Vector{Float64} of daily closing prices

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

# Step 5: Decode a state sequence from new data
new_prices = [...]
G_new = excess_growth_rates(new_prices; rf=0.05)
states = decode(model, G_new)

# Step 6: Convert simulated growth rates back to prices
path = result.paths[1]
synthetic_prices = prices_from_growth_rates(path.observations, prices[end]; rf=0.05)
```

### 6.2 Multi-Asset Portfolio (Copula Approach)

```julia
using JumpHMM

tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
price_matrix = [...]  # (n_obs × 5) matrix, columns match ticker order

# Fit marginals + Student-t copula (default)
portfolio = fit(PortfolioModel, tickers, price_matrix;
    dependence=StudentTCopula, rf=0.05, N=100)

# Tune jump params for each asset
portfolio = tune(portfolio, price_matrix)

# Simulate correlated paths
port_result = simulate(portfolio, 252; n_paths=1000)

# Access per-asset results
aapl_paths = port_result.results["AAPL"]

# Validate
reports = validate(portfolio, price_matrix)
for (ticker, report) in reports
    println("$ticker: KS=$(report.ks_pass_rate), ACF-MAE=$(report.mean_acf_mae)")
end
```

### 6.3 Multi-Asset Portfolio (Single-Index Model)

```julia
using JumpHMM

# Large universe — SIM scales better than copulas here
tickers = ["SPY", "AAPL", "MSFT", ...]  # 424 S&P 500 constituents
price_matrix = [...]  # (n_obs × 424) matrix

# Fit with SIM — SPY is the market factor
portfolio = fit(PortfolioModel, tickers, price_matrix;
    dependence=SingleIndexModel, market="SPY", rf=0.05)

# Same interface from here on
portfolio = tune(portfolio, price_matrix)
port_result = simulate(portfolio, 252; n_paths=1000)
reports = validate(portfolio, price_matrix)
```

### 6.4 Comparing Dependence Models

```julia
using JumpHMM

tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
price_matrix = [...]

# Fit both approaches
portfolio_copula = fit(PortfolioModel, tickers, price_matrix;
    dependence=StudentTCopula, rf=0.05)
portfolio_sim = fit(PortfolioModel, tickers, price_matrix;
    dependence=SingleIndexModel, market="SPY", rf=0.05)

# Tune and validate both
portfolio_copula = tune(portfolio_copula, price_matrix)
portfolio_sim = tune(portfolio_sim, price_matrix)

reports_copula = validate(portfolio_copula, price_matrix)
reports_sim = validate(portfolio_sim, price_matrix)

# Compare — which captures cross-asset dependence better?
for ticker in tickers
    println("$ticker: Copula KS=$(reports_copula[ticker].ks_pass_rate) " *
            "vs SIM KS=$(reports_sim[ticker].ks_pass_rate)")
end
```

---

## 7. Algorithm Details

### 7.1 Simulation Algorithm (matches paper Algorithm 1)

```
Input: model M = (partition, T, emissions, π̄, jump=(ϵ, λ, p_neg, N_tail))
Input: n_steps, starting state s₁

Initialize: t = 1, chain = [s₁]
While t ≤ n_steps:
    If rand() < ϵ:                          # Jump event
        K ~ Poisson(λ)                      # Jump duration
        For k = 1 to K (while t ≤ n_steps):
            If rand() < p_neg:
                s_t ~ Uniform({1, ..., N_tail})        # Bottom tail
            Else:
                s_t ~ Uniform({N-N_tail+1, ..., N})    # Top tail
            chain[t] = s_t, jumps[t] = true
            t += 1
    Else:                                   # Normal transition
        s_t ~ Categorical(T[s_{t-1}, :])
        chain[t] = s_t, jumps[t] = false
        t += 1

For each t:
    G_t ~ μ_{s_t} + σ_{s_t} × TDist(ν)    # Sample emission

Return (chain, observations, jumps)
```

### 7.2 Tuning Objective (matches paper Section 3.3)

```
For each (ϵ, λ) in grid:
    Simulate n_paths paths of length n_steps
    Keep only paths with ≥ 1 jump event
    For each kept path p:
        acf_sim_p = autocor(|G_sim_p|, lags=1:L)
        κ_sim_p = kurtosis(G_sim_p)

    J(ϵ, λ) = mean_p[ Σ_ℓ (acf_obs[ℓ] - acf_sim_p[ℓ])² + w_κ × (κ_obs - κ_sim_p)² ]

Return (ϵ*, λ*) = argmin J
```

### 7.3 Copula Simulation for Portfolios

```
1. Draw U ~ Copula(Σ, ν):  (n_steps × n_assets) matrix in [0,1]
2. For each asset i:
   a. Fit marginal model M_i independently
   b. Simulate state chain from M_i (with jumps)
   c. For observation generation: use copula rank to reorder
      the temporal emission samples, preserving cross-asset
      dependence while maintaining each marginal's dynamics
```

---

## 8. Constants and Defaults

| Parameter | Default | Paper Reference |
|-----------|---------|-----------------|
| N (states) | 100 | Section 3.1, Table 3 |
| ν (Student-t df) | 5.0 | Section 3.2, Figure S2 |
| ϵ (jump prob) | 1e-4 | Table 1 (SPY optimal) |
| λ (jump duration) | 100 | Table 1 (SPY optimal) |
| p_neg (negative bias) | 0.52 | Section 3.3 |
| N_tail (tail states) | 5 | Algorithm 1 |
| dt (time step) | 1/252 | Daily trading data |
| w_κ (kurtosis weight) | 0.20 | Section 3.3 |
| T^power for π̄ | 50 | Section 3.1 |
| min_obs for fallback | 2 | Empirical choice |
| α (significance) | 0.05 | Standard |
