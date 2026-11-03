# --- Abstract Types -------------------------------------------------------
"""Supertype for all Markov models in the package."""
abstract type AbstractMarkovModel end

"""Supertype for copula-based and factor-based dependence models."""
abstract type AbstractDependenceModel end

"""Supertype for validation report types."""
abstract type AbstractValidationResult end

# --- State Partition -------------------------------------------------------
"""
    LaplacePartition

Discretization of continuous observations into N states using equal-probability
quantile bins of a fitted Laplace distribution.
"""
struct LaplacePartition
    μ::Float64
    b::Float64
    N::Int
    boundaries::Vector{Float64}

    function LaplacePartition(μ::Float64, b::Float64, N::Int)
        @assert N ≥ 1 "N must be ≥ 1"
        @assert b > 0.0 "b must be positive"
        cutoffs = range(0.0, 1.0, length=N + 1) |> collect
        d = Laplace(μ, b)
        boundaries = [quantile(d, p) for p in cutoffs]
        boundaries[1] = -Inf
        boundaries[end] = Inf
        return new(μ, b, N, boundaries)
    end
end

# --- Emission Distributions ------------------------------------------------
"""
    StudentTEmission

Per-state emission: location-scale Student-t.
    G_t | S_t = k  ~  μ_k + σ_k × t_ν
"""
struct StudentTEmission
    μ::Float64
    σ::Float64
    ν::Float64
    n_obs::Int
    is_fallback::Bool
end

# --- Jump Parameters -------------------------------------------------------
"""
    JumpParameters

Poisson-driven jump mechanism that forces the chain into tail states.
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

# --- The Model -------------------------------------------------------------
"""
    JumpHiddenMarkovModel <: AbstractMarkovModel

Fitted single-asset Jump-HMM. Immutable — fitting/tuning return new instances.
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

# --- Simulation Output -----------------------------------------------------
"""
    SimulationPath

A single simulated path: states, observations, and jump indicators.
"""
struct SimulationPath
    states::Vector{Int}
    observations::Vector{Float64}
    jumps::Vector{Bool}
end

"""
    SimulationResult

Collection of simulated paths.
"""
struct SimulationResult
    paths::Vector{SimulationPath}
end

# --- Validation Types -------------------------------------------------------
"""
    PathTestResult

Statistical test results for a single simulated path vs empirical data.
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

Aggregate validation report across Monte Carlo paths.
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

# --- Dependence Models ------------------------------------------------------
"""
    GaussianCopula <: AbstractDependenceModel

Gaussian copula defined by a correlation matrix.
"""
struct GaussianCopula <: AbstractDependenceModel
    Σ::Matrix{Float64}
end

"""
    StudentTCopula <: AbstractDependenceModel

Student-t copula with tail dependence.
"""
struct StudentTCopula <: AbstractDependenceModel
    Σ::Matrix{Float64}
    ν::Float64
end

# --- Bivariate Copula Families (for Vine Copulas) --------------------------
"""Supertype for bivariate copula families used in vine copula construction."""
abstract type AbstractBivariateCopula end

"""Gaussian bivariate copula with correlation parameter ρ ∈ (-1, 1)."""
struct GaussianBiCopula <: AbstractBivariateCopula
    ρ::Float64
end

"""Student-t bivariate copula with correlation ρ and degrees of freedom ν."""
struct StudentTBiCopula <: AbstractBivariateCopula
    ρ::Float64
    ν::Float64
end

"""Clayton bivariate copula with parameter θ > 0. Lower tail dependence."""
struct ClaytonBiCopula <: AbstractBivariateCopula
    θ::Float64
end

"""Gumbel bivariate copula with parameter θ ≥ 1. Upper tail dependence."""
struct GumbelBiCopula <: AbstractBivariateCopula
    θ::Float64
end

"""Frank bivariate copula with parameter θ ≠ 0. No tail dependence, symmetric."""
struct FrankBiCopula <: AbstractBivariateCopula
    θ::Float64
end

"""
    VineEdge

One edge in a vine copula tree, representing a conditional bivariate copula.
"""
struct VineEdge
    var1::Int
    var2::Int
    conditioned::Vector{Int}
    copula::AbstractBivariateCopula
end

"""
    VineCopula <: AbstractDependenceModel

C-vine copula: each pair of assets gets its own bivariate copula family,
allowing heterogeneous dependence (e.g., heavy-tailed for tech pairs,
asymmetric for energy-finance). Automatic family selection via AIC.
"""
struct VineCopula <: AbstractDependenceModel
    d::Int
    order::Vector{Int}
    edges::Vector{Vector{VineEdge}}
end

"""
    SingleIndexModel <: AbstractDependenceModel

Single-Index Model: all cross-asset correlation flows through one market factor.
    G_i = αᵢ + βᵢ × G_market + εᵢ
"""
struct SingleIndexModel <: AbstractDependenceModel
    α::Vector{Float64}
    β::Vector{Float64}
    σ_ε::Vector{Float64}
    residuals::Matrix{Float64}
    residual_method::Symbol
    market_model::JumpHiddenMarkovModel
end

"""
    HybridSingleIndexModel <: AbstractDependenceModel

Variance-corrected Single-Index Model with per-ticker HMM marginals and copula
dependence. Combines:
- SIM factor structure: `G_i = αᵢ + βᵢ × G_market + εᵢ`
- Per-ticker HMM marginals for idiosyncratic draws (preserves tails, regimes)
- Variance correction so `Var[G_i] ≈ Var[marginal_i]` (marginal-preserving branch)
  or exact R² recovery (R²-preserving branch for high-R² ETFs)
- β clipping with an idiosyncratic floor to prevent degenerate paths
- Copula rank-reordering to preserve cross-sectional dependence

See `sample_dependence` for the full construction recipe.
"""
struct HybridSingleIndexModel <: AbstractDependenceModel
    α::Vector{Float64}                              # per-ticker intercepts (Jensen's alpha)
    β::Vector{Float64}                              # per-ticker calibrated betas
    r²::Vector{Float64}                             # per-ticker R² from OLS calibration
    σ_market::Float64                               # market growth-rate std (annualized)
    marginals::Dict{String,JumpHiddenMarkovModel}   # per-ticker HMM marginals
    copula::AbstractDependenceModel                 # fitted on OLS residuals
    market_model::JumpHiddenMarkovModel             # fitted market HMM
    tickers::Vector{String}                         # ordered ticker list (indexes α, β, r²)
    r2_preserve_threshold::Float64                  # R² cutoff for R²-preserving branch
    idiosyncratic_floor::Float64                    # minimum idiosyncratic variance share (f)
end

"""
    SectorFactorModel <: AbstractDependenceModel

Multi-asset composition with one market model, one model per sector, and one model per name:
    g_i(t) = α_i + β_i(t)·g_m(t) + δ_i·f_s(t) + s_i·ε̃_i(t)
Names are linked only through the shared market draw `g_m` and sector draw `f_s` on the same day
(no copula). α and β come from the single-index fit; β is fixed unless `β_scale` is nonzero, in which
case it drifts as an exponential moving average of normal shocks with half-life `β_halflife`.
See `fit(SectorFactorModel, ...)` and `sample_dependence(::SectorFactorModel, n; seed)`.
"""
struct SectorFactorModel <: AbstractDependenceModel
    tickers::Vector{String}                       # names, market excluded, in fit order
    sector::Vector{String}                        # one label per name
    sectors::Vector{String}                       # sorted unique labels
    sector_index::Vector{Int}                     # position of each name's sector in `sectors`
    α::Vector{Float64}                            # intercepts used (after the optional pull), yr⁻¹
    β::Vector{Float64}                            # single-index β̂
    δ::Vector{Float64}                            # sector loadings
    r²::Vector{Float64}                           # real R² of the single-index fit
    σ²_m::Float64                                 # variance of the market growth, yr⁻²
    σ²_s::Vector{Float64}                         # variance of each sector factor, yr⁻²
    σ²_gen::Vector{Float64}                       # variance of each name's growth, yr⁻²
    ε_center::Vector{Float64}                     # long-run mean of each name model, yr⁻¹
    β_halflife::Float64                           # trading days
    β_scale::Vector{Float64}                      # per-name drift scale; zeros = fixed β
    floor::Float64                                # smallest own-noise share of a name's variance
    r2_threshold::Float64                         # R² at or above which a name keeps its real R²
    market_model::JumpHiddenMarkovModel
    sector_models::Vector{JumpHiddenMarkovModel}  # one per `sectors`
    name_models::Vector{JumpHiddenMarkovModel}    # one per `tickers`
    sector_levels::Matrix{Float64}                # compounded sector-factor series (price-like), for tune
end

# --- Portfolio Types --------------------------------------------------------
"""
    PortfolioModel

Multi-asset model: marginal Jump-HMMs coupled by a dependence model.

The `tickers_map` field stores the original ticker → column index mapping from
the price matrix used during `fit`. For `SingleIndexModel` portfolios this includes
the market ticker (stored in `market_ticker`), even though it is excluded from
`tickers` and `marginals`.
"""
struct PortfolioModel
    tickers::Vector{String}
    marginals::Dict{String,JumpHiddenMarkovModel}
    dependence::AbstractDependenceModel
    tickers_map::Dict{String,Int}
    market_ticker::String
end

"""
    PortfolioSimulationResult

Correlated synthetic paths across multiple assets.
"""
struct PortfolioSimulationResult
    tickers::Vector{String}
    results::Dict{String,SimulationResult}
end
