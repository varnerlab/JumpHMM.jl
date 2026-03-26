# --- Abstract Types -------------------------------------------------------
abstract type AbstractMarkovModel end
abstract type AbstractDependenceModel end
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

"""
    SingleIndexModel <: AbstractDependenceModel

Single-Index Model: all cross-asset correlation flows through one market factor.
    G_i = αᵢ + βᵢ × G_market + εᵢ
"""
struct SingleIndexModel <: AbstractDependenceModel
    α::Vector{Float64}
    β::Vector{Float64}
    σ_ε::Vector{Float64}
    market_model::JumpHiddenMarkovModel
end

# --- Portfolio Types --------------------------------------------------------
"""
    PortfolioModel

Multi-asset model: marginal Jump-HMMs coupled by a dependence model.
"""
struct PortfolioModel
    tickers::Vector{String}
    marginals::Dict{String,JumpHiddenMarkovModel}
    dependence::AbstractDependenceModel
end

"""
    PortfolioSimulationResult

Correlated synthetic paths across multiple assets.
"""
struct PortfolioSimulationResult
    tickers::Vector{String}
    results::Dict{String,SimulationResult}
end
