module JumpHMM

# dependencies
using Distributions
using Statistics
using StatsBase
using LinearAlgebra
import PDMats
using HypothesisTests
using DataFrames
using Random
import SpecialFunctions: loggamma

# source files (order matters: Types first, then leaves, then orchestrators)
include("Types.jl")
include("Finance.jl")
include("Partition.jl")
include("Transition.jl")
include("Emission.jl")
include("Simulate.jl")
include("Decode.jl")
include("Tune.jl")
include("Validate.jl")
include("Copula.jl")
include("Vine.jl")
include("SIM.jl")
include("Portfolio.jl")

# --- Top-level fit orchestrator --------------------------------------------

"""
    fit(::Type{JumpHiddenMarkovModel}, prices; kwargs...) → JumpHiddenMarkovModel

Fit a Jump-HMM from a price time series. Returns a model with no active
jump mechanism (ϵ=0). Call `tune` to optimize jump parameters.
"""
function fit(::Type{JumpHiddenMarkovModel}, prices::AbstractVector{<:Real};
             rf::Float64=0.0, N::Int=100, ν::Float64=5.0,
             dt::Float64=1/252, min_obs::Int=2)

    G = excess_growth_rates(prices; rf=rf, dt=dt)
    partition = fit(LaplacePartition, G; N=N)
    states = assign_states(partition, G)
    T = estimate_transition(states, N)
    emissions = fit_emissions(states, G, N; ν=ν, min_obs=min_obs)
    π̄ = stationary_distribution(T)
    jump = JumpParameters(0.0, 1.0)  # no jumps until tuned

    return JumpHiddenMarkovModel(partition, T, emissions, π̄, jump, ν, rf, dt)
end

# --- Exports ---------------------------------------------------------------

# abstract types
export AbstractMarkovModel, AbstractDependenceModel, AbstractValidationResult

# concrete types
export LaplacePartition, StudentTEmission, JumpParameters
export JumpHiddenMarkovModel
export SimulationPath, SimulationResult
export PathTestResult, ValidationReport
export GaussianCopula, StudentTCopula, SingleIndexModel, HybridSingleIndexModel
export AbstractBivariateCopula, GaussianBiCopula, StudentTBiCopula
export ClaytonBiCopula, GumbelBiCopula, FrankBiCopula
export VineEdge, VineCopula
export PortfolioModel, PortfolioSimulationResult

# functions
export fit, assign_states
export estimate_transition, stationary_distribution
export fit_emissions, sample_emission
export simulate
export decode, forward_filter, log_likelihood
export tune
export validate
export excess_growth_rates, prices_from_growth_rates
export sample_dependence

end # module
