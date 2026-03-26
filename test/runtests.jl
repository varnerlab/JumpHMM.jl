using JumpHMM
using Test
using Random
using Statistics
using Distributions
using LinearAlgebra

# resolve ambiguity: JumpHMM.fit vs Distributions.fit
const fit = JumpHMM.fit

include("test_finance.jl")
include("test_partition.jl")
include("test_transition.jl")
include("test_emission.jl")
include("test_simulate.jl")
include("test_decode.jl")
include("test_tune.jl")
include("test_validate.jl")
include("test_copula.jl")
include("test_sim.jl")
include("test_portfolio.jl")
