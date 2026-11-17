# load external dependencies -
using Distributions
using Statistics
using StatsBase
using LinearAlgebra
using HypothesisTests
using DataFrames
using Random

# load my codes -
include(joinpath(@__DIR__, "Types.jl"));
include(joinpath(@__DIR__, "Finance.jl"));
include(joinpath(@__DIR__, "Estimate.jl"));
include(joinpath(@__DIR__, "Factory.jl"));
