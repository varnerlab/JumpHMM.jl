# a copula type that the package does not define: it wraps the Gaussian copula, so that a hybrid model can use it -
mutable struct WrappedGaussianCopulaModel <: JumpHMM.AbstractDependenceModel
    inner::JumpHMM.MyGaussianCopulaModel
    WrappedGaussianCopulaModel() = new()
end

"""
    JumpHMM.build(model::Type{WrappedGaussianCopulaModel}, data::NamedTuple) -> WrappedGaussianCopulaModel

Build the wrapped Gaussian copula from `data.growth_rates`.
"""
function JumpHMM.build(model::Type{WrappedGaussianCopulaModel}, data::NamedTuple)::WrappedGaussianCopulaModel
    wrapped = WrappedGaussianCopulaModel()
    wrapped.inner = JumpHMM.build(JumpHMM.MyGaussianCopulaModel, data)
    return wrapped
end

"""
    JumpHMM.sample_dependence(model::WrappedGaussianCopulaModel, number_of_steps::Int64) -> Array{Float64,2}

Draw uniform numbers from the wrapped Gaussian copula.
"""
function JumpHMM.sample_dependence(model::WrappedGaussianCopulaModel, number_of_steps::Int64)::Array{Float64,2}
    return JumpHMM.sample_dependence(model.inner, number_of_steps)
end

@testset "single-index models" begin
    P = inputs["prices"]
    GA = JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252)
    data = (growth_rates = GA[:, 2:6], market_growth_rates = GA[:, 1], number_of_states = 10, ν = 5.0)

    # the samples have the right shapes -
    sim = JumpHMM.build(JumpHMM.MySingleIndexDependenceModel, data)
    @test size(JumpHMM.sample_dependence(sim, 50)) == (50, 5)
    h = JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel, (
        tickers = inputs["tickers"][1:6], prices = P[:, 1:6], market = "SPY", number_of_states = 10, ν = 5.0))
    @test size(JumpHMM.sample_dependence(h, 50)) == (49, 5)

    # bad inputs are rejected -
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MySingleIndexDependenceModel, merge(data, (residual_method = :other,)))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MySingleIndexDependenceModel,
        merge(data, (market_growth_rates = GA[1:100, 1],)))

    # without market prices, the market model is built from prices rebuilt from the market growth rates -
    rebuilt = JumpHMM.prices_from_growth_rates(GA[:, 1], 100.0; rf = 0.0, Δt = 1/252)
    expected = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (
        growth_rates = JumpHMM.excess_growth_rates(rebuilt; rf = 0.0, Δt = 1/252), number_of_states = 10, ν = 5.0))
    @test sim.market_model.transition == expected.transition
    @test sim.market_model.emission_μ == expected.emission_μ
    @test sim.market_model.boundaries == expected.boundaries

    # the hybrid build rejects a missing market ticker, a tickers/columns mismatch and a constant market series -
    hybrid = (tickers = inputs["tickers"][1:6], prices = P[:, 1:6], market = "SPY", number_of_states = 10, ν = 5.0)
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel, merge(hybrid, (market = "XYZ",)))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel,
        merge(hybrid, (tickers = inputs["tickers"][1:5],)))
    constant_market = copy(P[:, 1:6])
    constant_market[:, 1] .= 100.0
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel,
        merge(hybrid, (prices = constant_market,)))

    # a hybrid build with a Gaussian copula runs and samples -
    gaussian = JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel, merge(hybrid, (copula = JumpHMM.MyGaussianCopulaModel,)))
    @test gaussian.copula isa JumpHMM.MyGaussianCopulaModel
    @test size(JumpHMM.sample_dependence(gaussian, 50)) == (49, 5)

    # the hybrid model takes any copula type that is a dependence model, and rejects an instance or another type -
    hybrid_data = (tickers = inputs["tickers"][1:6], prices = P[:, 1:6], market = "SPY", number_of_states = 10, ν = 5.0)
    h_gaussian = JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel, merge(hybrid_data, (copula = JumpHMM.MyGaussianCopulaModel,)))
    h_wrapped = JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel, merge(hybrid_data, (copula = WrappedGaussianCopulaModel,)))
    @test h_wrapped.copula isa WrappedGaussianCopulaModel
    Random.seed!(3)
    sample_wrapped = JumpHMM.sample_dependence(h_wrapped, 50)
    Random.seed!(3)
    sample_gaussian = JumpHMM.sample_dependence(h_gaussian, 50)
    @test sample_wrapped == sample_gaussian
    for bad_copula ∈ [JumpHMM.MyGaussianCopulaModel(), Float64]
        err = try
            JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel, merge(hybrid_data, (copula = bad_copula,)))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        if err isa ArgumentError
            @test occursin("copula must be a type that is a subtype of AbstractDependenceModel", err.msg)
        end
    end
end
