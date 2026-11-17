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
end
