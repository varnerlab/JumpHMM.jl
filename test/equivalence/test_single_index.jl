@testset "single-index models equal 0.4.1" begin
    P = inputs["prices"]
    GA = JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252)

    # the single-index model, with both residual methods -
    for method in [:bootstrap, :gaussian]
        prefix = "sim/" * String(method)
        sim = JumpHMM.build(JumpHMM.MySingleIndexDependenceModel, (
            growth_rates = GA[:, 2:13], market_growth_rates = GA[:, 1], market_prices = P[:, 1],
            number_of_states = 20, ν = 5.0, residual_method = method))
        @test sim.α == reference[prefix * "/α"]
        @test sim.β == reference[prefix * "/β"]
        @test sim.σ_ε == reference[prefix * "/σ_ε"]
        Random.seed!(12)
        @test JumpHMM.sample_dependence(sim, 200) == reference[prefix * "/sample"]
    end

    # the hybrid single-index model -
    h = JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel, (
        tickers = inputs["tickers"], prices = P, market = "SPY", number_of_states = 20, ν = 5.0))
    @test h.α == reference["hybrid/α"]
    @test h.β == reference["hybrid/β"]
    @test h.r² == reference["hybrid/r²"]
    @test h.σ_market == reference["hybrid/σ_market"]
    Random.seed!(13)
    @test JumpHMM.sample_dependence(h, 200) == reference["hybrid/sample"]
end
