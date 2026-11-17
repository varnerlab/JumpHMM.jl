@testset "portfolio models" begin
    P = inputs["prices"][:, 1:6]
    tickers = inputs["tickers"][1:6]
    sectors = inputs["sectors"][1:5]
    T = JumpHMM.MyPortfolioModel
    base = (tickers = tickers, prices = P, number_of_states = 20, ν = 5.0)

    # a single-index portfolio simulates what its dependence model samples, with states 0 and no episodes -
    si = JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySingleIndexDependenceModel, market = "SPY")))
    @test si.tickers == tickers[2:6]
    @test si.market_ticker == "SPY"
    @test si.tickers_map["SPY"] == 1
    paths = JumpHMM.simulate(si, 50; number_of_paths = 2, seed = 21)
    Random.seed!(21)
    first_path = JumpHMM.sample_dependence(si.dependence, 50)
    second_path = JumpHMM.sample_dependence(si.dependence, 50)
    for (j, ticker) ∈ enumerate(si.tickers)
        @test paths[ticker].growth_rates == hcat(first_path[:, j], second_path[:, j])
        @test paths[ticker].states == zeros(Int64, 50, 2)
        @test paths[ticker].episodes == fill(false, 50, 2)
    end

    # tuning a single-index portfolio tunes the market model on the market growth rates -
    ϵ_range = [1e-4, 3e-3]
    λ_range = [10.0, 100.0]
    tuned = JumpHMM.tune(si, P; ϵ_range = ϵ_range, λ_range = λ_range, number_of_paths = 5, seed = 22)
    market_growth_rates = JumpHMM.excess_growth_rates(P[:, 1]; rf = si.rf, Δt = si.Δt)
    expected = JumpHMM.tune(si.dependence.market_model, market_growth_rates; ϵ_range = ϵ_range, λ_range = λ_range,
        number_of_paths = 5, seed = 22)
    @test tuned.dependence.market_model.ϵ == expected.ϵ
    @test tuned.dependence.market_model.λ == expected.λ
    @test si.dependence.market_model.ϵ == 0.0

    # a hybrid portfolio has one day fewer than the number of steps -
    hy = JumpHMM.build(T, merge(base, (dependence = JumpHMM.MyHybridSingleIndexDependenceModel, market = "SPY")))
    paths = JumpHMM.simulate(hy, 50; number_of_paths = 2, seed = 23)
    for ticker ∈ hy.tickers
        @test size(paths[ticker].growth_rates) == (49, 2)
        @test size(paths[ticker].states) == (49, 2)
        @test size(paths[ticker].episodes) == (49, 2)
    end

    # bad inputs are rejected -
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySingleIndexDependenceModel,)))
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = JumpHMM.MyHybridSingleIndexDependenceModel,)))
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySectorFactorModel, sectors = sectors)))
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySectorFactorModel, market = "SPY")))
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = JumpHMM.MyGaussianCopulaModel, tickers = tickers[1:5])))
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySingleIndexDependenceModel, market = "XYZ")))
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = Float64,)))
end
