# a dependence model that the portfolio does not know -
struct UnknownDependenceModel <: JumpHMM.AbstractDependenceModel end

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

    # a copula portfolio stores an empty market ticker, even when a market is given -
    gc = JumpHMM.build(T, merge(base, (dependence = JumpHMM.MyGaussianCopulaModel, market = "SPY")))
    @test gc.market_ticker == ""

    # tuning with an explicit market column gives the same market model as the stored map -
    explicit = JumpHMM.tune(si, P; market_col = 1, ϵ_range = ϵ_range, λ_range = λ_range, number_of_paths = 5, seed = 22)
    @test explicit.dependence.market_model.ϵ == tuned.dependence.market_model.ϵ
    @test explicit.dependence.market_model.λ == tuned.dependence.market_model.λ

    # without a market column the single-index portfolio keeps its market model -
    asset_map = Dict{String,Int64}()
    for j ∈ 2:6
        asset_map[tickers[j]] = j
    end
    kept = JumpHMM.tune(si, P; tickers_map = asset_map, ϵ_range = ϵ_range, λ_range = λ_range, number_of_paths = 5, seed = 22)
    @test kept.dependence.market_model === si.dependence.market_model

    # tuning a hybrid portfolio tunes its market model like the market model on its own, and the marginals are shared -
    tuned_hybrid = JumpHMM.tune(hy, P; ϵ_range = ϵ_range, λ_range = λ_range, number_of_paths = 5, seed = 24)
    expected_market = JumpHMM.tune(hy.dependence.market_model, market_growth_rates; ϵ_range = ϵ_range, λ_range = λ_range,
        number_of_paths = 5, seed = 24)
    @test tuned_hybrid.dependence.market_model.ϵ == expected_market.ϵ
    @test tuned_hybrid.dependence.market_model.λ == expected_market.λ
    for ticker ∈ hy.tickers
        @test hy.marginals[ticker] === hy.dependence.marginals[ticker]
        @test tuned_hybrid.marginals[ticker] === tuned_hybrid.dependence.marginals[ticker]
    end

    # a sector factor portfolio validates each own model on its level series -
    sp = JumpHMM.build(T, (tickers = inputs["tickers"], prices = inputs["prices"], number_of_states = 20, ν = 5.0,
        dependence = JumpHMM.MySectorFactorModel, market = "SPY", sectors = inputs["sectors"],
        sector_options = (center_steps = 20_000,)))
    reports = JumpHMM.validate(sp, inputs["prices"]; number_of_paths = 5, seed = 3)
    dep = sp.dependence
    @test length(reports) == length(dep.tickers)
    for i ∈ eachindex(dep.tickers)
        growth_rates = JumpHMM.excess_growth_rates(dep.own_levels[:, i]; rf = 0.0, Δt = sp.Δt)
        expected = JumpHMM.validate(dep.own_models[i], growth_rates; number_of_paths = 5, seed = 3)
        @test reports[dep.tickers[i]].ks_pvalue == expected.ks_pvalue
        @test reports[dep.tickers[i]].wasserstein == expected.wasserstein
        @test reports[dep.tickers[i]].mean_kurtosis == expected.mean_kurtosis
    end

    # an unknown dependence model is rejected before the random number generator is seeded -
    unknown = JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySingleIndexDependenceModel, market = "SPY")))
    unknown.dependence = UnknownDependenceModel()
    @test_throws ArgumentError JumpHMM.simulate(unknown, 0)
    @test_throws ArgumentError JumpHMM.simulate(unknown, 10; seed = 5)
end
