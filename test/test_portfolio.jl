# a dependence model that the portfolio does not know -
struct UnknownDependenceModel <: JumpHMM.AbstractDependenceModel end

@testset "portfolio models" begin
    P = inputs["prices"][:, 1:6]
    tickers = inputs["tickers"][1:6]
    sector = inputs["sectors"][1:5]
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
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySectorFactorModel, sector = sector)))
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySectorFactorModel, market = "SPY")))
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = JumpHMM.MyGaussianCopulaModel, tickers = tickers[1:5])))
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySingleIndexDependenceModel, market = "XYZ")))
    @test_throws ArgumentError JumpHMM.build(T, merge(base, (dependence = Float64,)))

    # a copula portfolio stores an empty market ticker, even when a market is given -
    gc = JumpHMM.build(T, merge(base, (dependence = JumpHMM.MyGaussianCopulaModel, market = "SPY")))
    @test gc.market_ticker == ""

    # a portfolio of one ticker and a hybrid model of the market and one stock can be built and simulated -
    single = JumpHMM.build(T, (tickers = tickers[2:2], prices = P[:, 2:2], number_of_states = 20, ν = 5.0,
        dependence = JumpHMM.MyGaussianCopulaModel))
    single_paths = JumpHMM.simulate(single, 30; number_of_paths = 2, seed = 4)
    @test size(single_paths[tickers[2]].growth_rates) == (30, 2)
    hy_one = JumpHMM.build(T, (tickers = tickers[1:2], prices = P[:, 1:2], number_of_states = 20, ν = 5.0,
        dependence = JumpHMM.MyHybridSingleIndexDependenceModel, market = "SPY"))
    hy_one_paths = JumpHMM.simulate(hy_one, 30; number_of_paths = 2, seed = 4)
    @test size(hy_one_paths[tickers[2]].growth_rates) == (29, 2)

    # the number of paths must be at least 1 -
    @test_throws ArgumentError JumpHMM.simulate(gc, 10; number_of_paths = 0)
    @test_throws ArgumentError JumpHMM.validate(gc, P; number_of_paths = 0)
    @test_throws ArgumentError JumpHMM.tune(gc, P; ϵ_range = ϵ_range, λ_range = λ_range, number_of_paths = 0)

    # without p_neg and N_tail, each tuned model keeps its own tail settings; with them, every tuned model takes them -
    gc_tails = deepcopy(gc)
    gc_tails.marginals[tickers[2]].p_neg = 0.6
    gc_tails.marginals[tickers[2]].N_tail = 3
    tuned_tails = JumpHMM.tune(gc_tails, P; ϵ_range = ϵ_range, λ_range = λ_range, number_of_paths = 2, seed = 25)
    @test tuned_tails.marginals[tickers[2]].p_neg == 0.6
    @test tuned_tails.marginals[tickers[2]].N_tail == 3
    @test tuned_tails.marginals[tickers[3]].p_neg == 0.52
    @test tuned_tails.marginals[tickers[3]].N_tail == 5
    tuned_given = JumpHMM.tune(gc_tails, P; ϵ_range = ϵ_range, λ_range = λ_range, number_of_paths = 2, p_neg = 0.55, N_tail = 4,
        seed = 25)
    for ticker ∈ gc_tails.tickers
        @test tuned_given.marginals[ticker].p_neg == 0.55
        @test tuned_given.marginals[ticker].N_tail == 4
    end

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

    # tuning a hybrid portfolio tunes its market model like the market model on its own. The models of the assets are tuned twice,
    # once for the portfolio and once for the hybrid model, so the two tuned models are separate. With a seed, both passes agree -
    tuned_hybrid = JumpHMM.tune(hy, P; ϵ_range = ϵ_range, λ_range = λ_range, number_of_paths = 5, seed = 24)
    expected_market = JumpHMM.tune(hy.dependence.market_model, market_growth_rates; ϵ_range = ϵ_range, λ_range = λ_range,
        number_of_paths = 5, seed = 24)
    @test tuned_hybrid.dependence.market_model.ϵ == expected_market.ϵ
    @test tuned_hybrid.dependence.market_model.λ == expected_market.λ
    for ticker ∈ hy.tickers
        @test (tuned_hybrid.marginals[ticker] === tuned_hybrid.dependence.marginals[ticker]) == false
        @test tuned_hybrid.marginals[ticker].ϵ == tuned_hybrid.dependence.marginals[ticker].ϵ
        @test tuned_hybrid.marginals[ticker].λ == tuned_hybrid.dependence.marginals[ticker].λ
    end

    # a sector factor portfolio validates each own model on its level series -
    sp = JumpHMM.build(T, (tickers = inputs["tickers"], prices = inputs["prices"], number_of_states = 20, ν = 5.0,
        dependence = JumpHMM.MySectorFactorModel, market = "SPY", sector = inputs["sectors"]))
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

    # an unknown dependence model is rejected before the random number generator is seeded. We save the state of the generator,
    # call the rejected simulate, and check that the next draw is the draw from the saved state -
    unknown = JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySingleIndexDependenceModel, market = "SPY")))
    unknown.dependence = UnknownDependenceModel()
    @test_throws ArgumentError JumpHMM.simulate(unknown, 0)
    Random.seed!(99)
    saved_state = copy(Random.default_rng())
    @test_throws ArgumentError JumpHMM.simulate(unknown, 10; seed = 5)
    draw_after_rejection = rand()
    copy!(Random.default_rng(), saved_state)
    @test draw_after_rejection == rand()
end
