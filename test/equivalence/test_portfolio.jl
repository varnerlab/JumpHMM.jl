@testset "sector portfolio equals 0.4.1" begin
    P = inputs["prices"]
    tickers = inputs["tickers"]

    # build, tune, and simulate -
    pf = JumpHMM.build(JumpHMM.MyPortfolioModel, (
        tickers = tickers, prices = P, dependence = JumpHMM.MySectorFactorModel, market = "SPY", sector = inputs["sectors"],
        number_of_states = 20, ν = 5.0, sector_options = (center_steps = 20_000,)))
    pt = JumpHMM.tune(pf, P; ϵ_range = [1e-4, 3e-3], λ_range = [10.0, 100.0], number_of_paths = 5, seed = 15)
    @test pt isa JumpHMM.MyPortfolioModel
    dep = pt.dependence
    @test [m.ϵ for m ∈ dep.sector_models] == reference["portfolio/tuned/sector_ϵ"]
    @test [m.λ for m ∈ dep.sector_models] == reference["portfolio/tuned/sector_λ"]
    @test [m.ϵ for m ∈ dep.own_models] == reference["portfolio/tuned/own_ϵ"]
    @test [m.λ for m ∈ dep.own_models] == reference["portfolio/tuned/own_λ"]

    # the simulated matrices, one set for each ticker, the market included -
    paths = JumpHMM.simulate(pt, 300; number_of_paths = 3, seed = 16)
    @test sort(collect(keys(paths))) == sort(tickers)
    for ticker ∈ tickers
        @test paths[ticker].growth_rates == reference["portfolio/simulate/" * ticker * "/growth_rates"]
        @test paths[ticker].states == reference["portfolio/simulate/" * ticker * "/states"]
        @test paths[ticker].episodes == reference["portfolio/simulate/" * ticker * "/episodes"]
    end
end

@testset "Gaussian copula portfolio equals 0.4.1" begin
    P = inputs["prices"][:, 1:5]
    tickers = inputs["tickers"][1:5]
    pg = JumpHMM.build(JumpHMM.MyPortfolioModel, (
        tickers = tickers, prices = P, dependence = JumpHMM.MyGaussianCopulaModel, number_of_states = 20, ν = 5.0))

    # simulate -
    paths = JumpHMM.simulate(pg, 200; number_of_paths = 2, seed = 17)
    for ticker ∈ tickers
        @test paths[ticker].growth_rates == reference["portfolio_gauss/simulate/" * ticker * "/growth_rates"]
        @test paths[ticker].states == reference["portfolio_gauss/simulate/" * ticker * "/states"]
        @test paths[ticker].episodes == reference["portfolio_gauss/simulate/" * ticker * "/episodes"]
    end

    # validate. The Hellinger fields are compared with the corrected 0.4.1 keys (see test_report_equals_reference) -
    reports = JumpHMM.validate(pg, P; number_of_paths = 3, seed = 18)
    for ticker ∈ tickers
        @test reports[ticker] isa JumpHMM.MyValidationReportModel
        test_report_equals_reference(reports[ticker], "portfolio_gauss/validate/" * ticker)
    end
end

@testset "portfolios seeded once equal 0.4.1" begin
    P = inputs["prices"][1:301, :]
    tickers = inputs["tickers"]
    ϵ_range = [0.01, 0.05]
    λ_range = [2.0, 10.0]

    # a Gaussian copula portfolio: one Random.seed! and no seed keyword, so the order in which we visit the models sets the draws -
    pg = JumpHMM.build(JumpHMM.MyPortfolioModel, (
        tickers = tickers, prices = P, dependence = JumpHMM.MyGaussianCopulaModel, number_of_states = 10, ν = 5.0))
    Random.seed!(7)
    tuned = JumpHMM.tune(pg, P; ϵ_range = ϵ_range, λ_range = λ_range, number_of_paths = 3, number_of_steps = 80)
    for ticker ∈ tickers
        @test tuned.marginals[ticker].ϵ == reference["portfolio_once/gauss/tune/" * ticker * "/ϵ"]
        @test tuned.marginals[ticker].λ == reference["portfolio_once/gauss/tune/" * ticker * "/λ"]
    end
    Random.seed!(7)
    reports = JumpHMM.validate(pg, P; number_of_paths = 1)
    for ticker ∈ tickers
        test_report_equals_reference(reports[ticker], "portfolio_once/gauss/validate/" * ticker)
    end

    # a hybrid portfolio: the portfolio marginals, then the market, then the hybrid model's own marginals -
    ph = JumpHMM.build(JumpHMM.MyPortfolioModel, (
        tickers = tickers[1:5], prices = P[:, 1:5], dependence = JumpHMM.MyHybridSingleIndexDependenceModel,
        market = "SPY", number_of_states = 10, ν = 5.0))
    Random.seed!(7)
    th = JumpHMM.tune(ph, P[:, 1:5]; ϵ_range = ϵ_range, λ_range = λ_range, number_of_paths = 3, number_of_steps = 80)
    for ticker ∈ ph.tickers
        @test th.marginals[ticker].ϵ == reference["portfolio_once/hybrid/tune/marginals/" * ticker * "/ϵ"]
        @test th.marginals[ticker].λ == reference["portfolio_once/hybrid/tune/marginals/" * ticker * "/λ"]
        @test th.dependence.marginals[ticker].ϵ == reference["portfolio_once/hybrid/tune/dependence/" * ticker * "/ϵ"]
        @test th.dependence.marginals[ticker].λ == reference["portfolio_once/hybrid/tune/dependence/" * ticker * "/λ"]
    end
    @test th.dependence.market_model.ϵ == reference["portfolio_once/hybrid/tune/market/ϵ"]
    @test th.dependence.market_model.λ == reference["portfolio_once/hybrid/tune/market/λ"]
end

@testset "sector portfolio with a monthly time step equals 0.4.1" begin
    P = inputs["prices"]

    # the time step in sector_options is the one the portfolio stores and validates with -
    ps = JumpHMM.build(JumpHMM.MyPortfolioModel, (
        tickers = inputs["tickers"], prices = P, dependence = JumpHMM.MySectorFactorModel, market = "SPY",
        sector = inputs["sectors"], number_of_states = 20, ν = 5.0, sector_options = (Δt = 1/12, center_steps = 1000)))
    @test ps.Δt == 1/12
    reports = JumpHMM.validate(ps, P; number_of_paths = 1, seed = 7)
    for ticker ∈ ["ADP", "AFL"]
        test_report_equals_reference(reports[ticker], "portfolio_sector_dt/validate/" * ticker)
    end
end

@testset "hybrid portfolio of the market and one stock equals 0.4.1" begin
    P = inputs["prices"][:, 1:2]
    tickers = inputs["tickers"][1:2]

    # the hybrid model on its own -
    h = JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel, (
        tickers = tickers, prices = P, market = "SPY", number_of_states = 20, ν = 5.0))
    Random.seed!(5)
    @test JumpHMM.sample_dependence(h, 100) == reference["hybrid_one/sample"]

    # the same model in a portfolio -
    p = JumpHMM.build(JumpHMM.MyPortfolioModel, (
        tickers = tickers, prices = P, dependence = JumpHMM.MyHybridSingleIndexDependenceModel, market = "SPY",
        number_of_states = 20, ν = 5.0))
    paths = JumpHMM.simulate(p, 100; number_of_paths = 2, seed = 6)
    prefix = "hybrid_one/portfolio/simulate/" * tickers[2]
    @test paths[tickers[2]].growth_rates == reference[prefix * "/growth_rates"]
    @test paths[tickers[2]].states == reference[prefix * "/states"]
    @test paths[tickers[2]].episodes == reference[prefix * "/episodes"]
end

@testset "portfolio dictionaries keep the key order of 0.4.1" begin
    P = inputs["prices"]
    tickers = inputs["tickers"]
    T = JumpHMM.MyPortfolioModel
    base = (tickers = tickers, prices = P, number_of_states = 20, ν = 5.0)

    # the order of the marginals sets which model draws first when a portfolio is tuned or validated without a seed -
    portfolios = [
        ("sector", JumpHMM.build(T, merge(base, (dependence = JumpHMM.MySectorFactorModel, market = "SPY",
            sector = inputs["sectors"], sector_options = (center_steps = 1000,))))),
        ("gauss", JumpHMM.build(T, merge(base, (dependence = JumpHMM.MyGaussianCopulaModel,)))),
        ("hybrid", JumpHMM.build(T, merge(base, (dependence = JumpHMM.MyHybridSingleIndexDependenceModel, market = "SPY")))),
    ]
    for (name, portfolio) ∈ portfolios
        @test collect(keys(portfolio.marginals)) == reference["key_order/" * name * "/marginals"]
        @test collect(keys(portfolio.tickers_map)) == reference["key_order/" * name * "/tickers_map"]
    end
end
