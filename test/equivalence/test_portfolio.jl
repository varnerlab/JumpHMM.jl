@testset "sector portfolio equals 0.4.1" begin
    P = inputs["prices"]
    tickers = inputs["tickers"]

    # build, tune, and simulate -
    pf = JumpHMM.build(JumpHMM.MyPortfolioModel, (
        tickers = tickers, prices = P, dependence = JumpHMM.MySectorFactorModel, market = "SPY", sectors = inputs["sectors"],
        number_of_states = 20, ν = 5.0, sector_options = (center_steps = 20_000,)))
    pt = JumpHMM.tune(pf, P; ϵ_range = [1e-4, 3e-3], λ_range = [10.0, 100.0], number_of_paths = 5, seed = 15)
    @test pt isa JumpHMM.MyPortfolioModel
    dep = pt.dependence
    @test [m.ϵ for m ∈ dep.sector_models] == reference["portfolio/tuned/sector_ϵ"]
    @test [m.λ for m ∈ dep.sector_models] == reference["portfolio/tuned/sector_λ"]
    @test [m.ϵ for m ∈ dep.own_models] == reference["portfolio/tuned/own_ϵ"]
    @test [m.λ for m ∈ dep.own_models] == reference["portfolio/tuned/own_λ"]

    # the simulated matrices. The reference stores the assets, and the market is one more entry -
    paths = JumpHMM.simulate(pt, 300; number_of_paths = 3, seed = 16)
    @test sort(collect(keys(paths))) == sort(tickers)
    @test size(paths["SPY"].growth_rates) == (300, 3)
    for ticker ∈ pt.tickers
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

    # validate -
    reports = JumpHMM.validate(pg, P; number_of_paths = 3, seed = 18)
    for ticker ∈ tickers
        report = reports[ticker]
        @test report isa JumpHMM.MyValidationReportModel
        for name ∈ ["ks_pvalue", "ad_pvalue", "wasserstein", "hellinger", "acf_mae", "kurtosis_sim", "kurtosis_emp",
            "ks_pass_rate", "ad_pass_rate", "mean_acf_mae", "mean_wasserstein", "mean_hellinger", "mean_kurtosis", "α"]
            @test getfield(report, Symbol(name)) == reference["portfolio_gauss/validate/" * ticker * "/" * name]
        end
    end
end
