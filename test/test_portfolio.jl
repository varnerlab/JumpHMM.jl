@testset "Portfolio" begin

    @testset "fit and simulate with GaussianCopula" begin
        Random.seed!(42)
        n = 300
        # synthetic correlated prices
        Σ = [1.0 0.6; 0.6 1.0]
        Z = rand(MvNormal(zeros(2), Σ), n)'
        prices = 100.0 .* exp.(cumsum(Z .* 0.01, dims=1))

        tickers = ["A", "B"]
        portfolio = fit(PortfolioModel, tickers, prices;
                        dependence=GaussianCopula, N=10, ν=5.0)

        @test length(portfolio.tickers) == 2
        @test haskey(portfolio.marginals, "A")
        @test haskey(portfolio.marginals, "B")
        @test portfolio.dependence isa GaussianCopula

        # simulate
        result = simulate(portfolio, 50; n_paths=10, seed=42)
        @test result isa PortfolioSimulationResult
        @test haskey(result.results, "A")
        @test length(result.results["A"].paths) == 10
    end

    @testset "fit with SingleIndexModel" begin
        Random.seed!(42)
        n = 300
        # market + 2 assets
        market_ret = randn(n) .* 0.01
        asset1 = 0.001 .+ 1.1 .* market_ret .+ randn(n) .* 0.005
        asset2 = -0.001 .+ 0.9 .* market_ret .+ randn(n) .* 0.005

        prices = Matrix{Float64}(undef, n + 1, 3)
        prices[1, :] .= 100.0
        for t in 1:n
            prices[t+1, 1] = prices[t, 1] * exp(market_ret[t] * (1/252))
            prices[t+1, 2] = prices[t, 2] * exp(asset1[t] * (1/252))
            prices[t+1, 3] = prices[t, 3] * exp(asset2[t] * (1/252))
        end

        tickers = ["MKT", "A", "B"]
        portfolio = fit(PortfolioModel, tickers, prices;
                        dependence=SingleIndexModel, market="MKT", N=10, ν=5.0)

        @test portfolio.dependence isa SingleIndexModel
        # market ticker should not be in marginals
        @test !haskey(portfolio.marginals, "MKT")
        @test haskey(portfolio.marginals, "A")
    end

end
