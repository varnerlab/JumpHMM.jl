@testset "Single-Index Model" begin

    @testset "fit and sample" begin
        Random.seed!(42)

        # generate synthetic market + 2 assets
        n = 500
        market = cumsum(randn(n)) .* 0.01
        β_true = [1.2, 0.8]
        α_true = [0.01, -0.01]

        returns = Matrix{Float64}(undef, n, 2)
        for i in 1:2
            returns[:, i] = α_true[i] .+ β_true[i] .* market .+ randn(n) .* 0.1
        end

        # need market prices for fit
        market_prices = 100.0 .* exp.(cumsum(market .* (1/252)))

        sim = fit(SingleIndexModel, returns, market;
                  market_prices=market_prices, N=10, ν=5.0)

        @test length(sim.α) == 2
        @test length(sim.β) == 2
        @test length(sim.σ_ε) == 2
        # betas should be roughly correct
        @test abs(sim.β[1] - 1.2) < 0.3
        @test abs(sim.β[2] - 0.8) < 0.3

        # residuals stored with correct shape
        @test size(sim.residuals) == (n, 2)
        @test sim.residual_method == :bootstrap

        # sample (bootstrap, default)
        dep_returns = sample_dependence(sim, 100)
        @test size(dep_returns) == (100, 2)
    end

    @testset "gaussian residual method" begin
        Random.seed!(123)

        n = 500
        market = cumsum(randn(n)) .* 0.01
        returns = Matrix{Float64}(undef, n, 2)
        for i in 1:2
            returns[:, i] = 0.01 .+ 1.0 .* market .+ randn(n) .* 0.1
        end
        market_prices = 100.0 .* exp.(cumsum(market .* (1/252)))

        sim = fit(SingleIndexModel, returns, market;
                  market_prices=market_prices, N=10, ν=5.0,
                  residual_method=:gaussian)

        @test sim.residual_method == :gaussian

        dep_returns = sample_dependence(sim, 100)
        @test size(dep_returns) == (100, 2)
    end

    @testset "bootstrap preserves tail structure" begin
        Random.seed!(99)

        # create fat-tailed residuals using t-distribution
        n = 2000
        market = cumsum(randn(n)) .* 0.01
        heavy_noise = rand(Distributions.TDist(3), n) .* 0.05

        returns = Matrix{Float64}(undef, n, 1)
        returns[:, 1] = 0.0 .+ 1.0 .* market .+ heavy_noise
        market_prices = 100.0 .* exp.(cumsum(market .* (1/252)))

        sim_boot = fit(SingleIndexModel, returns, market;
                       market_prices=market_prices, N=10, ν=5.0,
                       residual_method=:bootstrap)
        sim_gauss = fit(SingleIndexModel, returns, market;
                        market_prices=market_prices, N=10, ν=5.0,
                        residual_method=:gaussian)

        # generate many samples and compare residual kurtosis
        n_sim = 5000
        boot_returns = sample_dependence(sim_boot, n_sim)
        gauss_returns = sample_dependence(sim_gauss, n_sim)

        # bootstrap should preserve higher kurtosis than gaussian
        boot_kurt = kurtosis(boot_returns[:, 1])
        gauss_kurt = kurtosis(gauss_returns[:, 1])
        @test boot_kurt > gauss_kurt
    end

    @testset "bootstrap preserves cross-asset residual correlation" begin
        Random.seed!(77)

        # construct residuals with perfect positive correlation: rows are [x, x]
        n = 500
        market = cumsum(randn(n)) .* 0.01
        shared_noise = randn(n) .* 0.1
        returns = Matrix{Float64}(undef, n, 2)
        returns[:, 1] = 0.0 .+ 0.0 .* market .+ shared_noise
        returns[:, 2] = 0.0 .+ 0.0 .* market .+ shared_noise
        market_prices = 100.0 .* exp.(cumsum(market .* (1/252)))

        sim = fit(SingleIndexModel, returns, market;
                  market_prices=market_prices, N=10, ν=5.0,
                  residual_method=:bootstrap)

        dep_returns = sample_dependence(sim, 2000)
        # with shared bootstrap indices, residual correlation should be preserved
        @test cor(dep_returns[:, 1], dep_returns[:, 2]) > 0.9
    end

    @testset "invalid residual_method throws" begin
        Random.seed!(55)

        n = 100
        market = cumsum(randn(n)) .* 0.01
        returns = Matrix{Float64}(undef, n, 1)
        returns[:, 1] = 0.01 .+ 1.0 .* market .+ randn(n) .* 0.1
        market_prices = 100.0 .* exp.(cumsum(market .* (1/252)))

        @test_throws ArgumentError fit(SingleIndexModel, returns, market;
                                       market_prices=market_prices, N=10, ν=5.0,
                                       residual_method=:bogus)
    end

end
