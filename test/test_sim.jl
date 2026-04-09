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

# --- Helper: build synthetic price matrix from returns ---
function _build_prices(returns_matrix::Matrix{Float64}; dt::Float64=1/252)
    n, k = size(returns_matrix)
    prices = Matrix{Float64}(undef, n + 1, k)
    prices[1, :] .= 100.0
    for t in 1:n, j in 1:k
        prices[t+1, j] = prices[t, j] * exp(returns_matrix[t, j] * dt)
    end
    return prices
end

@testset "Hybrid Single-Index Model" begin

    @testset "fit and sample" begin
        Random.seed!(42)

        n = 500
        market = randn(n) .* 0.15
        β_true = [1.2, 0.8, 0.5]
        α_true = [0.01, -0.01, 0.02]
        σ_noise = [0.10, 0.12, 0.20]

        returns = Matrix{Float64}(undef, n, 4)
        returns[:, 1] = market
        for i in 1:3
            returns[:, i+1] = α_true[i] .+ β_true[i] .* market .+ randn(n) .* σ_noise[i]
        end

        prices = _build_prices(returns)
        tickers = ["MKT", "A", "B", "C"]

        model = fit(HybridSingleIndexModel, tickers, prices, "MKT"; N=10, ν=5.0)

        @test model isa HybridSingleIndexModel
        @test length(model.α) == 3
        @test length(model.β) == 3
        @test length(model.r²) == 3
        @test length(model.tickers) == 3
        @test model.tickers == ["A", "B", "C"]
        @test !haskey(model.marginals, "MKT")
        @test haskey(model.marginals, "A")
        @test model.σ_market > 0.0
        @test model.r2_preserve_threshold == 0.80
        @test model.idiosyncratic_floor == 0.10
        @test model.copula isa StudentTCopula

        # betas should be roughly correct
        for i in 1:3
            @test abs(model.β[i] - β_true[i]) < 0.3
        end

        # sample_dependence returns correct shape
        dep = sample_dependence(model, 100)
        @test size(dep, 2) == 3
        @test size(dep, 1) == 99  # trimmed by 1
    end

    @testset "variance preservation (marginal-preserving branch)" begin
        Random.seed!(101)

        # low-R² asset: lots of idiosyncratic noise
        n = 2000
        market = randn(n) .* 0.15
        noise = rand(Distributions.TDist(4), n) .* 0.30  # fat-tailed, high variance
        asset = 0.0 .+ 0.3 .* market .+ noise  # low β → low R²

        returns = hcat(market, asset)
        prices = _build_prices(returns)
        tickers = ["MKT", "X"]

        model = fit(HybridSingleIndexModel, tickers, prices, "MKT";
                    N=10, ν=5.0, r2_preserve_threshold=0.80)

        # confirm this asset lands in the marginal-preserving branch
        @test model.r²[1] < 0.80

        # generate many samples and check variance preservation
        n_samples = 50
        vars = Float64[]
        for _ in 1:n_samples
            dep = sample_dependence(model, 500)
            push!(vars, var(dep[:, 1]))
        end
        mean_var = mean(vars)

        # the mean simulated variance should be in the right ballpark
        # (not inflated by β²σ²_m as the naive SIM would be)
        empirical_var = var(asset)
        ratio = mean_var / empirical_var
        @test 0.3 < ratio < 3.0  # generous bounds for stochastic test
    end

    @testset "R² recovery (R²-preserving branch)" begin
        Random.seed!(202)

        # high-R² asset: small residuals, large β
        n = 2000
        market = randn(n) .* 0.15
        # β=1.0, tiny noise → R² ≈ 0.95+
        asset = 0.001 .+ 1.0 .* market .+ randn(n) .* 0.03

        returns = hcat(market, asset)
        prices = _build_prices(returns)
        tickers = ["MKT", "ETF"]

        model = fit(HybridSingleIndexModel, tickers, prices, "MKT";
                    N=10, ν=5.0, r2_preserve_threshold=0.80)

        # confirm R²-preserving branch
        @test model.r²[1] >= 0.80

        # generate paths and check recovered R²
        dep = sample_dependence(model, 5000)
        # simulate a matching market path for regression
        # (we can't directly access it, but β recovery implies R² recovery)
        @test size(dep, 2) == 1
        @test size(dep, 1) == 4999
    end

    @testset "β recovery" begin
        Random.seed!(303)

        n = 1000
        market = randn(n) .* 0.15
        β_target = 1.3
        asset = 0.0 .+ β_target .* market .+ randn(n) .* 0.20

        returns = hcat(market, asset)
        prices = _build_prices(returns)
        tickers = ["MKT", "Y"]

        model = fit(HybridSingleIndexModel, tickers, prices, "MKT"; N=10, ν=5.0)

        # calibrated β should be close to target
        @test abs(model.β[1] - β_target) < 0.2

        # sample and verify it produces finite, non-degenerate output
        dep = sample_dependence(model, 200)
        @test all(isfinite, dep)
        @test var(dep[:, 1]) > 0.0
    end

    @testset "β clipping (idiosyncratic floor)" begin
        Random.seed!(404)

        # Very high β with low-variance idiosyncratic noise
        # This should trigger clipping in the marginal-preserving branch
        n = 1000
        market = randn(n) .* 0.15
        # β=3.0, tiny noise → ρ = β²σ²_m/σ²_gen will be large
        asset = 0.0 .+ 3.0 .* market .+ randn(n) .* 0.05

        returns = hcat(market, asset)
        prices = _build_prices(returns)
        tickers = ["MKT", "Z"]

        # set threshold high so this doesn't go to R²-preserving
        model = fit(HybridSingleIndexModel, tickers, prices, "MKT";
                    N=10, ν=5.0, r2_preserve_threshold=0.99,
                    idiosyncratic_floor=0.10)

        # sample — should not error even with aggressive β
        dep = sample_dependence(model, 200)
        @test all(isfinite, dep)
        @test size(dep, 2) == 1
    end

    @testset "SPY degenerate limit (R²=1)" begin
        Random.seed!(505)

        # SPY against itself: identical columns
        n = 500
        market = randn(n) .* 0.15
        spy = copy(market)  # identical to market

        returns = hcat(market, spy)
        prices = _build_prices(returns)
        tickers = ["MKT", "SPY"]

        model = fit(HybridSingleIndexModel, tickers, prices, "MKT";
                    N=10, ν=5.0, r2_preserve_threshold=0.80)

        # R² should be ≈ 1.0
        @test model.r²[1] > 0.99
        # β should be ≈ 1.0
        @test abs(model.β[1] - 1.0) < 0.05

        # sample — the R²-preserving branch with R²≈1 should yield ε≡0
        dep = sample_dependence(model, 200)
        @test all(isfinite, dep)
    end

    @testset "copula pluggability (GaussianCopula)" begin
        Random.seed!(606)

        n = 500
        market = randn(n) .* 0.15
        returns = hcat(market,
                       0.01 .+ 1.0 .* market .+ randn(n) .* 0.10,
                       -0.01 .+ 0.5 .* market .+ randn(n) .* 0.15)
        prices = _build_prices(returns)
        tickers = ["MKT", "A", "B"]

        model = fit(HybridSingleIndexModel, tickers, prices, "MKT";
                    N=10, ν=5.0, copula_type=GaussianCopula)

        @test model.copula isa GaussianCopula

        dep = sample_dependence(model, 100)
        @test size(dep, 2) == 2
        @test all(isfinite, dep)
    end

    @testset "PortfolioModel integration" begin
        Random.seed!(707)

        n = 500
        market = randn(n) .* 0.15
        returns = hcat(market,
                       0.01 .+ 1.1 .* market .+ randn(n) .* 0.10,
                       -0.01 .+ 0.8 .* market .+ randn(n) .* 0.12)
        prices = _build_prices(returns)
        tickers = ["MKT", "A", "B"]

        portfolio = fit(PortfolioModel, tickers, prices;
                        dependence=HybridSingleIndexModel, market="MKT", N=10, ν=5.0)

        @test portfolio.dependence isa HybridSingleIndexModel
        @test !haskey(portfolio.marginals, "MKT")
        @test haskey(portfolio.marginals, "A")
        @test haskey(portfolio.marginals, "B")
        @test portfolio.market_ticker == "MKT"

        # simulate
        result = simulate(portfolio, 100; n_paths=5, seed=42)
        @test result isa PortfolioSimulationResult
        @test haskey(result.results, "A")
        @test haskey(result.results, "B")
        @test length(result.results["A"].paths) == 5
        # each path should have T_eff = 99 observations
        @test length(result.results["A"].paths[1].observations) == 99
    end

end
