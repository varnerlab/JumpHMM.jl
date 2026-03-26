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

        # sample
        dep_returns = sample_dependence(sim, 100)
        @test size(dep_returns) == (100, 2)
    end

end
