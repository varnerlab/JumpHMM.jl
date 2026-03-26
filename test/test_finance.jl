@testset "Finance Helpers" begin

    @testset "excess_growth_rates - Vector" begin
        prices = [100.0, 101.0, 99.0, 102.0]
        G = excess_growth_rates(prices; rf=0.0, dt=1/252)
        @test length(G) == 3
        @test G[1] ≈ 252.0 * log(101.0 / 100.0)
        @test G[2] ≈ 252.0 * log(99.0 / 101.0)
    end

    @testset "excess_growth_rates - with risk-free rate" begin
        prices = [100.0, 105.0]
        G = excess_growth_rates(prices; rf=0.05, dt=1/252)
        @test G[1] ≈ 252.0 * log(105.0 / 100.0) - 0.05
    end

    @testset "excess_growth_rates - Matrix" begin
        prices = [100.0 200.0; 101.0 202.0; 99.0 198.0]
        G = excess_growth_rates(prices; rf=0.0, dt=1/252)
        @test size(G) == (2, 2)
        @test G[1, 1] ≈ 252.0 * log(101.0 / 100.0)
        @test G[1, 2] ≈ 252.0 * log(202.0 / 200.0)
    end

    @testset "prices_from_growth_rates" begin
        prices = [100.0, 101.0, 99.0, 102.0, 98.0]
        rf = 0.05
        dt = 1/252
        G = excess_growth_rates(prices; rf=rf, dt=dt)
        P_reconstructed = prices_from_growth_rates(G, prices[1]; rf=rf, dt=dt)
        @test length(P_reconstructed) == length(prices)
        @test P_reconstructed ≈ prices atol=1e-10
    end

end
