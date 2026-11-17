@testset "finance equals 0.4.1" begin
    P = inputs["prices"]
    @test JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.0, Δt = 1/252) == reference["finance/G_spy"]
    @test JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252) == reference["finance/G_all"]
    @test JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.05, Δt = 1/252) == reference["finance/G_rf"]
    G = reference["finance/G_spy"]
    @test JumpHMM.prices_from_growth_rates(G, P[1, 1]; rf = 0.0, Δt = 1/252) == reference["finance/prices_back"]
end
