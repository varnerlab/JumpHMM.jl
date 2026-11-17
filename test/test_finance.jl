@testset "finance unit tests" begin
    @test all(JumpHMM.excess_growth_rates(fill(100.0, 10)) .== 0.0)
    prices = [100.0, 101.5, 99.0, 102.0, 103.5]
    growth_rates = JumpHMM.excess_growth_rates(prices; rf = 0.03, Δt = 1/252)
    rebuilt = JumpHMM.prices_from_growth_rates(growth_rates, prices[1]; rf = 0.03, Δt = 1/252)
    @test maximum(abs.(rebuilt .- prices)) < 1e-10
end
