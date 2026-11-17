using StatsBase

@testset "validate output" begin
    G = reference["finance/G_spy"]
    m = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 20, ν = 5.0, ϵ = 1e-3, λ = 50.0))
    r = JumpHMM.validate(m, G; number_of_paths = 6, seed = 1)
    @test length(r.ks_pvalue) == 6
    @test length(r.ad_pvalue) == 6
    @test length(r.wasserstein) == 6
    @test length(r.hellinger) == 6
    @test length(r.acf_mae) == 6
    @test length(r.kurtosis_sim) == 6
    @test 0.0 <= r.ks_pass_rate <= 1.0
    @test 0.0 <= r.ad_pass_rate <= 1.0
    @test r.α == 0.05
    @test r.kurtosis_emp == StatsBase.kurtosis(G)
end

@testset "distances" begin
    x = [0.1, -0.2, 0.05, 0.3, -0.1]
    @test JumpHMM.wasserstein_distance(x, x) == 0.0
    @test JumpHMM.hellinger_distance(x, x) == 0.0
    @test JumpHMM.wasserstein_distance(x, x .+ 1.0) > 0.0
    @test JumpHMM.hellinger_distance(fill(1.0, 4), fill(1.0, 4)) == 0.0
    @test JumpHMM.wasserstein_distance(x, [0.0, 0.2, 0.4]) >= 0.0
end
