@testset "validate equals 0.4.1" begin
    P = inputs["prices"]
    T = JumpHMM.MyJumpHiddenMarkovModel
    G = JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.0, Δt = 1/252)
    mj = JumpHMM.build(T, (growth_rates = G, number_of_states = 100, ν = 5.0, ϵ = 1e-3, λ = 90.0))
    report = JumpHMM.validate(mj, G; number_of_paths = 20, seed = 9)
    @test report isa JumpHMM.MyValidationReportModel
    for name ∈ ["ks_pvalue", "ad_pvalue", "wasserstein", "hellinger", "acf_mae", "kurtosis_sim", "kurtosis_emp",
        "ks_pass_rate", "ad_pass_rate", "mean_acf_mae", "mean_wasserstein", "mean_hellinger", "mean_kurtosis", "α"]
        @test getfield(report, Symbol(name)) == reference["validate/" * name]
    end
end

@testset "distances equal 0.4.1" begin
    P = inputs["prices"]
    GA = JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252)
    @test JumpHMM.wasserstein_distance(GA[:, 2], GA[:, 3]) == reference["distance/wasserstein"]
    @test JumpHMM.hellinger_distance(GA[:, 2], GA[:, 3]) == reference["distance/hellinger"]
end
