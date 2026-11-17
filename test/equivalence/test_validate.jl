@testset "validate equals 0.4.1" begin
    P = inputs["prices"]
    T = JumpHMM.MyJumpHiddenMarkovModel
    G = JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.0, Δt = 1/252)
    mj = JumpHMM.build(T, (growth_rates = G, number_of_states = 100, ν = 5.0, ϵ = 1e-3, λ = 90.0))
    report = JumpHMM.validate(mj, G; number_of_paths = 20, seed = 9)
    @test report isa JumpHMM.MyValidationReportModel

    # the Hellinger fields are compared with the corrected 0.4.1 keys (see test_report_equals_reference) -
    test_report_equals_reference(report, "validate")
end

@testset "distances equal 0.4.1" begin
    P = inputs["prices"]
    GA = JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252)
    @test JumpHMM.wasserstein_distance(GA[:, 2], GA[:, 3]) == reference["distance/wasserstein"]

    # 0.5.0 counts the largest observation, which the 0.4.1 histogram dropped, so we compare with the corrected key -
    @test JumpHMM.hellinger_distance(GA[:, 2], GA[:, 3]) == reference["distance/hellinger_corrected"]
end
