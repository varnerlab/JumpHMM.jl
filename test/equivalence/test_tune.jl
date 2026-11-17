@testset "tune equals 0.4.1" begin
    P = inputs["prices"]
    T = JumpHMM.MyJumpHiddenMarkovModel

    # the full sample -
    G = JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.0, Δt = 1/252)
    m = JumpHMM.build(T, (growth_rates = G, number_of_states = 100, ν = 5.0))
    t = JumpHMM.tune(m, G; ϵ_range = range(1e-4, 5e-3, length = 3), λ_range = range(10.0, 160.0, length = 3),
        number_of_paths = 20, seed = 5660)
    @test t.ϵ == reference["tune/ϵ"]
    @test t.λ == reference["tune/λ"]

    # a short sample where no candidate produces an episode -
    G_short = JumpHMM.excess_growth_rates(P[1:201, 1]; rf = 0.0, Δt = 1/252)
    m_short = JumpHMM.build(T, (growth_rates = G_short, number_of_states = 20, ν = 5.0))
    t_short = @test_logs (:warn,) JumpHMM.tune(m_short, G_short; ϵ_range = [1e-6], λ_range = [10.0],
        number_of_paths = 2, seed = 1)
    @test t_short.ϵ == reference["tune_none/ϵ"]
    @test t_short.λ == reference["tune_none/λ"]
end
