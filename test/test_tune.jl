@testset "tune output" begin
    G = reference["finance/G_spy"]
    T = JumpHMM.MyJumpHiddenMarkovModel
    m = JumpHMM.build(T, (growth_rates = G, number_of_states = 20, ν = 5.0))
    t = JumpHMM.tune(m, G; ϵ_range = [1e-3, 5e-3], λ_range = [10.0, 50.0], number_of_paths = 5, number_of_steps = 300,
        p_neg = 0.6, N_tail = 3, seed = 1)
    @test t isa T
    @test t.ϵ in [1e-3, 5e-3]
    @test t.λ in [10.0, 50.0]
    @test t.p_neg == 0.6
    @test t.N_tail == 3
    @test t.transition == m.transition

    # the input model is unchanged -
    @test m.ϵ == 0.0
    @test m.λ == 1.0
    @test m.p_neg != 0.6
end
