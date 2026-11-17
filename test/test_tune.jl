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

    # without p_neg and N_tail, the tuned model keeps the tail settings of the input model -
    m_tails = JumpHMM.build(T, (growth_rates = G, number_of_states = 20, ν = 5.0, p_neg = 0.6, N_tail = 3))
    kept = JumpHMM.tune(m_tails, G; ϵ_range = [1e-3, 5e-3], λ_range = [10.0, 50.0], number_of_paths = 5, number_of_steps = 300, seed = 1)
    @test kept.p_neg == 0.6
    @test kept.N_tail == 3

    # bad settings are rejected before any simulation, NaN included -
    @test_throws ArgumentError JumpHMM.tune(m, G; ϵ_range = [-0.1], λ_range = [10.0], number_of_paths = 2)
    @test_throws ArgumentError JumpHMM.tune(m, G; ϵ_range = [1.1], λ_range = [10.0], number_of_paths = 2)
    @test_throws ArgumentError JumpHMM.tune(m, G; ϵ_range = [NaN], λ_range = [10.0], number_of_paths = 2)
    @test_throws ArgumentError JumpHMM.tune(m, G; ϵ_range = [1e-3], λ_range = [0.0], number_of_paths = 2)
    @test_throws ArgumentError JumpHMM.tune(m, G; ϵ_range = [1e-3], λ_range = [NaN], number_of_paths = 2)
    @test_throws ArgumentError JumpHMM.tune(m, G; ϵ_range = [1e-3], λ_range = [10.0], number_of_paths = 2, p_neg = 1.1)
    @test_throws ArgumentError JumpHMM.tune(m, G; ϵ_range = [1e-3], λ_range = [10.0], number_of_paths = 2, p_neg = NaN)
    @test_throws ArgumentError JumpHMM.tune(m, G; ϵ_range = [1e-3], λ_range = [10.0], number_of_paths = 2, N_tail = 0)
    @test_throws ArgumentError JumpHMM.tune(m, G; ϵ_range = [1e-3], λ_range = [10.0], number_of_paths = 0)

    # the rejected calls leave the random number generator as it was -
    Random.seed!(31)
    saved_state = copy(Random.default_rng())
    @test_throws ArgumentError JumpHMM.tune(m, G; ϵ_range = [1.1], λ_range = [10.0], number_of_paths = 2, seed = 5)
    draw_after_rejection = rand()
    copy!(Random.default_rng(), saved_state)
    @test draw_after_rejection == rand()

    # a short number_of_steps clips acf_lags (default 25) to number_of_steps - 1 -
    short = JumpHMM.tune(m, G; ϵ_range = [1e-2, 5e-2], λ_range = [10.0, 50.0], number_of_paths = 5, number_of_steps = 10, seed = 1)
    @test short isa T
    @test short.ϵ in [1e-2, 5e-2]
    @test short.λ in [10.0, 50.0]
end
