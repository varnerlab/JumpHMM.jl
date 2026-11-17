@testset "simulate output" begin
    G = reference["finance/G_spy"]
    T = JumpHMM.MyJumpHiddenMarkovModel
    model = JumpHMM.build(T, (growth_rates = G, number_of_states = 20, ν = 5.0))
    paths = JumpHMM.simulate(model, 30; number_of_paths = 7, seed = 1)
    @test size(paths.growth_rates) == (30, 7)
    @test size(paths.states) == (30, 7)
    @test size(paths.episodes) == (30, 7)
    @test !any(paths.episodes)
    @test_throws ArgumentError JumpHMM.simulate(model, 0)

    # start = k puts state k in row 1 of every path -
    for k in [1, 7, 20]
        started = JumpHMM.simulate(model, 5; number_of_paths = 4, start = k, seed = 2)
        @test all(started.states[1, :] .== k)
    end

    # the same seed gives identical matrices -
    first_call = JumpHMM.simulate(model, 30; number_of_paths = 7, seed = 11)
    second_call = JumpHMM.simulate(model, 30; number_of_paths = 7, seed = 11)
    @test first_call.growth_rates == second_call.growth_rates
    @test first_call.states == second_call.states
    @test first_call.episodes == second_call.episodes
end
