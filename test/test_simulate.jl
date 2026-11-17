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
end
