@testset "simulate equals 0.4.1" begin
    P = inputs["prices"]
    G = JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.0, Δt = 1/252)
    T = JumpHMM.MyJumpHiddenMarkovModel
    base = (growth_rates = G, number_of_states = 100, ν = 5.0)
    jump = JumpHMM.build(T, merge(base, (ϵ = 1e-3, λ = 90.0)))
    cases = [
        ("jump", jump, 2766, (number_of_paths = 20, seed = 5660)),
        ("nojump", JumpHMM.build(T, base), 2766, (number_of_paths = 20, seed = 5660)),
        ("k0", JumpHMM.build(T, merge(base, (ϵ = 0.05, λ = 0.5))), 500, (number_of_paths = 10, seed = 1)),
        ("overrun", JumpHMM.build(T, merge(base, (ϵ = 0.05, λ = 90.0))), 50, (number_of_paths = 10, seed = 2)),
        ("start", jump, 100, (number_of_paths = 5, start = 50, seed = 3)),
    ]
    for (name, model, steps, options) in cases
        paths = JumpHMM.simulate(model, steps; options...)
        @test paths.growth_rates == reference["simulate/$(name)/growth_rates"]
        @test paths.states == reference["simulate/$(name)/states"]
        @test paths.episodes == reference["simulate/$(name)/episodes"]
    end
end
