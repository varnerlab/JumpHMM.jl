@testset "build rejects bad inputs" begin
    G = reference["finance/G_spy"]
    T = JumpHMM.MyJumpHiddenMarkovModel
    base = (growth_rates = G, number_of_states = 10, ν = 5.0)
    bad = [
        (merge(base, (ϵ = 1.5,)), "ϵ"),
        (merge(base, (λ = 0.0,)), "λ"),
        (merge(base, (p_neg = -0.1,)), "p_neg"),
        (merge(base, (N_tail = 0,)), "N_tail"),
        (merge(base, (number_of_states = 0,)), "number_of_states"),
        ((number_of_states = 10, ν = 5.0), "growth_rates"),
    ]
    for (data, key) in bad
        err = try
            JumpHMM.build(T, data)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(key, err.msg)
    end
    m = JumpHMM.build(T, merge(base, (ϵ = 0.2, Δt = 1/12)))
    @test m.ϵ == 0.2 && m.Δt == 1/12 && m.N_tail == 5 && m.p_neg == 0.52
end
