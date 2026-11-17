@testset "decode, forward_filter, log_likelihood" begin
    P = inputs["prices"]
    G = JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.0, Δt = 1/252)
    m = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 20, ν = 5.0, ϵ = 1e-3, λ = 50.0))
    states = JumpHMM.decode(m, G)
    @test length(states) == length(G)
    @test all(1 .<= states .<= m.number_of_states)
    filtered = JumpHMM.forward_filter(m, G)
    @test size(filtered) == (length(G), m.number_of_states)
    @test all(abs.(sum(filtered; dims = 2) .- 1.0) .< 1e-10)
    @test isfinite(JumpHMM.log_likelihood(m, G))
    @test_throws ArgumentError JumpHMM.decode(m, Float64[])
    @test_throws ArgumentError JumpHMM.forward_filter(m, Float64[])
    @test_throws ArgumentError JumpHMM.log_likelihood(m, Float64[])
end
