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

@testset "zero-likelihood days" begin
    P = inputs["prices"]
    G = JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.0, Δt = 1/252)
    m = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 20, ν = 5.0, ϵ = 1e-3, λ = 50.0))
    uniform_row = fill(1.0 / m.number_of_states, m.number_of_states)

    # a growth rate of Inf has zero likelihood in every state, on day 1 and on a later day -
    first_day = vcat(Inf, G[1:5])
    later_day = vcat(G[1:5], Inf)
    @test JumpHMM.forward_filter(m, first_day)[1, :] == uniform_row
    @test JumpHMM.forward_filter(m, later_day)[6, :] == uniform_row
    @test all(abs.(sum(JumpHMM.forward_filter(m, later_day); dims = 2) .- 1.0) .< 1e-10)
    @test JumpHMM.log_likelihood(m, first_day) == -Inf
    @test JumpHMM.log_likelihood(m, later_day) == -Inf
    @test length(JumpHMM.decode(m, later_day)) == 6
end
