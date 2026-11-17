@testset "decode equals 0.4.1" begin
    P = inputs["prices"]
    G = JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.0, Δt = 1/252)
    mj = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 100, ν = 5.0, ϵ = 1e-3, λ = 90.0))
    @test JumpHMM.decode(mj, G[1:500]) == reference["decode/states"]
    @test JumpHMM.forward_filter(mj, G[1:500]) == reference["decode/filter"]
    @test JumpHMM.log_likelihood(mj, G[1:500]) == reference["decode/loglik"]
end
