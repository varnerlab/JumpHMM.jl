using Random

@testset "copulas equal 0.4.1" begin
    P = inputs["prices"]
    GA = JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252)
    R = GA[:, 2:5]

    # Gaussian copula -
    gc = JumpHMM.build(JumpHMM.MyGaussianCopulaModel, (growth_rates = R,))
    @test gc.Σ == reference["copula/gaussian/Σ"]
    Random.seed!(7)
    @test JumpHMM.sample_dependence(gc, 200) == reference["copula/gaussian/U"]

    # Student-t copula -
    tc = JumpHMM.build(JumpHMM.MyStudentTCopulaModel, (growth_rates = R,))
    @test tc.Σ == reference["copula/studentt/Σ"]
    @test tc.ν == reference["copula/studentt/ν"]
    Random.seed!(8)
    @test JumpHMM.sample_dependence(tc, 200) == reference["copula/studentt/U"]
end
