@testset "Gaussian and Student-t copulas" begin
    P = inputs["prices"]
    GA = JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252)
    R = GA[:, 2:5]
    models = [JumpHMM.build(JumpHMM.MyGaussianCopulaModel, (growth_rates = R,)),
              JumpHMM.build(JumpHMM.MyStudentTCopulaModel, (growth_rates = R,))]
    for m ∈ models
        @test m.Σ == transpose(m.Σ)
        @test all(diag(m.Σ) .== 1.0)
        U = JumpHMM.sample_dependence(m, 50)
        @test size(U) == (50, 4)
        @test all(0.0 .< U .< 1.0)
    end
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyGaussianCopulaModel, (growth_rates = R[:, 1:1],))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyStudentTCopulaModel, (growth_rates = R[:, 1:1],))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyGaussianCopulaModel, (growth_rates = R[1:1, :],))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyGaussianCopulaModel, (growth_rates = fill(NaN, 5, 2),))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyGaussianCopulaModel, (returns = R,))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyStudentTCopulaModel, (returns = R,))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyStudentTCopulaModel, (growth_rates = fill(NaN, 5, 2),))
end
