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

@testset "copulas of one asset equal 0.4.1" begin
    P = inputs["prices"]
    GA = JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252)
    R = GA[:, 2:2]

    # Gaussian copula -
    gc = JumpHMM.build(JumpHMM.MyGaussianCopulaModel, (growth_rates = R,))
    @test gc.Σ == reference["copula_one/gaussian/Σ"]
    Random.seed!(19)
    @test JumpHMM.sample_dependence(gc, 50) == reference["copula_one/gaussian/U"]

    # Student-t copula -
    tc = JumpHMM.build(JumpHMM.MyStudentTCopulaModel, (growth_rates = R,))
    @test tc.Σ == reference["copula_one/studentt/Σ"]
    @test tc.ν == reference["copula_one/studentt/ν"]
    Random.seed!(20)
    @test JumpHMM.sample_dependence(tc, 50) == reference["copula_one/studentt/U"]

    # vine copula: one asset gives a vine with no edges -
    vc = JumpHMM.build(JumpHMM.MyVineCopulaModel, (growth_rates = R,))
    @test vc.order == reference["copula_one/vine/order"]
    number_of_edges = 0
    for tree ∈ vc.edges
        number_of_edges += length(tree)
    end
    @test number_of_edges == reference["copula_one/vine/number_of_edges"]
    Random.seed!(21)
    @test JumpHMM.sample_dependence(vc, 50) == reference["copula_one/vine/U"]
end
