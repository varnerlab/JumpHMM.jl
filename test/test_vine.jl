@testset "vine copula" begin
    P = inputs["prices"]
    GA = JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252)
    R = GA[:, 2:5]
    vine = JumpHMM.build(JumpHMM.MyVineCopulaModel, (growth_rates = R,))

    # the samples are uniform numbers in the right shape -
    U = JumpHMM.sample_dependence(vine, 50)
    @test size(U) == (50, 4)
    @test all(0.0 .< U .< 1.0)

    # each h-function is undone by its inverse -
    copulas = [JumpHMM.build(JumpHMM.MyGaussianBivariateCopulaModel, (ρ = 0.5,)),
               JumpHMM.build(JumpHMM.MyStudentTBivariateCopulaModel, (ρ = 0.5, ν = 5.0)),
               JumpHMM.build(JumpHMM.MyClaytonBivariateCopulaModel, (θ = 2.0,)),
               JumpHMM.build(JumpHMM.MyGumbelBivariateCopulaModel, (θ = 2.0,)),
               JumpHMM.build(JumpHMM.MyFrankBivariateCopulaModel, (θ = 3.0,))]
    for c ∈ copulas
        for (u, v) ∈ [(0.2, 0.3), (0.5, 0.5), (0.7, 0.4), (0.9, 0.8)]
            w = JumpHMM._bicop_hfunc(c, u, v)
            @test abs(JumpHMM._bicop_hinv(c, w, v) - u) < 1e-8
        end
    end

    # the Frank copula with θ = 0 is the independence copula: the h-function returns u, its inverse returns w, and the log-density is 0 -
    independent = JumpHMM.build(JumpHMM.MyFrankBivariateCopulaModel, (θ = 0.0,))
    for (u, v) ∈ [(0.2, 0.3), (0.5, 0.5), (0.7, 0.4), (0.9, 0.8)]
        @test JumpHMM._bicop_hfunc(independent, u, v) == u
        @test JumpHMM._bicop_hinv(independent, u, v) == u
        @test JumpHMM._bicop_logpdf(independent, u, v) == 0.0
    end

    # a vine of one asset has no edges, and its draws are uniform numbers -
    single = JumpHMM.build(JumpHMM.MyVineCopulaModel, (growth_rates = R[:, 1:1],))
    @test single.number_of_assets == 1
    number_of_edges = 0
    for tree ∈ single.edges
        number_of_edges += length(tree)
    end
    @test number_of_edges == 0
    U = JumpHMM.sample_dependence(single, 40)
    @test size(U) == (40, 1)
    @test all(0.0 .< U .< 1.0)

    # bad inputs are rejected -
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyVineCopulaModel, (growth_rates = R[:, 1:0],))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyVineCopulaModel, (growth_rates = R[1:1, :],))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyVineCopulaModel, (growth_rates = fill(NaN, 5, 2),))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyGaussianBivariateCopulaModel, (ρ = 1.5,))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MyGumbelBivariateCopulaModel, (θ = 0.5,))
end
