@testset "Vine Copula" begin

    @testset "Bivariate copula h-function / hinv consistency" begin
        Random.seed!(42)
        copulas = [
            JumpHMM.GaussianBiCopula(0.5),
            JumpHMM.StudentTBiCopula(0.5, 5.0),
            JumpHMM.ClaytonBiCopula(2.0),
            JumpHMM.GumbelBiCopula(2.0),
            JumpHMM.FrankBiCopula(5.0),
        ]
        for cop in copulas
            for _ in 1:20
                u = rand() * 0.8 + 0.1  # avoid extremes
                v = rand() * 0.8 + 0.1
                h = JumpHMM._bicop_hfunc(cop, u, v)
                u_recovered = JumpHMM._bicop_hinv(cop, h, v)
                @test u_recovered ≈ u atol=1e-4
            end
        end
    end

    @testset "Bivariate copula fitting" begin
        Random.seed!(42)
        n = 1000
        # generate Gaussian copula data with known ρ
        ρ_true = 0.6
        Z = rand(MvNormal([1.0 ρ_true; ρ_true 1.0]), n)'
        u = cdf.(Normal(), Z[:, 1])
        v = cdf.(Normal(), Z[:, 2])

        cop = JumpHMM._bicop_fit(GaussianBiCopula, u, v)
        @test abs(cop.ρ - ρ_true) < 0.15
    end

    @testset "AIC family selection" begin
        Random.seed!(42)
        n = 1000
        # Gaussian data should select Gaussian (or similar symmetric)
        ρ = 0.5
        Z = rand(MvNormal([1.0 ρ; ρ 1.0]), n)'
        u = cdf.(Normal(), Z[:, 1])
        v = cdf.(Normal(), Z[:, 2])

        cop = JumpHMM._select_bicop(u, v)
        # should select a symmetric family (Gaussian, Frank, or StudentT)
        @test cop isa Union{GaussianBiCopula, StudentTBiCopula, FrankBiCopula}
    end

    @testset "VineCopula fit and sample" begin
        Random.seed!(42)
        # 4-asset correlated data
        Σ = [1.0 0.7 0.3 0.1;
             0.7 1.0 0.5 0.2;
             0.3 0.5 1.0 0.4;
             0.1 0.2 0.4 1.0]
        Z = rand(MvNormal(zeros(4), Σ), 500)'

        vc = fit(VineCopula, Z)
        @test vc.d == 4
        @test length(vc.order) == 4
        @test length(vc.edges) == 3  # d-1 tree levels

        # sample
        U = sample_dependence(vc, 500)
        @test size(U) == (500, 4)
        @test all(0 .≤ U .≤ 1)
    end

    @testset "VineCopula preserves correlation structure" begin
        Random.seed!(42)
        Σ = [1.0 0.6; 0.6 1.0]
        Z = rand(MvNormal(zeros(2), Σ), 2000)'

        vc = fit(VineCopula, Z)
        U = sample_dependence(vc, 2000)

        # rank correlation should be positive and non-trivial
        ρ_rank = corspearman(U[:, 1], U[:, 2])
        @test ρ_rank > 0.2
    end

    @testset "VineCopula in PortfolioModel" begin
        Random.seed!(42)
        n = 300
        Σ = [1.0 0.5; 0.5 1.0]
        Z = rand(MvNormal(zeros(2), Σ), n)'
        prices = 100.0 .* exp.(cumsum(Z .* 0.01, dims=1))

        tickers = ["A", "B"]
        portfolio = fit(PortfolioModel, tickers, prices;
                        dependence=VineCopula, N=10, ν=5.0)

        @test portfolio.dependence isa VineCopula
        @test haskey(portfolio.marginals, "A")

        result = simulate(portfolio, 50; n_paths=10, seed=42)
        @test result isa PortfolioSimulationResult
        @test length(result.results["A"].paths) == 10
    end

end
