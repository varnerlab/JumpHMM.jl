@testset "Copula Models" begin

    @testset "GaussianCopula" begin
        Random.seed!(42)
        # generate correlated data
        Σ_true = [1.0 0.7; 0.7 1.0]
        Z = rand(MvNormal(zeros(2), Σ_true), 1000)'
        returns = Z  # use as "returns"

        gc = fit(GaussianCopula, returns)
        @test size(gc.Σ) == (2, 2)
        @test issymmetric(gc.Σ)
        @test gc.Σ[1, 2] > 0.3  # should capture positive correlation

        U = sample_dependence(gc, 500)
        @test size(U) == (500, 2)
        @test all(0 .≤ U .≤ 1)
    end

    @testset "StudentTCopula" begin
        Random.seed!(42)
        Σ_true = [1.0 0.5; 0.5 1.0]
        Z = rand(MvNormal(zeros(2), Σ_true), 1000)'

        tc = fit(StudentTCopula, Z)
        @test size(tc.Σ) == (2, 2)
        @test tc.ν > 0

        U = sample_dependence(tc, 500)
        @test size(U) == (500, 2)
        @test all(0 .≤ U .≤ 1)
    end

end
