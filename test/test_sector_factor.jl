@testset "SectorFactorModel" begin

    @testset "drift primitives" begin
        @test JumpHMM.ema_rate(1.0) ≈ 0.5
        @test_throws ArgumentError JumpHMM.ema_rate(0.0)
        @test JumpHMM.drift_path(zeros(5), 126.0, 0.3) == zeros(5)
        Random.seed!(7); z = randn(400_000)
        x = JumpHMM.drift_path(z, 126.0, 0.3)
        @test isapprox(var(x), JumpHMM.drift_variance(126.0, 0.3); rtol=0.05)
        @test JumpHMM.drift_path(z, 126.0, 0.6) ≈ 2 .* x            # linear in k
    end

    @testset "sector factors and loadings" begin
        Random.seed!(11); T = 5000
        f = randn(T, 2)                                    # two true sector factors
        sector = ["A", "A", "A", "B", "B"]
        δtrue = [0.8, 1.0, 1.2, 0.9, 1.1]
        E = hcat([δtrue[i] .* f[:, sector[i] == "A" ? 1 : 2] .+ 0.5 .* randn(T) for i in 1:5]...)
        sf = JumpHMM.sector_factors(E, sector)
        @test sf.sectors == ["A", "B"]
        @test sf.F[:, 1] ≈ vec(mean(E[:, 1:3]; dims=2))
        δ = JumpHMM.sector_loadings(E, sector)
        # generator co-movement matches the real mean pairwise covariance in each sector
        for (s, name) in enumerate(sf.sectors)
            m = findall(==(name), sector); n = length(m)
            C = cov(E[:, m]); real_cov = (sum(C) - sum(diag(C))) / (n * (n - 1))
            model_cov = (sum(δ[m])^2 - sum(δ[m] .^ 2)) * var(sf.F[:, s]) / (n * (n - 1))
            @test isapprox(model_cov, real_cov; rtol=1e-8)
        end
    end

end
