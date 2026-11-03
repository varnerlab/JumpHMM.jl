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

    @testset "compose_name" begin
        T = 4; α = fill(0.1, T); β = fill(1.2, T)
        g_m = [1.0, -2.0, 0.5, 0.0]; f = [0.3, 0.0, -0.1, 0.2]; ε = [0.5, 0.5, 1.5, -0.5]
        out = JumpHMM.compose_name(α, β, 0.5, g_m, f, ε, 1.0, 1.0, 4.0, 0.3; ε_center=0.5, β²_mean=1.44)
        s² = 1 - (1.44 + 0.25) / 4.0
        @test out.g ≈ 0.1 .+ 1.2 .* g_m .+ 0.5 .* f .+ sqrt(s²) .* (ε .- 0.5)
        @test out.n_clipped == 0 && out.β_eff == β
        # factor share above 1 - floor: loadings shrink, own scale = floor
        hi = JumpHMM.compose_name(α, fill(3.0, T), 0.5, g_m, f, ε, 1.0, 1.0, 4.0, 0.3; ε_center=0.0, β²_mean=9.0)
        c = sqrt(0.9 / ((9.0 + 0.25) / 4.0))
        @test hi.n_clipped == T && hi.β_eff ≈ fill(3.0c, T)
        @test hi.g ≈ 0.1 .+ 3.0c .* g_m .+ 0.5c .* f .+ sqrt(0.10) .* ε
        # high real R²: keep the real R² (own-noise share from v̄(1 − r²)/r²)
        r2 = JumpHMM.compose_name(α, β, 0.5, g_m, f, ε, 1.0, 1.0, 4.0, 0.9; ε_center=0.0, β²_mean=1.44)
        @test r2.g ≈ 0.1 .+ 1.2 .* g_m .+ 0.5 .* f .+ sqrt((1.44 + 0.25) * (1 - 0.9) / 0.9 / 4.0) .* ε
        @test_throws ArgumentError JumpHMM.compose_name(α, β, 0.5, g_m, f[1:3], ε, 1.0, 1.0, 4.0, 0.3; ε_center=0.0, β²_mean=1.44)
    end

end
