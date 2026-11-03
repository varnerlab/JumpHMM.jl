# synthetic universe for SectorFactorModel tests: market MKT, sectors S1 (A1..A3) and S2 (B1..B3)
function sector_universe(; T::Int=3000, seed::Int=3, flip_S2::Bool=false)
    rng = MersenneTwister(seed); dt = 1/252
    g_m = 0.08 .+ 2.8 .* randn(rng, T)                 # annualized market growth
    f = 1.0 .* randn(rng, T, 2)                         # sector factors
    β = [0.8, 1.0, 1.2, 0.9, 1.1, 1.3]; δ = [0.8, 1.0, 1.2, 0.9, 1.1, 1.0]
    sector = ["S1", "S1", "S1", "S2", "S2", "S2"]
    G = zeros(T, 6)
    for i in 1:6
        s = sector[i] == "S1" ? 1 : 2
        G[:, i] = 0.02 .+ β[i] .* g_m .+ δ[i] .* f[:, s] .+ 2.0 .* randn(rng, T)
    end
    if flip_S2                                          # S2 residuals: B2 = −B1, B3 independent
        u = 2.0 .* randn(rng, T)
        G[:, 4] = 0.02 .+ β[4] .* g_m .+ u
        G[:, 5] = 0.02 .+ β[5] .* g_m .- u
        G[:, 6] = 0.02 .+ β[6] .* g_m .+ 2.0 .* randn(rng, T)
    end
    prices = 100.0 .* exp.(vcat(zeros(1, 7), cumsum(hcat(g_m, G) .* dt; dims=1)))
    return (["MKT", "A1", "A2", "A3", "B1", "B2", "B3"], prices, sector)
end

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

    @testset "fit SectorFactorModel" begin
        tickers, prices, sector = sector_universe()
        model, marginals = fit(SectorFactorModel, tickers, prices, "MKT", sector; N=10, center_steps=20_000)
        @test model.tickers == tickers[2:end] && model.sectors == ["S1", "S2"]
        @test all(isapprox.(model.β, [0.8, 1.0, 1.2, 0.9, 1.1, 1.3]; atol=0.1))
        @test model.β_scale == zeros(6) && sort(collect(keys(marginals))) == sort(model.tickers)
        @test length(model.sector_models) == 2 && size(model.sector_levels, 2) == 2
        @test length(model.name_models) == 6
        # intercept pull 0 sets every α to the universe mean
        m0, _ = fit(SectorFactorModel, tickers, prices, "MKT", sector; N=10, center_steps=20_000, intercept_pull=0.0)
        @test all(m0.α .≈ mean(m0.α))
        # review focus 1: a sector with one name
        @test_throws ArgumentError fit(SectorFactorModel, tickers[1:5], prices[:, 1:5], "MKT", ["S1", "S1", "S1", "S2"]; N=10)
        # review focus 2: non-positive mean pairwise residual covariance names the sector
        tf, pf, sf = sector_universe(; flip_S2=true)
        err = try
            fit(SectorFactorModel, tf, pf, "MKT", sf; N=10, center_steps=20_000); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("S2", err.msg)
    end

    @testset "PortfolioModel with SectorFactorModel" begin
        tickers, prices, sector = sector_universe()
        opts = (center_steps=20_000,)
        pf = fit(PortfolioModel, tickers, prices; dependence=SectorFactorModel, market="MKT", sectors=sector,
                 N=10, sector_options=opts)
        @test pf.dependence isa SectorFactorModel && pf.market_ticker == "MKT"
        # review focus 4: same seed, same paths; market path returned; different seed, different paths
        r1 = simulate(pf, 250; n_paths=3, seed=5); r2 = simulate(pf, 250; n_paths=3, seed=5)
        @test r1.results["A1"].paths[2].observations == r2.results["A1"].paths[2].observations
        @test haskey(r1.results, "MKT") && length(r1.results["MKT"].paths) == 3
        @test r1.results["A1"].paths[1].observations != simulate(pf, 250; n_paths=1, seed=6).results["A1"].paths[1].observations
        # same-day dependence: within-sector correlation above cross-sector correlation on a long path
        d = JumpHMM.sample_dependence(pf.dependence, 20_000; seed=9)
        C = cor(d.G)
        @test mean([C[1, 2], C[1, 3], C[2, 3]]) > mean([C[i, j] for i in 1:3 for j in 4:6]) + 0.05
        # variance target on a long path (fixed β, unclipped names)
        @test all(isapprox.(vec(var(d.G; dims=1)), pf.dependence.σ²_gen; rtol=0.15))
        # a drifting β changes the composed β path day by day
        pd = fit(PortfolioModel, tickers, prices; dependence=SectorFactorModel, market="MKT", sectors=sector,
                 N=10, sector_options=(center_steps=20_000, β_scale=fill(0.3, 6)))
        dd = JumpHMM.sample_dependence(pd.dependence, 500; seed=9)
        @test std(dd.β_eff[:, 1]) > 0 && length(unique(d.β_eff[:, 1])) == 1
        # review focus 5: tune touches market, sectors, names
        pt = tune(pf, prices; ϵ_range=range(1e-4, 1e-2, length=2), λ_range=range(10.0, 20.0, length=2), n_paths=4)
        @test pt.dependence isa SectorFactorModel && length(pt.dependence.sector_models) == 2
        @test pt.dependence.market_model.jump.ϵ > 0
        @test all(m.jump.ϵ > 0 for m in pt.dependence.sector_models)
        @test all(pt.marginals[t].jump.ϵ > 0 for t in pt.tickers)
        @test_throws ArgumentError fit(PortfolioModel, tickers, prices; dependence=SectorFactorModel, market="MKT", N=10)
    end

    @testset "final review fixes" begin
        tickers, prices, sector = sector_universe()
        pf = fit(PortfolioModel, tickers, prices; dependence=SectorFactorModel, market="MKT", sectors=sector,
                 N=10, sector_options=(center_steps=20_000,))
        dep = pf.dependence
        # 1: runs with nearby seeds share no paths
        r10 = simulate(pf, 50; n_paths=3, seed=10); r11 = simulate(pf, 50; n_paths=3, seed=11)
        @test all(a.observations != b.observations for a in r10.results["A1"].paths for b in r11.results["A1"].paths)
        @test length(unique([p.observations for p in r10.results["A1"].paths])) == 3
        # 2: seed range guard (stream offsets assume 0 ≤ seed < 10^8)
        @test_throws ArgumentError JumpHMM.sample_dependence(dep, 10; seed=10^8)
        @test_throws ArgumentError JumpHMM.sample_dependence(dep, 10; seed=-1)
        # 3: clipping is reported per name
        d = JumpHMM.sample_dependence(dep, 100; seed=3)
        @test d.n_clipped == zeros(Int, 6)
        vals = [f === :β ? vcat(10.0, dep.β[2:end]) : getfield(dep, f) for f in fieldnames(SectorFactorModel)]
        big = SectorFactorModel(vals...)
        dbig = JumpHMM.sample_dependence(big, 100; seed=3)
        @test dbig.n_clipped[1] == 100 && all(dbig.n_clipped[2:end] .== 0)
    end

    @testset "codex audit fixes" begin
        tickers, prices, sector = sector_universe()
        pf = fit(PortfolioModel, tickers, prices; dependence=SectorFactorModel, market="MKT", sectors=sector,
                 N=10, sector_options=(center_steps=20_000,))
        dep = pf.dependence
        # centering draws use seeds far from any path seed (reference: 4·10^12 + j)
        @test dep.center_seed == 4 * 10^12 && dep.center_steps == 20_000
        # tune recomputes each name's noise center from the tuned model
        pt = tune(pf, prices; ϵ_range=range(5e-3, 1e-2, length=2), λ_range=range(10.0, 20.0, length=2), n_paths=4)
        dt = pt.dependence
        for i in eachindex(dt.tickers)
            expected = mean(simulate(dt.name_models[i], dt.center_steps; n_paths=1, seed=dt.center_seed + i).paths[1].observations)
            @test dt.ε_center[i] ≈ expected
        end
        # a name's jump flags mark days any of its components (market, sector, own) was in a jump episode
        d = JumpHMM.sample_dependence(dt, 2_000; seed=4)
        @test size(d.name_jumps) == (2_000, 6)
        @test all(all(d.name_jumps[:, j] .>= d.jumps) for j in 1:6)
        @test any(d.name_jumps[:, 1] .& .!d.jumps)              # own or sector episodes add days
        r = simulate(pt, 300; n_paths=1, seed=2)
        @test r.results["A1"].paths[1].jumps !== r.results["MKT"].paths[1].jumps
    end

end
