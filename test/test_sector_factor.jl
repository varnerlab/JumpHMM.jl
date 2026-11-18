@testset "sector factor model" begin
    P = inputs["prices"]
    tickers = inputs["tickers"]
    sectors = inputs["sectors"]
    data = (tickers = tickers, prices = P, market = "SPY", sector = sectors, number_of_states = 20, ν = 5.0)
    m = JumpHMM.build(JumpHMM.MySectorFactorModel, data)
    @test m isa JumpHMM.MySectorFactorModel
    @test m.tickers == tickers[2:end]
    @test m.β_scale == zeros(12)

    # given models replace the fitted ones, and give the same fit -
    given = JumpHMM.build(JumpHMM.MySectorFactorModel, merge(data, (
        market_model = m.market_model, sector_models = m.sector_models, own_models = m.own_models)))
    @test given.α == m.α
    @test given.δ == m.δ
    @test given.ψ == m.ψ
    @test given.market_model === m.market_model
    @test given.sector_models[1] === m.sector_models[1]
    @test given.own_models[1] === m.own_models[1]

    # intercept_pull moves each α toward the mean α -
    α̂ = m.α
    ᾱ = mean(α̂)
    pulled = JumpHMM.build(JumpHMM.MySectorFactorModel, merge(data, (
        intercept_pull = 0.5, market_model = m.market_model, sector_models = m.sector_models, own_models = m.own_models)))
    # this checks the formula, so it allows the last bit to differ across Julia versions -
    @test isapprox(pulled.α, ᾱ .+ 0.5 .* (α̂ .- ᾱ); rtol = 1e-12)

    # the model has no centering field. The own draw enters the growth rate as ψ times the draw -
    @test hasfield(JumpHMM.MySectorFactorModel, :ε_center) == false
    D = JumpHMM.sample_draws(m, 20; seed = 3)
    one = JumpHMM.compose_name(m, D, 2)
    β_2 = fill(m.β[2], 20)
    @test one.g == m.α[2] .+ β_2 .* D.g_m .+ m.δ[2] .* D.F[:, m.sector_index[2]] .+ m.ψ[2] .* D.E[:, 2]

    # build rejects a missing market ticker and a sector list of the wrong length -
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MySectorFactorModel, merge(data, (market = "QQQ",)))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MySectorFactorModel, merge(data, (sector = sectors[1:11],)))

    # build rejects a wrong-length β_scale, own_models and sector_models, and a missing required key -
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MySectorFactorModel, merge(data, (β_scale = fill(0.3, 11),)))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MySectorFactorModel, merge(data, (own_models = m.own_models[1:11],)))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MySectorFactorModel, merge(data, (sector_models = m.sector_models[1:end-1],)))
    @test_throws ArgumentError JumpHMM.build(JumpHMM.MySectorFactorModel, Base.structdiff(data, (ν = 5.0,)))

    # the seeds of the draws must stay in range -
    @test_throws ArgumentError JumpHMM.sample_draws(m, 10; seed = 10^8)
    @test_throws ArgumentError JumpHMM.sample_draws(m, 10; seed = -1)

    # a draw assembled from every layer of draws A composes to the same universe as draws A -
    A = JumpHMM.sample_draws(m, 50; seed = 1)
    B = JumpHMM.sample_draws(m, 50; seed = 2)
    composed_A = JumpHMM.compose(m, A)
    assembled = (g_m = A.g_m, episodes = A.episodes, F = A.F, sector_episodes = A.sector_episodes,
        E = A.E, own_episodes = A.own_episodes, Z = A.Z)
    composed_assembled = JumpHMM.compose(m, assembled)
    @test composed_assembled.G == composed_A.G
    @test composed_assembled.β_path == composed_A.β_path
    @test composed_assembled.name_episodes == composed_A.name_episodes

    # a draw that mixes layers from draws A and B composes, with the right shapes -
    mixed = (g_m = A.g_m, m_states = A.m_states, episodes = A.episodes, F = B.F, sector_episodes = B.sector_episodes,
        E = A.E, own_episodes = A.own_episodes, Z = B.Z)
    composed_mixed = JumpHMM.compose(m, mixed)
    @test size(composed_mixed.G) == (50, 12)
    @test size(composed_mixed.β_path) == (50, 12)
    @test size(composed_mixed.name_episodes) == (50, 12)
    @test composed_mixed.G != composed_A.G

    # one name composed alone is the same column of the whole universe -
    @test JumpHMM.compose_name(m, A, 3).g == composed_A.G[:, 3]
    @test_throws ArgumentError JumpHMM.compose_name(m, A, 13)

    # with_parameters makes a new model, shares the fitted models, and leaves its input alone -
    changed = JumpHMM.with_parameters(m; β_scale = fill(0.3, 12))
    @test changed.β_scale == fill(0.3, 12)
    @test changed.ψ == JumpHMM.own_scales(m; β_scale = fill(0.3, 12))
    @test m.ψ == JumpHMM.own_scales(m)
    @test changed.own_models === m.own_models
    @test changed.sector_levels === m.sector_levels
    @test JumpHMM.compose(changed, A).G[:, 5] == JumpHMM.compose_name(m, A, 5; β_scale = 0.3).g

    # with_parameters rejects a negative β_scale entry and a wrong-length α -
    negative = fill(0.3, 12)
    negative[4] = -0.1
    @test_throws ArgumentError JumpHMM.with_parameters(m; β_scale = negative)
    @test_throws ArgumentError JumpHMM.with_parameters(m; α = zeros(11))
end
