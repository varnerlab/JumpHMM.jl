@testset "sector factor model equals 0.4.1" begin
    P = inputs["prices"]
    G = JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.0, Δt = 1/252)

    # the fitted model -
    sf = JumpHMM.build(JumpHMM.MySectorFactorModel, (
        tickers = inputs["tickers"], prices = P, market = "SPY", sector = inputs["sectors"],
        number_of_states = 20, ν = 5.0))
    @test sf.α == reference["sector/α"]
    @test sf.β == reference["sector/β"]
    @test sf.δ == reference["sector/δ"]
    @test sf.r² == reference["sector/r²"]
    @test sf.ψ == reference["sector/ψ"]
    @test sf.σ²_own == reference["sector/σ²_own"]
    @test sf.σ²_u == reference["sector/σ²_u"]
    @test sf.σ²_m == reference["sector/σ²_m"]
    @test sf.market_model.transition == reference["sector/market_transition"]
    @test sf.sector_models[1].transition == reference["sector/sector_transition_1"]
    @test sf.own_models[1].transition == reference["sector/own_transition_1"]

    # the draws: the episode keys are named "jumps" in the 0.4.1 reference -
    draws = JumpHMM.sample_draws(sf, 500; seed = 11)
    @test collect(keys(draws)) == [:g_m, :m_states, :episodes, :F, :sector_episodes, :E, :own_episodes, :Z]
    @test draws.g_m == reference["sector/draws/g_m"]
    @test draws.m_states == reference["sector/draws/m_states"]
    @test draws.episodes == reference["sector/draws/jumps"]
    @test draws.F == reference["sector/draws/F"]
    @test draws.sector_episodes == reference["sector/draws/sector_jumps"]
    @test draws.E == reference["sector/draws/E"]
    @test draws.own_episodes == reference["sector/draws/own_jumps"]
    @test draws.Z == reference["sector/draws/Z"]

    # the composed universe: name_episodes is named "name_jumps" in the 0.4.1 reference. Version 0.6.0 no longer subtracts
    # the long-run mean ε_center of each own model, so each column of G is the 0.4.1 column plus ψ_j ε_center[j] -
    ε_center = reference["sector/ε_center"]
    composed = JumpHMM.compose(sf, draws)
    @test collect(keys(composed)) == [:G, :β_path, :name_episodes]
    @test centered_columns_match(composed.G, reference["sector/compose/G"], sf.ψ, ε_center)
    @test composed.β_path == reference["sector/compose/β_path"]
    @test composed.name_episodes == reference["sector/compose/name_jumps"]

    # one name, with its own drift scale -
    one_name = JumpHMM.compose_name(sf, draws, 1; β_scale = 0.5)
    σ²_drift = JumpHMM.drift_variance(sf.β_halflife, 0.5) * sf.σ²_m
    ψ_1 = JumpHMM.own_scale(sf.σ²_own[1], sf.σ²_u[1], σ²_drift)
    @test centered_columns_match(reshape(one_name.g, :, 1), reshape(reference["sector/compose_name/g"], :, 1), [ψ_1], ε_center[1:1])
    @test one_name.β_path == reference["sector/compose_name/β_path"]

    # new drift settings -
    changed = JumpHMM.with_parameters(sf; β_scale = fill(0.3, 12), β_halflife = 126.0)
    @test changed.ψ == reference["sector/with/ψ"]

    # one seeded sample of the universe -
    @test centered_columns_match(JumpHMM.sample_dependence(sf, 300; seed = 14).G, reference["sector/sample/G"], sf.ψ, ε_center)

    # the β-drift helpers -
    @test JumpHMM.ema_rate(126.0) == reference["sector/helpers/ema_rate"]
    @test JumpHMM.drift_variance(126.0, 0.3) == reference["sector/helpers/drift_variance"]
    @test JumpHMM.drift_path(G[1:100] ./ 100, 126.0, 0.3) == reference["sector/helpers/drift_path"]
end
