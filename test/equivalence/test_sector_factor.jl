@testset "sector factor model equals 0.4.1" begin
    P = inputs["prices"]
    G = JumpHMM.excess_growth_rates(P[:, 1]; rf = 0.0, Δt = 1/252)

    # the fitted model -
    sf = JumpHMM.build(JumpHMM.MySectorFactorModel, (
        tickers = inputs["tickers"], prices = P, market = "SPY", sector = inputs["sectors"],
        number_of_states = 20, ν = 5.0, center_steps = 20_000, center_seed = 4 * 10^12))
    @test sf.α == reference["sector/α"]
    @test sf.β == reference["sector/β"]
    @test sf.δ == reference["sector/δ"]
    @test sf.r² == reference["sector/r²"]
    @test sf.ψ == reference["sector/ψ"]
    @test sf.σ²_own == reference["sector/σ²_own"]
    @test sf.σ²_u == reference["sector/σ²_u"]
    @test sf.σ²_m == reference["sector/σ²_m"]
    @test sf.ε_center == reference["sector/ε_center"]
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

    # the composed universe: name_episodes is named "name_jumps" in the 0.4.1 reference -
    composed = JumpHMM.compose(sf, draws)
    @test collect(keys(composed)) == [:G, :β_path, :name_episodes]
    @test composed.G == reference["sector/compose/G"]
    @test composed.β_path == reference["sector/compose/β_path"]
    @test composed.name_episodes == reference["sector/compose/name_jumps"]

    # one name, with its own drift scale -
    one_name = JumpHMM.compose_name(sf, draws, 1; β_scale = 0.5)
    @test one_name.g == reference["sector/compose_name/g"]
    @test one_name.β_path == reference["sector/compose_name/β_path"]

    # new drift settings -
    changed = JumpHMM.with_parameters(sf; β_scale = fill(0.3, 12), β_halflife = 126.0)
    @test changed.ψ == reference["sector/with/ψ"]

    # one seeded sample of the universe -
    @test JumpHMM.sample_dependence(sf, 300; seed = 14).G == reference["sector/sample/G"]

    # the β-drift helpers -
    @test JumpHMM.ema_rate(126.0) == reference["sector/helpers/ema_rate"]
    @test JumpHMM.drift_variance(126.0, 0.3) == reference["sector/helpers/drift_variance"]
    @test JumpHMM.drift_path(G[1:100] ./ 100, 126.0, 0.3) == reference["sector/helpers/drift_path"]
end
