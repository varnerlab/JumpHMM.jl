@testset "build rejects bad inputs" begin
    G = reference["finance/G_spy"]
    T = JumpHMM.MyJumpHiddenMarkovModel
    base = (growth_rates = G, number_of_states = 10, ν = 5.0)

    # each bad input, and the key that the message must name. NaN values and values of the wrong type are rejected too -
    bad = [
        (merge(base, (ϵ = 1.5,)), "ϵ"),
        (merge(base, (ϵ = NaN,)), "ϵ"),
        (merge(base, (λ = 0.0,)), "λ"),
        (merge(base, (λ = NaN,)), "λ"),
        (merge(base, (p_neg = -0.1,)), "p_neg"),
        (merge(base, (p_neg = NaN,)), "p_neg"),
        (merge(base, (ν = 0.0,)), "ν"),
        (merge(base, (ν = NaN,)), "ν"),
        (merge(base, (ν = 5,)), "ν"),
        (merge(base, (N_tail = 0,)), "N_tail"),
        (merge(base, (N_tail = 2.5,)), "N_tail"),
        (merge(base, (min_obs = 2.0,)), "min_obs"),
        (merge(base, (Δt = 1,)), "Δt"),
        (merge(base, (number_of_states = 0,)), "number_of_states"),
        (merge(base, (number_of_states = 10.0,)), "number_of_states"),
        ((number_of_states = 10, ν = 5.0), "growth_rates"),
        (merge(base, (ε = 0.01,)), "ε"),
        (merge(base, (epsilon = 0.01,)), "epsilon"),
        (merge(base, (dt = 1/12,)), "dt"),
    ]
    for (data, key) in bad
        err = try
            JumpHMM.build(T, data)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        if err isa ArgumentError
            @test occursin(key, err.msg)
        end
    end
    m = JumpHMM.build(T, merge(base, (ϵ = 0.2, Δt = 1/12)))
    @test m.ϵ == 0.2 && m.Δt == 1/12 && m.N_tail == 5 && m.p_neg == 0.52

    # the Greek letter ε (U+03B5) is an unknown key, and the message points to ϵ (U+03F5), typed \epsilon -
    err = try
        JumpHMM.build(T, merge(base, (ε = 0.01,)))
        nothing
    catch e
        e
    end
    @test occursin("ϵ (U+03F5)", err.msg)
    @test occursin("\\epsilon", err.msg)
    @test occursin("growth_rates, number_of_states, ν, ϵ, λ, p_neg, N_tail, Δt, min_obs", err.msg)
end

@testset "multi-asset builds reject bad keys and types" begin
    P = inputs["prices"]
    tickers = inputs["tickers"]
    sectors = inputs["sectors"]
    GA = JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252)
    PT = JumpHMM.MyPortfolioModel

    # each bad call, and the key that the message must name -
    bad = [
        (JumpHMM.MyGaussianCopulaModel, (growth_rates = GA[:, 2:3], ν = 5.0), "ν"),
        (JumpHMM.MyGaussianBivariateCopulaModel, (ρ = 0.5, θ = 2.0), "θ"),
        (JumpHMM.MyGaussianBivariateCopulaModel, (ρ = 1,), "ρ"),
        (JumpHMM.MyFrankBivariateCopulaModel, (θ = 2,), "θ"),
        (JumpHMM.MyHybridSingleIndexDependenceModel, (tickers = tickers[1:3], prices = P[:, 1:3], market = "SPY",
            number_of_states = 10, ν = 5.0, copula_type = JumpHMM.MyGaussianCopulaModel), "copula_type"),
        (JumpHMM.MyHybridSingleIndexDependenceModel, (tickers = tickers[1:3], prices = P[:, 1:3], market = "SPY",
            number_of_states = 10, ν = 5.0, copula = JumpHMM.MyGaussianCopulaModel()), "copula"),
        (JumpHMM.MySectorFactorModel, (tickers = tickers, prices = P, market = "SPY", sector = sectors,
            number_of_states = 10, ν = 5.0, β_halflife = 126), "β_halflife"),
        (JumpHMM.MySectorFactorModel, (tickers = tickers, prices = P, market = "SPY", sector = sectors,
            number_of_states = 10, ν = 5.0, β_halflife = 0.0), "β_halflife"),
        (JumpHMM.MySectorFactorModel, (tickers = tickers, prices = P, market = "SPY", sector = sectors,
            number_of_states = 10, ν = 5.0, center_steps = 2.0e4), "center_steps"),
        (JumpHMM.MySectorFactorModel, (tickers = tickers, prices = P, market = "SPY", sectors = sectors,
            number_of_states = 10, ν = 5.0), "sectors"),
        (PT, (tickers = tickers, prices = P, number_of_states = 10, ν = 5.0, dependence = JumpHMM.MyGaussianCopulaModel,
            dt = 1/12), "dt"),
        (PT, (tickers = tickers, prices = P, number_of_states = 10, ν = 5.0, dependence = JumpHMM.MySectorFactorModel,
            market = "SPY", sectors = sectors), "sectors"),
        (PT, (tickers = tickers, prices = P, number_of_states = 10, ν = 5.0, dependence = JumpHMM.MySectorFactorModel,
            market = "SPY", sector = sectors, sector_options = (dt = 1/12,)), "dt"),
        (PT, (tickers = tickers, prices = P, number_of_states = 10, ν = 5.0, dependence = JumpHMM.MySectorFactorModel,
            market = "SPY", sector = sectors, sector_options = (β_halflife = 126,)), "β_halflife"),
        (PT, (tickers = tickers, prices = P, number_of_states = 10, ν = 5, dependence = JumpHMM.MyGaussianCopulaModel), "ν"),
        (PT, (tickers = tickers, prices = P, number_of_states = 10, ν = 5.0, dependence = JumpHMM.MyGaussianCopulaModel()),
            "dependence"),
        (PT, (tickers = tickers[1:3], prices = P, number_of_states = 10, ν = 5.0, dependence = JumpHMM.MyGaussianCopulaModel),
            "tickers has 3 labels but prices has 13 columns"),
    ]
    for (model_type, data, key) in bad
        err = try
            JumpHMM.build(model_type, data)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        if err isa ArgumentError
            @test occursin(key, err.msg)
        end
    end

    # a bivariate range error gives the value -
    err = try
        JumpHMM.build(JumpHMM.MyGumbelBivariateCopulaModel, (θ = 0.5,))
        nothing
    catch e
        e
    end
    @test occursin("0.5", err.msg)

    # the growth rates of laplace_boundaries must be finite -
    @test_throws ArgumentError JumpHMM.laplace_boundaries([0.1, NaN, -0.2], 5)
    @test_throws ArgumentError JumpHMM.laplace_boundaries([0.1, Inf, -0.2], 5)
end
