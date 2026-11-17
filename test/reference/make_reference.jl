# make_reference.jl
#
# Freezes what JumpHMM 0.4.1 returns on the fixed inputs in inputs.jld2, so that later versions can be
# checked against these numbers with `==`. Every value written is a plain Array, number, Bool, or String
# (no JumpHMM types). Keys use "/" to group related values.
#
# Run: julia --project=test/reference test/reference/make_reference.jl [inputs.jld2] [output.jld2]

using JumpHMM
using JLD2
using Random
using Statistics
using TOML

const DEFAULT_INPUTS = joinpath(@__DIR__, "inputs.jld2")
const DEFAULT_OUTPUT = joinpath(@__DIR__, "reference-0.4.1.jld2")

# ---------------------------------------------------------------------------
# Helpers that turn 0.4.1 objects into plain arrays
# ---------------------------------------------------------------------------

"""
    path_matrix(result, field, T) -> Matrix{T}

Stack one field of every path in a `SimulationResult` into a days x paths matrix.
`field` is `:observations`, `:states`, or `:jumps`.
"""
function path_matrix(result::SimulationResult, field::Symbol, T::DataType)
    n_paths = length(result.paths)
    n_steps = length(getfield(result.paths[1], field))
    out = Matrix{T}(undef, n_steps, n_paths)
    for p in 1:n_paths
        out[:, p] = getfield(result.paths[p], field)
    end
    return out
end

"""
    add_simulation!(ref, prefix, result)

Store the growth rates, states, and jump episodes of a simulation as days x paths matrices.
"""
function add_simulation!(ref::Dict{String,Any}, prefix::String, result::SimulationResult)
    ref[prefix * "/growth_rates"] = path_matrix(result, :observations, Float64)
    ref[prefix * "/states"] = path_matrix(result, :states, Int)
    ref[prefix * "/episodes"] = path_matrix(result, :jumps, Bool)
end

"""
    add_model!(ref, prefix, model, G)

Store the fitted fields of a `JumpHiddenMarkovModel`, and the states `G` is assigned to.
"""
function add_model!(ref::Dict{String,Any}, prefix::String, model::JumpHiddenMarkovModel,
                    G::Vector{Float64})
    ref[prefix * "/laplace_μ"] = model.partition.μ
    ref[prefix * "/laplace_b"] = model.partition.b
    ref[prefix * "/boundaries"] = copy(model.partition.boundaries)
    ref[prefix * "/states"] = assign_states(model.partition, G)
    ref[prefix * "/transition"] = copy(model.transition)
    ref[prefix * "/stationary"] = copy(model.stationary)
    ref[prefix * "/emission_μ"] = [e.μ for e in model.emissions]
    ref[prefix * "/emission_σ"] = [e.σ for e in model.emissions]
    ref[prefix * "/emission_n"] = [e.n_obs for e in model.emissions]
    ref[prefix * "/emission_fallback"] = [e.is_fallback for e in model.emissions]
end

"""
    with_jump(model, ϵ, λ) -> JumpHiddenMarkovModel

Copy `model` with new jump settings, using the 0.4.1 positional constructor.
"""
function with_jump(model::JumpHiddenMarkovModel, ϵ::Float64, λ::Float64)
    return JumpHiddenMarkovModel(model.partition, model.transition, model.emissions,
                                 model.stationary, JumpParameters(ϵ, λ), model.ν,
                                 model.rf, model.dt)
end

"""
    add_report!(ref, prefix, report)

Store a `ValidationReport`: one vector per `PathTestResult` field and one number per summary field.
"""
function add_report!(ref::Dict{String,Any}, prefix::String, report::ValidationReport)
    results = report.path_results
    ref[prefix * "/ks_pvalue"] = [r.ks_pvalue for r in results]
    ref[prefix * "/ad_pvalue"] = [r.ad_pvalue for r in results]
    ref[prefix * "/wasserstein"] = [r.wasserstein for r in results]
    ref[prefix * "/hellinger"] = [r.hellinger for r in results]
    ref[prefix * "/acf_mae"] = [r.acf_mae for r in results]
    ref[prefix * "/kurtosis_sim"] = [r.kurtosis_sim for r in results]
    ref[prefix * "/kurtosis_emp"] = results[1].kurtosis_emp
    ref[prefix * "/ks_pass_rate"] = report.ks_pass_rate
    ref[prefix * "/ad_pass_rate"] = report.ad_pass_rate
    ref[prefix * "/mean_acf_mae"] = report.mean_acf_mae
    ref[prefix * "/mean_wasserstein"] = report.mean_wasserstein
    ref[prefix * "/mean_hellinger"] = report.mean_hellinger
    ref[prefix * "/mean_kurtosis"] = report.mean_kurtosis
    ref[prefix * "/α"] = report.α
end

"""
    family_label(copula) -> String

Name the family of a bivariate copula: "gaussian", "studentt", "clayton", "gumbel", or "frank".
"""
function family_label(copula::AbstractBivariateCopula)
    if copula isa GaussianBiCopula
        return "gaussian"
    elseif copula isa StudentTBiCopula
        return "studentt"
    elseif copula isa ClaytonBiCopula
        return "clayton"
    elseif copula isa GumbelBiCopula
        return "gumbel"
    elseif copula isa FrankBiCopula
        return "frank"
    end
    error("unknown bivariate copula family $(typeof(copula))")
end

"""
    family_parameters(copula) -> Vector{Float64}

Parameters of a bivariate copula: ρ for gaussian; ρ then ν for studentt; θ for the others.
"""
function family_parameters(copula::AbstractBivariateCopula)
    if copula isa GaussianBiCopula
        return [copula.ρ]
    elseif copula isa StudentTBiCopula
        return [copula.ρ, copula.ν]
    end
    return [copula.θ]
end

"""
    plain_array(value) -> Array

Copy a draw field into a plain array (a BitMatrix becomes a Matrix{Bool}).
"""
function plain_array(value::AbstractArray)
    if eltype(value) == Bool
        return Array{Bool}(value)
    end
    return Array(value)
end

"""
    jumphmm_commit(project_path) -> String

Read the pinned 0.4.1 commit from the `[sources]` table of the reference Project.toml.
"""
function jumphmm_commit(project_path::String)
    project = TOML.parsefile(project_path)
    return String(project["sources"]["JumpHMM"]["rev"])
end

# ---------------------------------------------------------------------------
# One function per group of keys
# ---------------------------------------------------------------------------

"""
    add_finance!(ref, P, G, GA)

Growth rates and the price rebuild.
"""
function add_finance!(ref::Dict{String,Any}, P::Matrix{Float64}, G::Vector{Float64},
                      GA::Matrix{Float64})
    ref["finance/G_spy"] = G
    ref["finance/G_all"] = GA
    ref["finance/G_rf"] = excess_growth_rates(P[:, 1]; rf=0.05, dt=1/252)
    ref["finance/prices_back"] = prices_from_growth_rates(G, P[1, 1]; rf=0.0, dt=1/252)
end

"""
    add_fits!(ref, P, G, m0)

The full-sample fit and the short-sample fit.
"""
function add_fits!(ref::Dict{String,Any}, P::Matrix{Float64}, G::Vector{Float64},
                   m0::JumpHiddenMarkovModel)
    add_model!(ref, "fit", m0, G)
    G_short = excess_growth_rates(P[1:301, 1]; rf=0.0, dt=1/252)
    m_short = fit(JumpHiddenMarkovModel, P[1:301, 1]; N=100, ν=5.0)
    add_model!(ref, "fit_short", m_short, G_short)
end

"""
    add_simulations!(ref, m0, mj)

Single-asset simulations: with jumps, without jumps, no jumps drawn (k0), overrun, and a fixed start state.
"""
function add_simulations!(ref::Dict{String,Any}, m0::JumpHiddenMarkovModel,
                          mj::JumpHiddenMarkovModel)
    add_simulation!(ref, "simulate/jump", simulate(mj, 2766; n_paths=20, seed=5660))
    add_simulation!(ref, "simulate/nojump", simulate(m0, 2766; n_paths=20, seed=5660))
    k0 = with_jump(m0, 0.05, 0.5)
    add_simulation!(ref, "simulate/k0", simulate(k0, 500; n_paths=10, seed=1))
    overrun = with_jump(m0, 0.05, 90.0)
    add_simulation!(ref, "simulate/overrun", simulate(overrun, 50; n_paths=10, seed=2))
    add_simulation!(ref, "simulate/start", simulate(mj, 100; n_paths=5, start=50, seed=3))
end

"""
    add_tuning!(ref, P, m0)

The jump-parameter search on the full sample, and on a short sample where no candidate produces a jump.
"""
function add_tuning!(ref::Dict{String,Any}, P::Matrix{Float64}, m0::JumpHiddenMarkovModel)
    tuned = tune(m0, P[:, 1]; ϵ_range=range(1e-4, 5e-3, length=3),
                 λ_range=range(10.0, 160.0, length=3), n_paths=20, seed=5660)
    ref["tune/ϵ"] = tuned.jump.ϵ
    ref["tune/λ"] = tuned.jump.λ
    m_small = fit(JumpHiddenMarkovModel, P[1:201, 1]; N=20)
    tuned_none = tune(m_small, P[1:201, 1]; ϵ_range=[1e-6], λ_range=[10.0], n_paths=2, seed=1)
    ref["tune_none/ϵ"] = tuned_none.jump.ϵ
    ref["tune_none/λ"] = tuned_none.jump.λ
end

"""
    add_validation!(ref, P, GA, mj)

The validation report and the two distances it uses.
"""
function add_validation!(ref::Dict{String,Any}, P::Matrix{Float64}, GA::Matrix{Float64},
                         mj::JumpHiddenMarkovModel)
    add_report!(ref, "validate", validate(mj, P[:, 1]; n_paths=20, seed=9))
    ref["distance/wasserstein"] = JumpHMM._wasserstein1(GA[:, 2], GA[:, 3])
    ref["distance/hellinger"] = JumpHMM._hellinger(GA[:, 2], GA[:, 3])
end

"""
    add_decoding!(ref, G, mj)

Viterbi states, forward-filter output, and log-likelihood on the first 500 growth rates.
"""
function add_decoding!(ref::Dict{String,Any}, G::Vector{Float64}, mj::JumpHiddenMarkovModel)
    ref["decode/states"] = decode(mj, G[1:500])
    ref["decode/filter"] = forward_filter(mj, G[1:500])
    ref["decode/loglik"] = log_likelihood(mj, G[1:500])
end

"""
    add_copulas!(ref, GA)

Gaussian, Student-t, and vine copulas fit to four stocks, and 200 draws from each.
"""
function add_copulas!(ref::Dict{String,Any}, GA::Matrix{Float64})
    returns = GA[:, 2:5]

    gc = fit(GaussianCopula, returns)
    Random.seed!(7)
    ref["copula/gaussian/Σ"] = copy(gc.Σ)
    ref["copula/gaussian/U"] = sample_dependence(gc, 200)

    tc = fit(StudentTCopula, returns)
    Random.seed!(8)
    ref["copula/studentt/Σ"] = copy(tc.Σ)
    ref["copula/studentt/ν"] = tc.ν
    ref["copula/studentt/U"] = sample_dependence(tc, 200)

    vc = fit(VineCopula, returns)
    families = String[]
    parameters = Float64[]
    for tree in vc.edges
        for edge in tree
            push!(families, family_label(edge.copula))
            append!(parameters, family_parameters(edge.copula))
        end
    end
    Random.seed!(10)
    ref["vine/order"] = copy(vc.order)
    ref["vine/families"] = families
    ref["vine/parameters"] = parameters
    ref["vine/U"] = sample_dependence(vc, 200)

    add_bicops!(ref, GA)
end

"""
    add_bicops!(ref, GA)

Each bivariate copula family fit to a positive-dependence pair (`pos`) and a negative-dependence pair (`neg`) of
pseudo-observations, with its log-density, h-function, and inverse h-function on a 5 x 5 grid, and the family that AIC selects.
"""
function add_bicops!(ref::Dict{String,Any}, GA::Matrix{Float64})
    pits = JumpHMM._pit(GA[:, 2:3])
    u0 = pits[:, 1]
    v0 = pits[:, 2]
    cases = Dict("pos" => (u0, v0), "neg" => (u0, 1.0 .- v0))
    families = [("gaussian", JumpHMM.GaussianBiCopula), ("studentt", JumpHMM.StudentTBiCopula),
                ("clayton", JumpHMM.ClaytonBiCopula), ("gumbel", JumpHMM.GumbelBiCopula),
                ("frank", JumpHMM.FrankBiCopula)]
    grid = [0.05, 0.25, 0.5, 0.75, 0.95]
    for case in ["pos", "neg"]
        u, v = cases[case]
        for (label, family) in families
            prefix = "bicop/" * label * "/" * case
            c = JumpHMM._bicop_fit(family, u, v)
            ref[prefix * "/parameters"] = family_parameters(c)
            logpdf = Float64[]
            hfunc = Float64[]
            hinv = Float64[]
            for a in grid
                for b in grid
                    push!(logpdf, JumpHMM._bicop_logpdf(c, a, b))
                    push!(hfunc, JumpHMM._bicop_hfunc(c, a, b))
                    push!(hinv, JumpHMM._bicop_hinv(c, a, b))
                end
            end
            ref[prefix * "/logpdf"] = logpdf
            ref[prefix * "/hfunc"] = hfunc
            ref[prefix * "/hinv"] = hinv
        end
        ref["bicop/select/" * case] = family_label(JumpHMM._select_bicop(u, v))
    end
end

"""
    add_single_index!(ref, P, GA, tickers)

The single-index model with both residual methods, and the hybrid single-index model.
"""
function add_single_index!(ref::Dict{String,Any}, P::Matrix{Float64}, GA::Matrix{Float64},
                           tickers::Vector{String})
    methods = [:bootstrap, :gaussian]
    for method in methods
        sim = fit(SingleIndexModel, GA[:, 2:13], GA[:, 1]; market_prices=P[:, 1], N=20,
                  residual_method=method)
        prefix = "sim/" * String(method)
        ref[prefix * "/α"] = copy(sim.α)
        ref[prefix * "/β"] = copy(sim.β)
        ref[prefix * "/σ_ε"] = copy(sim.σ_ε)
        Random.seed!(12)
        ref[prefix * "/sample"] = sample_dependence(sim, 200)
    end

    h = fit(HybridSingleIndexModel, tickers, P, "SPY"; N=20)
    ref["hybrid/α"] = copy(h.α)
    ref["hybrid/β"] = copy(h.β)
    ref["hybrid/r²"] = copy(h.r²)
    ref["hybrid/σ_market"] = h.σ_market
    Random.seed!(13)
    ref["hybrid/sample"] = sample_dependence(h, 200)

    # a second hybrid fit whose settings send tickers through the R2-preserving branch and the beta clip
    hb = fit(HybridSingleIndexModel, tickers, P, "SPY"; N=20,
             r2_preserve_threshold=0.45, idiosyncratic_floor=0.7)
    ref["hybrid_branches/α"] = copy(hb.α)
    ref["hybrid_branches/β"] = copy(hb.β)
    ref["hybrid_branches/r²"] = copy(hb.r²)
    ref["hybrid_branches/σ_market"] = hb.σ_market
    Random.seed!(14)
    ref["hybrid_branches/sample"] = sample_dependence(hb, 200)
    report_hybrid_branches(hb)
end

"""
    report_hybrid_branches(h)

Print which branch of the hybrid sample each ticker takes: preserve (r² at or above the threshold), clip (beta lowered because
rho = beta^2 * sigma_market^2 / var(marginal) is above 1 - floor), or plain. The marginal variance is approximated by a long simulation.
"""
function report_hybrid_branches(h)
    saved = copy(Random.default_rng())
    Random.seed!(99)
    for k in 1:length(h.tickers)
        obs = simulate(h.marginals[h.tickers[k]], 20000; n_paths=1).paths[1].observations
        ρ = h.β[k]^2 * h.σ_market^2 / Statistics.var(obs)
        branch = "plain"
        if h.r²[k] >= h.r2_preserve_threshold
            branch = "preserve"
        elseif ρ > 1.0 - h.idiosyncratic_floor
            branch = "clip"
        end
        println("hybrid_branches ", h.tickers[k], " r²=", round(h.r²[k]; digits=3), " rho=", round(ρ; digits=3), " ", branch)
    end
    copy!(Random.default_rng(), saved)
end

"""
    add_sector_model!(ref, G, sf)

Fitted fields of the sector-factor model, its draws, composition, parameter changes, and helpers.
"""
function add_sector_model!(ref::Dict{String,Any}, G::Vector{Float64}, sf::SectorFactorModel)
    ref["sector/α"] = copy(sf.α)
    ref["sector/β"] = copy(sf.β)
    ref["sector/δ"] = copy(sf.δ)
    ref["sector/r²"] = copy(sf.r²)
    ref["sector/ψ"] = copy(sf.ψ)
    ref["sector/σ²_own"] = copy(sf.σ²_own)
    ref["sector/σ²_u"] = copy(sf.σ²_u)
    ref["sector/σ²_m"] = sf.σ²_m
    ref["sector/ε_center"] = copy(sf.ε_center)
    ref["sector/market_transition"] = copy(sf.market_model.transition)
    ref["sector/sector_transition_1"] = copy(sf.sector_models[1].transition)
    ref["sector/own_transition_1"] = copy(sf.own_models[1].transition)

    draws = JumpHMM.sample_draws(sf, 500; seed=11)
    for field in keys(draws)
        ref["sector/draws/" * String(field)] = plain_array(getfield(draws, field))
    end

    composed = JumpHMM.compose(sf, draws)
    ref["sector/compose/G"] = composed.G
    ref["sector/compose/β_path"] = composed.β_path
    ref["sector/compose/name_jumps"] = plain_array(composed.name_jumps)

    one_name = JumpHMM.compose_name(sf, draws, 1; β_scale=0.5)
    ref["sector/compose_name/g"] = one_name.g
    ref["sector/compose_name/β_path"] = one_name.β_path

    changed = JumpHMM.with_parameters(sf; β_scale=fill(0.3, 12), β_halflife=126.0)
    ref["sector/with/ψ"] = copy(changed.ψ)

    ref["sector/sample/G"] = JumpHMM.sample_dependence(sf, 300; seed=14).G

    ref["sector/helpers/ema_rate"] = JumpHMM.ema_rate(126.0)
    ref["sector/helpers/drift_variance"] = JumpHMM.drift_variance(126.0, 0.3)
    ref["sector/helpers/drift_path"] = JumpHMM.drift_path(G[1:100] ./ 100, 126.0, 0.3)
end

"""
    add_portfolio_simulation!(ref, prefix, result)

Store each ticker's simulation in a `PortfolioSimulationResult`.
"""
function add_portfolio_simulation!(ref::Dict{String,Any}, prefix::String,
                                   result::PortfolioSimulationResult)
    for ticker in result.tickers
        add_simulation!(ref, prefix * "/" * ticker, result.results[ticker])
    end
end

"""
    add_portfolios!(ref, P, tickers, sectors)

The sector-factor portfolio (fit, tune, simulate) and the Gaussian-copula portfolio (fit, simulate, validate).
The sector-factor simulation also stores the market paths under `portfolio/simulate/SPY`, which `result.tickers` leaves out.
"""
function add_portfolios!(ref::Dict{String,Any}, P::Matrix{Float64}, tickers::Vector{String},
                         sectors::Vector{String})
    pf = fit(PortfolioModel, tickers, P; dependence=SectorFactorModel, market="SPY",
             sectors=sectors, N=20, sector_options=(center_steps=20_000,))
    pt = tune(pf, P; ϵ_range=[1e-4, 3e-3], λ_range=[10.0, 100.0], n_paths=5, seed=15)
    dep = pt.dependence
    ref["portfolio/tuned/sector_ϵ"] = [m.jump.ϵ for m in dep.sector_models]
    ref["portfolio/tuned/sector_λ"] = [m.jump.λ for m in dep.sector_models]
    ref["portfolio/tuned/own_ϵ"] = [m.jump.ϵ for m in dep.own_models]
    ref["portfolio/tuned/own_λ"] = [m.jump.λ for m in dep.own_models]
    sector_paths = simulate(pt, 300; n_paths=3, seed=16)
    add_portfolio_simulation!(ref, "portfolio/simulate", sector_paths)
    add_simulation!(ref, "portfolio/simulate/SPY", sector_paths.results["SPY"])

    pg = fit(PortfolioModel, tickers[1:5], P[:, 1:5]; dependence=GaussianCopula, N=20)
    add_portfolio_simulation!(ref, "portfolio_gauss/simulate", simulate(pg, 200; n_paths=2, seed=17))
    reports = validate(pg, P[:, 1:5]; n_paths=3, seed=18)
    for ticker in pg.tickers
        add_report!(ref, "portfolio_gauss/validate/" * ticker, reports[ticker])
    end
end

"""
    once_seeded_gaussian_portfolio(P, tickers) -> (PortfolioModel, Matrix{Float64})

The Gaussian-copula portfolio of the once-seeded keys, fit to the first 301 days of every ticker with 10 states, and its prices.
"""
function once_seeded_gaussian_portfolio(P::Matrix{Float64}, tickers::Vector{String})
    P_short = P[1:301, :]
    return fit(PortfolioModel, tickers, P_short; dependence=GaussianCopula, N=10), P_short
end

"""
    sector_dt_portfolio(P, tickers, sectors) -> PortfolioModel

The sector-factor portfolio whose `sector_options` set a monthly time step, `dt = 1/12`, with 20 states and a short centering draw.
"""
function sector_dt_portfolio(P::Matrix{Float64}, tickers::Vector{String}, sectors::Vector{String})
    return fit(PortfolioModel, tickers, P; dependence=SectorFactorModel, market="SPY", sectors=sectors, N=20,
               sector_options=(dt=1/12, center_steps=1000))
end

"""
    add_once_seeded_portfolios!(ref, P, tickers)

Portfolio tune and validate after one `Random.seed!(7)` and with no `seed` keyword, so that the order in which the portfolio
visits its models sets the draws of each model. The Gaussian-copula portfolio uses all 13 tickers, and the hybrid portfolio uses
the first five columns. Both use the first 301 days and 10 states. The hybrid keys hold the portfolio marginals, the hybrid
model's own marginals (tuned in a second pass, after the market), and the market model.
"""
function add_once_seeded_portfolios!(ref::Dict{String,Any}, P::Matrix{Float64}, tickers::Vector{String})
    pg, P_short = once_seeded_gaussian_portfolio(P, tickers)
    Random.seed!(7)
    tuned = tune(pg, P_short; ϵ_range=[0.01, 0.05], λ_range=[2.0, 10.0], n_paths=3, n_steps=80)
    for ticker in tickers
        ref["portfolio_once/gauss/tune/" * ticker * "/ϵ"] = tuned.marginals[ticker].jump.ϵ
        ref["portfolio_once/gauss/tune/" * ticker * "/λ"] = tuned.marginals[ticker].jump.λ
    end
    Random.seed!(7)
    reports = validate(pg, P_short; n_paths=1)
    for ticker in tickers
        add_report!(ref, "portfolio_once/gauss/validate/" * ticker, reports[ticker])
    end

    P_hybrid = P[1:301, 1:5]
    ph = fit(PortfolioModel, tickers[1:5], P_hybrid; dependence=HybridSingleIndexModel, market="SPY", N=10)
    Random.seed!(7)
    th = tune(ph, P_hybrid; ϵ_range=[0.01, 0.05], λ_range=[2.0, 10.0], n_paths=3, n_steps=80)
    for ticker in ph.tickers
        ref["portfolio_once/hybrid/tune/marginals/" * ticker * "/ϵ"] = th.marginals[ticker].jump.ϵ
        ref["portfolio_once/hybrid/tune/marginals/" * ticker * "/λ"] = th.marginals[ticker].jump.λ
        ref["portfolio_once/hybrid/tune/dependence/" * ticker * "/ϵ"] = th.dependence.marginals[ticker].jump.ϵ
        ref["portfolio_once/hybrid/tune/dependence/" * ticker * "/λ"] = th.dependence.marginals[ticker].jump.λ
    end
    ref["portfolio_once/hybrid/tune/market/ϵ"] = th.dependence.market_model.jump.ϵ
    ref["portfolio_once/hybrid/tune/market/λ"] = th.dependence.market_model.jump.λ
end

"""
    add_sector_dt_portfolio!(ref, P, tickers, sectors)

A sector-factor portfolio whose `sector_options` set `dt = 1/12`, validated with one path and `seed = 7`. Each own model is
checked on growth rates made with its own `dt`. The keys hold the reports of ADP and AFL.
"""
function add_sector_dt_portfolio!(ref::Dict{String,Any}, P::Matrix{Float64}, tickers::Vector{String},
                                  sectors::Vector{String})
    p = sector_dt_portfolio(P, tickers, sectors)
    reports = validate(p, P; n_paths=1, seed=7)
    for ticker in ["ADP", "AFL"]
        add_report!(ref, "portfolio_sector_dt/validate/" * ticker, reports[ticker])
    end
end

"""
    add_one_column!(ref, P, GA, tickers)

Copulas fit to one column of growth rates (the vine has no edges), and a hybrid single-index model of the market and one stock,
sampled on its own and simulated as a portfolio.
"""
function add_one_column!(ref::Dict{String,Any}, P::Matrix{Float64}, GA::Matrix{Float64}, tickers::Vector{String})
    R = GA[:, 2:2]

    gc = fit(GaussianCopula, R)
    Random.seed!(19)
    ref["copula_one/gaussian/Σ"] = copy(gc.Σ)
    ref["copula_one/gaussian/U"] = sample_dependence(gc, 50)

    tc = fit(StudentTCopula, R)
    Random.seed!(20)
    ref["copula_one/studentt/Σ"] = copy(tc.Σ)
    ref["copula_one/studentt/ν"] = tc.ν
    ref["copula_one/studentt/U"] = sample_dependence(tc, 50)

    vc = fit(VineCopula, R)
    Random.seed!(21)
    ref["copula_one/vine/order"] = copy(vc.order)
    ref["copula_one/vine/number_of_edges"] = sum(length(tree) for tree in vc.edges; init=0)
    ref["copula_one/vine/U"] = sample_dependence(vc, 50)

    h = fit(HybridSingleIndexModel, tickers[1:2], P[:, 1:2], "SPY"; N=20)
    Random.seed!(5)
    ref["hybrid_one/sample"] = sample_dependence(h, 100)

    p = fit(PortfolioModel, tickers[1:2], P[:, 1:2]; dependence=HybridSingleIndexModel, market="SPY", N=20)
    add_portfolio_simulation!(ref, "hybrid_one/portfolio/simulate", simulate(p, 100; n_paths=2, seed=6))
end

"""
    add_key_orders!(ref, P, tickers, sectors)

The iteration order of the `marginals` and `tickers_map` dictionaries of the sector-factor, Gaussian-copula, and hybrid
portfolios on all 13 tickers. The order of `marginals` sets which model draws first when a portfolio is tuned or validated
without a seed. These keys hold no random numbers.
"""
function add_key_orders!(ref::Dict{String,Any}, P::Matrix{Float64}, tickers::Vector{String}, sectors::Vector{String})
    ps = fit(PortfolioModel, tickers, P; dependence=SectorFactorModel, market="SPY", sectors=sectors, N=20,
             sector_options=(center_steps=1000,))
    pg = fit(PortfolioModel, tickers, P; dependence=GaussianCopula, N=20)
    ph = fit(PortfolioModel, tickers, P; dependence=HybridSingleIndexModel, market="SPY", N=20)
    for (name, p) in [("sector", ps), ("gauss", pg), ("hybrid", ph)]
        ref["key_order/" * name * "/marginals"] = collect(keys(p.marginals))
        ref["key_order/" * name * "/tickers_map"] = collect(keys(p.tickers_map))
    end
end

# ---------------------------------------------------------------------------
# The corrected Hellinger distance: 0.5.0 counts the largest observation
# ---------------------------------------------------------------------------

"""
    corrected_hellinger(x, y) -> Float64

The Hellinger distance with every observation counted. The bins split the range of both samples into `n_bins` equal widths.
A value `v` falls in bin `floor(Int, (v - lo) / (hi - lo) * n_bins) + 1`, and the largest value goes into the last bin.
0.4.1's `_hellinger` used a histogram closed on the left, which dropped every value equal to the largest observation.
"""
function corrected_hellinger(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    lo = min(minimum(x), minimum(y))
    hi = max(maximum(x), maximum(y))
    if lo == hi
        return 0.0
    end
    n_bins = max(30, round(Int, sqrt(max(length(x), length(y)))))
    counts_x = zeros(Int, n_bins)
    for v in x
        k = floor(Int, (v - lo) / (hi - lo) * n_bins) + 1
        if k > n_bins
            k = n_bins
        end
        counts_x[k] += 1
    end
    counts_y = zeros(Int, n_bins)
    for v in y
        k = floor(Int, (v - lo) / (hi - lo) * n_bins) + 1
        if k > n_bins
            k = n_bins
        end
        counts_y[k] += 1
    end
    px = counts_x ./ sum(counts_x)
    py = counts_y ./ sum(counts_y)
    return sqrt(0.5 * sum((sqrt.(px) .- sqrt.(py)).^2))
end

"""
    patch_hellinger!()

Replace 0.4.1's `JumpHMM._hellinger` with `corrected_hellinger`, so that 0.4.1's `validate` reports the corrected distance on
the same simulated paths. The method keeps the 0.4.1 signature, so that it replaces the old method. Call the functions that
use it through `Base.invokelatest`.
"""
function patch_hellinger!()
    @eval JumpHMM _hellinger(x::AbstractVector{Float64}, y::AbstractVector{Float64}) = $(corrected_hellinger)(x, y)
end

"""
    add_corrected_report!(ref, prefix, report)

Store the `hellinger` and `mean_hellinger` fields of a report under the original keys with `_corrected` appended.
"""
function add_corrected_report!(ref::Dict{String,Any}, prefix::String, report::ValidationReport)
    ref[prefix * "/hellinger_corrected"] = [r.hellinger for r in report.path_results]
    ref[prefix * "/mean_hellinger_corrected"] = report.mean_hellinger
end

"""
    add_corrected_hellinger!(ref, inputs_path)

Run every `validate` call of this script again, after `patch_hellinger!`, with the same models and the same seeding, and the
direct distance call. Only the Hellinger values change, because the distance draws no random numbers.
"""
function add_corrected_hellinger!(ref::Dict{String,Any}, inputs_path::String)
    inputs = load(inputs_path)
    P = inputs["prices"]
    tickers = String.(inputs["tickers"])
    sectors = String.(inputs["sectors"])
    GA = excess_growth_rates(P; rf=0.0, dt=1/252)

    # the single-asset report and the direct call
    m0 = fit(JumpHiddenMarkovModel, P[:, 1]; N=100, ν=5.0)
    mj = with_jump(m0, 1e-3, 90.0)
    add_corrected_report!(ref, "validate", validate(mj, P[:, 1]; n_paths=20, seed=9))
    ref["distance/hellinger_corrected"] = JumpHMM._hellinger(GA[:, 2], GA[:, 3])

    # the Gaussian-copula portfolio
    pg = fit(PortfolioModel, tickers[1:5], P[:, 1:5]; dependence=GaussianCopula, N=20)
    reports = validate(pg, P[:, 1:5]; n_paths=3, seed=18)
    for ticker in pg.tickers
        add_corrected_report!(ref, "portfolio_gauss/validate/" * ticker, reports[ticker])
    end

    # the once-seeded Gaussian-copula portfolio
    po, P_short = once_seeded_gaussian_portfolio(P, tickers)
    Random.seed!(7)
    reports = validate(po, P_short; n_paths=1)
    for ticker in tickers
        add_corrected_report!(ref, "portfolio_once/gauss/validate/" * ticker, reports[ticker])
    end

    # the sector-factor portfolio with a monthly time step
    ps = sector_dt_portfolio(P, tickers, sectors)
    reports = validate(ps, P; n_paths=1, seed=7)
    for ticker in ["ADP", "AFL"]
        add_corrected_report!(ref, "portfolio_sector_dt/validate/" * ticker, reports[ticker])
    end
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    build_reference(inputs_path) -> Dict{String,Any}

Run every 0.4.1 call listed in the reference table and collect the results by key.
"""
function build_reference(inputs_path::String)
    inputs = load(inputs_path)
    P = inputs["prices"]
    tickers = String.(inputs["tickers"])
    sectors = String.(inputs["sectors"])

    G = excess_growth_rates(P[:, 1]; rf=0.0, dt=1/252)
    GA = excess_growth_rates(P; rf=0.0, dt=1/252)
    m0 = fit(JumpHiddenMarkovModel, P[:, 1]; N=100, ν=5.0)
    mj = with_jump(m0, 1e-3, 90.0)

    ref = Dict{String,Any}()
    ref["meta/julia_version"] = string(VERSION)
    ref["meta/jumphmm_version"] = string(pkgversion(JumpHMM))
    ref["meta/jumphmm_commit"] = jumphmm_commit(joinpath(@__DIR__, "Project.toml"))

    add_finance!(ref, P, G, GA)
    add_fits!(ref, P, G, m0)
    add_simulations!(ref, m0, mj)
    add_tuning!(ref, P, m0)
    add_validation!(ref, P, GA, mj)
    add_decoding!(ref, G, mj)
    add_copulas!(ref, GA)
    add_single_index!(ref, P, GA, tickers)

    sector_fit = fit(SectorFactorModel, tickers, P, "SPY", sectors; N=20,
                     center_steps=20_000, center_seed=4 * 10^12)
    add_sector_model!(ref, G, sector_fit[1])

    add_portfolios!(ref, P, tickers, sectors)

    # keys added after the first release of the files. They come after every earlier key, so that the earlier keys keep their draws
    add_once_seeded_portfolios!(ref, P, tickers)
    add_sector_dt_portfolio!(ref, P, tickers, sectors)
    add_one_column!(ref, P, GA, tickers)
    add_key_orders!(ref, P, tickers, sectors)
    return ref
end

"""
    main(inputs_path, output_path)

Build the reference values and write them to `output_path`. The corrected Hellinger keys come last: we patch 0.4.1's
`_hellinger` only after every other key is made, and run the `validate` calls again in the newest world.
"""
function main(inputs_path::String, output_path::String)
    started = time()
    ref = build_reference(inputs_path)
    patch_hellinger!()
    Base.invokelatest(add_corrected_hellinger!, ref, inputs_path)
    save(output_path, ref)
    println("wrote ", output_path)
    println("keys: ", length(ref))
    println("seconds: ", round(time() - started; digits=1))
end

if abspath(PROGRAM_FILE) == @__FILE__
    inputs_path = DEFAULT_INPUTS
    output_path = DEFAULT_OUTPUT
    if length(ARGS) >= 1
        inputs_path = ARGS[1]
    end
    if length(ARGS) >= 2
        output_path = ARGS[2]
    end
    main(inputs_path, output_path)
end
