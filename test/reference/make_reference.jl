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
    return ref
end

"""
    main(inputs_path, output_path)

Build the reference values and write them to `output_path`.
"""
function main(inputs_path::String, output_path::String)
    started = time()
    ref = build_reference(inputs_path)
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
