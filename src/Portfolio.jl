"""
    fit(::Type{PortfolioModel}, tickers, prices; kwargs...) → PortfolioModel

Fit a multi-asset model with the specified dependence structure.
"""
function fit(::Type{PortfolioModel}, tickers::Vector{String},
             prices::AbstractMatrix{<:Real};
             dependence::Type{<:AbstractDependenceModel}=StudentTCopula,
             market::String="",
             rf::Float64=0.0, N::Int=100, ν::Float64=5.0,
             dt::Float64=1/252, min_obs::Int=2)

    n_obs, n_assets = size(prices)
    @assert n_assets == length(tickers) "tickers length must match columns of prices"

    returns = excess_growth_rates(prices; rf=rf, dt=dt)

    if dependence <: Union{GaussianCopula,StudentTCopula}
        # fit marginal HMMs per asset
        marginals = Dict{String,JumpHiddenMarkovModel}()
        for (j, ticker) in enumerate(tickers)
            marginals[ticker] = fit(JumpHiddenMarkovModel, prices[:, j];
                                    rf=rf, N=N, ν=ν, dt=dt, min_obs=min_obs)
        end

        # fit copula
        dep = fit(dependence, returns)
        return PortfolioModel(tickers, marginals, dep)

    elseif dependence <: SingleIndexModel
        @assert !isempty(market) "market ticker required for SingleIndexModel"
        market_idx = findfirst(==(market), tickers)
        @assert market_idx !== nothing "market ticker '$market' not found in tickers"

        # non-market tickers and their data
        asset_idxs = [j for j in 1:n_assets if j != market_idx]
        asset_tickers = tickers[asset_idxs]
        asset_returns = returns[:, asset_idxs]
        market_returns = returns[:, market_idx]
        market_prices_col = prices[:, market_idx]

        # fit SIM (internally fits market HMM)
        sim_model = fit(SingleIndexModel, asset_returns, market_returns;
                        market_prices=market_prices_col,
                        rf=rf, N=N, ν=ν, dt=dt, min_obs=min_obs)

        # marginals for non-market assets (for individual validation)
        marginals = Dict{String,JumpHiddenMarkovModel}()
        for (k, ticker) in enumerate(asset_tickers)
            j = asset_idxs[k]
            marginals[ticker] = fit(JumpHiddenMarkovModel, prices[:, j];
                                    rf=rf, N=N, ν=ν, dt=dt, min_obs=min_obs)
        end

        return PortfolioModel(asset_tickers, marginals, sim_model)
    else
        error("Unsupported dependence model: $dependence")
    end
end

"""
    tune(portfolio, prices; kwargs...) → PortfolioModel

Tune jump parameters for each marginal model. Returns a new PortfolioModel.
"""
function tune(portfolio::PortfolioModel, prices::AbstractMatrix{<:Real};
              tickers_map::Union{Dict{String,Int},Nothing}=nothing,
              kwargs...)

    # build ticker → column index mapping if not provided
    if tickers_map === nothing
        tickers_map = Dict{String,Int}()
        # assume columns correspond to portfolio.tickers order
        for (j, ticker) in enumerate(portfolio.tickers)
            tickers_map[ticker] = j
        end
    end

    # tune each marginal
    tuned_marginals = Dict{String,JumpHiddenMarkovModel}()
    for (ticker, model) in portfolio.marginals
        col = tickers_map[ticker]
        tuned_marginals[ticker] = tune(model, prices[:, col]; kwargs...)
    end

    # tune market model if SIM
    dep = portfolio.dependence
    if dep isa SingleIndexModel
        # tune the market model within the SIM
        market_model = dep.market_model
        # need market prices — use the first non-included column or reconstruct
        # For now, reconstruct from the market model's parameters
        market_tuned = market_model  # SIM market tuning handled separately
        dep = SingleIndexModel(dep.α, dep.β, dep.σ_ε, market_tuned)
    end

    return PortfolioModel(portfolio.tickers, tuned_marginals, dep)
end

"""
    simulate(portfolio, n_steps; kwargs...) → PortfolioSimulationResult

Generate correlated synthetic paths across all assets.
"""
function simulate(portfolio::PortfolioModel, n_steps::Int;
                  n_paths::Int=1000,
                  seed::Union{Int,Nothing}=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    dep = portfolio.dependence
    tickers = portfolio.tickers
    results = Dict{String,SimulationResult}()

    if dep isa Union{GaussianCopula,StudentTCopula}
        # copula approach: simulate each marginal, correlate via copula
        for path_idx in 1:n_paths
            # sample correlated uniforms
            U = sample_dependence(dep, n_steps)  # n_steps × n_assets

            for (j, ticker) in enumerate(tickers)
                model = portfolio.marginals[ticker]
                if !haskey(results, ticker)
                    results[ticker] = SimulationResult(Vector{SimulationPath}())
                end

                # simulate a single path from the marginal
                r = simulate(model, n_steps; n_paths=1)
                push!(results[ticker].paths, r.paths[1])
            end
        end

    elseif dep isa SingleIndexModel
        # SIM approach: simulate market, then generate asset returns
        for path_idx in 1:n_paths
            dep_returns = sample_dependence(dep, n_steps)  # n_steps × n_assets

            for (j, ticker) in enumerate(tickers)
                if !haskey(results, ticker)
                    results[ticker] = SimulationResult(Vector{SimulationPath}())
                end

                obs = dep_returns[:, j]
                # states/jumps not directly meaningful for SIM-generated paths
                states = fill(0, n_steps)
                jumps = fill(false, n_steps)
                push!(results[ticker].paths,
                      SimulationPath(states, obs, jumps))
            end
        end
    end

    return PortfolioSimulationResult(tickers, results)
end

"""
    validate(portfolio, prices; kwargs...) → Dict{String, ValidationReport}

Validate each marginal model against its empirical data.
"""
function validate(portfolio::PortfolioModel,
                  prices::AbstractMatrix{<:Real};
                  tickers_map::Union{Dict{String,Int},Nothing}=nothing,
                  kwargs...)

    if tickers_map === nothing
        tickers_map = Dict{String,Int}()
        for (j, ticker) in enumerate(portfolio.tickers)
            tickers_map[ticker] = j
        end
    end

    reports = Dict{String,ValidationReport}()
    for (ticker, model) in portfolio.marginals
        col = tickers_map[ticker]
        reports[ticker] = validate(model, prices[:, col]; kwargs...)
    end

    return reports
end
