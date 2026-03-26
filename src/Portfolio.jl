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

    if dependence <: Union{GaussianCopula,StudentTCopula,VineCopula}
        # fit marginal HMMs per asset
        marginals = Dict{String,JumpHiddenMarkovModel}()
        for (j, ticker) in enumerate(tickers)
            marginals[ticker] = fit(JumpHiddenMarkovModel, prices[:, j];
                                    rf=rf, N=N, ν=ν, dt=dt, min_obs=min_obs)
        end

        # fit copula
        dep = fit(dependence, returns)
        tickers_map = Dict{String,Int}(t => j for (j, t) in enumerate(tickers))
        return PortfolioModel(tickers, marginals, dep, tickers_map, "")

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

        # store original column mapping including market
        tickers_map = Dict{String,Int}(t => j for (j, t) in enumerate(tickers))
        return PortfolioModel(asset_tickers, marginals, sim_model, tickers_map, market)
    else
        error("Unsupported dependence model: $dependence")
    end
end

"""
    tune(portfolio, prices; tickers_map=nothing, market_col=nothing, kwargs...) → PortfolioModel

Tune jump parameters for each marginal model. Returns a new PortfolioModel.

By default, uses the column mapping stored in `portfolio.tickers_map` (set during `fit`).
For `SingleIndexModel` portfolios, the market model is also tuned automatically using
the stored market ticker mapping.
"""
function tune(portfolio::PortfolioModel, prices::AbstractMatrix{<:Real};
              tickers_map::Union{Dict{String,Int},Nothing}=nothing,
              market_col::Union{Int,Nothing}=nothing,
              kwargs...)

    # use stored mapping by default
    col_map = tickers_map !== nothing ? tickers_map : portfolio.tickers_map

    # tune each marginal
    tuned_marginals = Dict{String,JumpHiddenMarkovModel}()
    for (ticker, model) in portfolio.marginals
        col = col_map[ticker]
        tuned_marginals[ticker] = tune(model, prices[:, col]; kwargs...)
    end

    # tune market model if SIM
    dep = portfolio.dependence
    if dep isa SingleIndexModel
        # resolve market column: explicit arg > stored mapping
        mcol = market_col
        if mcol === nothing && !isempty(portfolio.market_ticker)
            mcol = get(col_map, portfolio.market_ticker, nothing)
        end
        if mcol !== nothing
            market_tuned = tune(dep.market_model, prices[:, mcol]; kwargs...)
        else
            market_tuned = dep.market_model
        end
        dep = SingleIndexModel(dep.α, dep.β, dep.σ_ε, market_tuned)
    end

    return PortfolioModel(portfolio.tickers, tuned_marginals, dep,
                          portfolio.tickers_map, portfolio.market_ticker)
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

    if dep isa Union{GaussianCopula,StudentTCopula,VineCopula}
        # copula approach per SPECIFICATION.md §7.3:
        # 1. Simulate each marginal independently (preserving state dynamics + jumps)
        # 2. Use copula ranks to reorder observations, preserving cross-asset dependence
        #    while maintaining each marginal's distributional properties
        for ticker in tickers
            results[ticker] = SimulationResult(Vector{SimulationPath}())
        end

        for _ in 1:n_paths
            # sample correlated uniforms
            U = sample_dependence(dep, n_steps)  # n_steps × n_assets

            for (j, ticker) in enumerate(tickers)
                model = portfolio.marginals[ticker]

                # simulate a single path to get state sequence, observations, and jump flags
                r = simulate(model, n_steps; n_paths=1)
                path = r.paths[1]

                # rank-reorder: sort the marginal observations, then place them
                # according to the copula-induced ranks to inject cross-asset dependence
                sorted_obs = sort(path.observations)
                copula_ranks = ordinalrank(U[:, j])
                obs_out = sorted_obs[copula_ranks]

                push!(results[ticker].paths,
                      SimulationPath(path.states, obs_out, path.jumps))
            end
        end

    elseif dep isa SingleIndexModel
        # SIM approach: simulate market, then generate asset returns
        for _ in 1:n_paths
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
    validate(portfolio, prices; tickers_map=nothing, kwargs...) → Dict{String, ValidationReport}

Validate each marginal model against its empirical data.

By default, uses the column mapping stored in `portfolio.tickers_map` (set during `fit`).
"""
function validate(portfolio::PortfolioModel,
                  prices::AbstractMatrix{<:Real};
                  tickers_map::Union{Dict{String,Int},Nothing}=nothing,
                  kwargs...)

    col_map = tickers_map !== nothing ? tickers_map : portfolio.tickers_map

    reports = Dict{String,ValidationReport}()
    for (ticker, model) in portfolio.marginals
        col = col_map[ticker]
        reports[ticker] = validate(model, prices[:, col]; kwargs...)
    end

    return reports
end
