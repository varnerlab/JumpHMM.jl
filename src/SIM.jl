"""
    fit(::Type{SingleIndexModel}, returns, market_returns; kwargs...) → SingleIndexModel

Fit a Single-Index Model. For each asset, estimates α, β, σ_ε via OLS:
    G_i = αᵢ + βᵢ × G_market + εᵢ

Also fits a JumpHiddenMarkovModel to the market returns.
"""
function fit(::Type{SingleIndexModel}, returns::AbstractMatrix{Float64},
             market_returns::AbstractVector{Float64};
             market_prices::Union{AbstractVector{<:Real},Nothing}=nothing,
             rf::Float64=0.0, N::Int=100, ν::Float64=5.0,
             dt::Float64=1/252, min_obs::Int=2)

    n_obs, n_assets = size(returns)
    @assert length(market_returns) == n_obs "market_returns must match returns row count"

    # fit market model
    if market_prices !== nothing
        market_model = fit(JumpHiddenMarkovModel, market_prices;
                           rf=rf, N=N, ν=ν, dt=dt, min_obs=min_obs)
    else
        # reconstruct prices from returns for fitting (use P0=100 as dummy)
        market_prices_synth = prices_from_growth_rates(market_returns, 100.0; rf=rf, dt=dt)
        market_model = fit(JumpHiddenMarkovModel, market_prices_synth;
                           rf=rf, N=N, ν=ν, dt=dt, min_obs=min_obs)
    end

    # OLS regression per asset: G_i = α_i + β_i * G_market + ε_i
    α = Vector{Float64}(undef, n_assets)
    β = Vector{Float64}(undef, n_assets)
    σ_ε = Vector{Float64}(undef, n_assets)

    G_m = market_returns
    G_m_mean = mean(G_m)
    G_m_var = var(G_m)
    if G_m_var ≈ 0.0
        throw(ArgumentError("Cannot fit SingleIndexModel: market returns have zero variance (constant market factor)."))
    end

    for i in 1:n_assets
        G_i = returns[:, i]
        G_i_mean = mean(G_i)
        β[i] = cov(G_i, G_m) / G_m_var
        α[i] = G_i_mean - β[i] * G_m_mean
        residuals = G_i .- α[i] .- β[i] .* G_m
        σ_ε[i] = std(residuals)
    end

    return SingleIndexModel(α, β, σ_ε, market_model)
end

"""
    sample_dependence(sim::SingleIndexModel, n) → Matrix{Float64}

Simulate market returns via HMM, then generate asset returns:
    G_i = αᵢ + βᵢ × G_market + εᵢ
"""
function sample_dependence(sim::SingleIndexModel, n::Int)
    # simulate one market path
    market_result = simulate(sim.market_model, n; n_paths=1)
    G_market = market_result.paths[1].observations

    n_assets = length(sim.α)
    returns = Matrix{Float64}(undef, n, n_assets)
    for i in 1:n_assets
        for t in 1:n
            returns[t, i] = sim.α[i] + sim.β[i] * G_market[t] +
                            sim.σ_ε[i] * randn()
        end
    end

    return returns
end
