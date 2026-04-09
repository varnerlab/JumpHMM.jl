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
             dt::Float64=1/252, min_obs::Int=2,
             residual_method::Symbol=:bootstrap)

    residual_method in (:bootstrap, :gaussian) ||
        throw(ArgumentError("residual_method must be :bootstrap or :gaussian, got :$residual_method"))

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
    resid_matrix = Matrix{Float64}(undef, n_obs, n_assets)

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
        resid_matrix[:, i] = G_i .- α[i] .- β[i] .* G_m
        σ_ε[i] = std(resid_matrix[:, i])
    end

    return SingleIndexModel(α, β, σ_ε, resid_matrix, residual_method, market_model)
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
    T_train = size(sim.residuals, 1)
    returns = Matrix{Float64}(undef, n, n_assets)

    if sim.residual_method == :bootstrap
        idx = rand(1:T_train, n)
        for i in 1:n_assets
            for t in 1:n
                returns[t, i] = sim.α[i] + sim.β[i] * G_market[t] + sim.residuals[idx[t], i]
            end
        end
    else  # :gaussian
        for i in 1:n_assets
            for t in 1:n
                returns[t, i] = sim.α[i] + sim.β[i] * G_market[t] +
                                sim.σ_ε[i] * randn()
            end
        end
    end

    return returns
end

# --- Hybrid Single-Index Model ------------------------------------------------

"""
    fit(::Type{HybridSingleIndexModel}, tickers, prices, market_ticker; kwargs...) → HybridSingleIndexModel

Fit a variance-corrected hybrid SIM end-to-end from a price matrix. This:
1. Fits a JumpHMM to the market column
2. Fits per-ticker HMM marginals for all non-market tickers
3. Runs OLS regression per ticker against the market → α, β, r²
4. Fits a copula on the OLS residuals

The resulting model can be passed to `sample_dependence` to generate synthetic
growth rates via the hybrid construction (variance correction + copula reorder).

### Arguments
- `tickers` — column labels for `prices`, must include `market_ticker`
- `prices` — `(n_obs × n_assets)` price matrix
- `market_ticker` — which ticker is the market index (e.g. "SPY")
- `copula_type` — dependence model for residuals (default `StudentTCopula`)
- `rf, N, ν, dt, min_obs` — passed through to `fit(JumpHiddenMarkovModel, ...)`
- `r2_preserve_threshold` — R² cutoff for the R²-preserving branch (default 0.80)
- `idiosyncratic_floor` — minimum idiosyncratic variance share f (default 0.10)
"""
function fit(::Type{HybridSingleIndexModel},
             tickers::Vector{String},
             prices::AbstractMatrix{<:Real},
             market_ticker::String;
             copula_type::Type{<:AbstractDependenceModel}=StudentTCopula,
             rf::Float64=0.0, N::Int=100, ν::Float64=5.0,
             dt::Float64=1/252, min_obs::Int=2,
             r2_preserve_threshold::Float64=0.80,
             idiosyncratic_floor::Float64=0.10)

    n_obs, n_cols = size(prices)
    @assert n_cols == length(tickers) "tickers length must match columns of prices"

    market_idx = findfirst(==(market_ticker), tickers)
    @assert market_idx !== nothing "market ticker '$market_ticker' not found in tickers"

    # --- Excess growth rates ---
    returns = excess_growth_rates(prices; rf=rf, dt=dt)
    market_returns = returns[:, market_idx]
    σ_market = std(market_returns)

    if σ_market ≈ 0.0
        throw(ArgumentError("Cannot fit HybridSingleIndexModel: market returns have zero variance."))
    end

    # --- Fit market HMM ---
    market_model = fit(JumpHiddenMarkovModel, prices[:, market_idx];
                       rf=rf, N=N, ν=ν, dt=dt, min_obs=min_obs)

    # --- Non-market tickers ---
    asset_idxs = [j for j in 1:n_cols if j != market_idx]
    asset_tickers = tickers[asset_idxs]
    n_assets = length(asset_tickers)

    # --- Fit per-ticker marginals + OLS ---
    marginals = Dict{String,JumpHiddenMarkovModel}()
    α_vec = Vector{Float64}(undef, n_assets)
    β_vec = Vector{Float64}(undef, n_assets)
    r²_vec = Vector{Float64}(undef, n_assets)
    resid_matrix = Matrix{Float64}(undef, size(returns, 1), n_assets)

    G_m_mean = mean(market_returns)
    G_m_var = var(market_returns)

    for (k, j) in enumerate(asset_idxs)
        ticker = asset_tickers[k]

        # fit HMM marginal
        marginals[ticker] = fit(JumpHiddenMarkovModel, prices[:, j];
                                rf=rf, N=N, ν=ν, dt=dt, min_obs=min_obs)

        # OLS: G_i = α + β·G_m + ε
        G_i = returns[:, j]
        G_i_mean = mean(G_i)
        β_vec[k] = cov(G_i, market_returns) / G_m_var
        α_vec[k] = G_i_mean - β_vec[k] * G_m_mean

        resid = G_i .- α_vec[k] .- β_vec[k] .* market_returns
        resid_matrix[:, k] = resid

        SS_res = dot(resid, resid)
        SS_tot = dot(G_i .- G_i_mean, G_i .- G_i_mean)
        r²_vec[k] = SS_tot > 0.0 ? 1.0 - SS_res / SS_tot : 0.0
    end

    # --- Fit copula on OLS residuals ---
    copula = fit(copula_type, resid_matrix)

    return HybridSingleIndexModel(
        α_vec, β_vec, r²_vec, σ_market,
        marginals, copula, market_model, asset_tickers,
        r2_preserve_threshold, idiosyncratic_floor
    )
end

"""
    sample_dependence(model::HybridSingleIndexModel, n) → Matrix{Float64}

Generate `n` growth-rate observations for each ticker via the hybrid SIM recipe:
1. Simulate a market path from the market HMM
2. For each ticker, draw an idiosyncratic path from its HMM marginal
3. Apply variance correction (R²-preserving or marginal-preserving branch)
4. Copula rank-reorder the scaled innovations
5. Compose: `G_i = α_i + β_i · G_market + ε_i`

Returns an `(n_eff × n_assets)` matrix of annualized growth rates, where
`n_eff = n - 1` (first observation trimmed for alignment with the HMM
simulation convention).

### Branches
- **R²-preserving** (`r² ≥ r2_preserve_threshold`): scales ε so synthetic R²
  matches calibrated R² exactly. SPY-like tickers with R²≈1 produce ε≡0.
- **Marginal-preserving** (`r² < threshold`): scales ε to preserve the HMM
  marginal variance, with β clipping when the market loading exceeds
  `1 - idiosyncratic_floor` of the generator variance.
"""
function sample_dependence(model::HybridSingleIndexModel, n::Int)

    n_assets = length(model.tickers)
    σ²_m = model.σ_market^2
    f = model.idiosyncratic_floor
    threshold = model.r2_preserve_threshold

    # --- 1. Simulate market path ---
    market_result = simulate(model.market_model, n; n_paths=1)
    G_full = Float64.(market_result.paths[1].observations)
    G_market = G_full[2:end]    # trim first obs; length = n - 1
    T_eff = length(G_market)

    # --- 2. Sample copula uniforms ---
    U = sample_dependence(model.copula, n)
    # trim to align with G_market
    U_trimmed = U[2:end, :]     # (T_eff × n_copula_cols)

    # --- 3. Per-ticker hybrid composition ---
    returns = Matrix{Float64}(undef, T_eff, n_assets)

    for k in 1:n_assets
        ticker = model.tickers[k]
        α_k = model.α[k]
        β_k = model.β[k]
        r²_k = model.r²[k]

        # draw HMM marginal path and trim
        r_j = simulate(model.marginals[ticker], n; n_paths=1)
        obs_full = Float64.(r_j.paths[1].observations)
        obs_j = obs_full[2:end]     # length T_eff
        σ²_HMM = var(obs_j)

        # --- choose construction branch ---
        β_eff = β_k
        if r²_k >= threshold
            # R²-PRESERVING BRANCH
            if r²_k >= 1.0 - 1e-12
                σ²_ε_target = 0.0
            else
                σ²_ε_target = β_k^2 * σ²_m * (1.0 - r²_k) / r²_k
            end
            ε_scaled = (σ²_HMM > 0.0 && σ²_ε_target > 0.0) ?
                       sqrt(σ²_ε_target / σ²_HMM) .* obs_j :
                       zeros(T_eff)
        else
            # MARGINAL-PRESERVING BRANCH with β clipping
            ρ = β_k^2 * σ²_m / max(σ²_HMM, 1e-30)
            if ρ > 1.0 - f
                β_eff = sign(β_k) * sqrt((1.0 - f) * σ²_HMM / σ²_m)
                s² = f
            else
                s² = 1.0 - ρ
            end
            ε_scaled = sqrt(s²) .* obs_j
        end

        # --- copula rank-reorder ---
        sorted_eps = sort(ε_scaled)
        copula_ranks = ordinalrank(U_trimmed[:, k])
        ε_reordered = sorted_eps[copula_ranks]

        # --- compose ---
        returns[:, k] = α_k .+ β_eff .* G_market .+ ε_reordered
    end

    return returns
end
