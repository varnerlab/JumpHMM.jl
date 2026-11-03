# --- Sector factor model ----------------------------------------------------
#
# Multi-asset composition from the INFORMS 2026 study (Varner, arXiv:2611.03022):
#   g_i(t) = α_i + β_i(t)·g_m(t) + δ_i·f_s(t) + s_i·ε̃_i(t)
# where g_m is one draw of the market model, f_s one draw of the model of name i's sector,
# and ε̃_i one draw of name i's own model. Names are linked only through the shared market
# and sector draws on the same day; there is no copula.

"""
    ema_rate(h) → Float64

Weight `a = 1 − 2^(−1/h)` of an exponential moving average with half-life `h` (trading days).
Throws `ArgumentError` if `h ≤ 0`.
"""
function ema_rate(h::Real)::Float64
    h > 0 || throw(ArgumentError("half-life h = $(h) must be positive"))
    return 1.0 - 2.0^(-1.0 / h)
end

"""
    drift_path(z, h, k) → Vector{Float64}

Exponential moving average of the standard-normal shocks `z`, scaled by `k`, with half-life `h`
(trading days). With `a = ema_rate(h)`: `x₁ = k·√(a/(2−a))·z₁` and `xₜ = (1−a)·xₜ₋₁ + a·k·zₜ`.
The start is drawn from the stationary distribution, so every `xₜ` has variance `drift_variance(h, k)`.
The output is linear in `k`.
"""
function drift_path(z::AbstractVector{<:Real}, h::Real, k::Real)::Vector{Float64}
    a = ema_rate(h)
    x = zeros(Float64, length(z))
    isempty(z) && return x
    x[1] = k * sqrt(a / (2 - a)) * z[1]
    for t in 2:length(z)
        x[t] = (1 - a) * x[t-1] + a * k * z[t]
    end
    return x
end

"""
    drift_variance(h, k) → Float64

Stationary variance `k²·a/(2−a)` of `drift_path` with half-life `h` and scale `k`, where `a = ema_rate(h)`.
"""
function drift_variance(h::Real, k::Real)::Float64
    a = ema_rate(h)
    return k^2 * a / (2 - a)
end

"""
    sector_factors(E, sector) → (sectors, F)

Equal-weight sector factors from the residual matrix `E` (days × names). `sectors` holds the sorted
unique labels of `sector` (one label per column of `E`), and column `s` of `F` (days × sectors) is the
mean of the columns of `E` whose label is `sectors[s]`.
"""
function sector_factors(E::AbstractMatrix, sector::Vector{String})
    size(E, 2) == length(sector) ||
        throw(ArgumentError("E has $(size(E, 2)) columns but sector has $(length(sector)) labels"))
    sectors = sort(unique(sector))
    F = zeros(Float64, size(E, 1), length(sectors))
    for (s, name) in enumerate(sectors)
        members = findall(==(name), sector)
        F[:, s] = vec(mean(view(E, :, members); dims=2))
    end
    return (sectors=sectors, F=F)
end

"""
    loo_loadings(E, sector) → Vector{Float64}

Leave-one-out sector loadings: `δᵢ` is the OLS slope, with an intercept, of column `i` of `E` on the
mean of the other columns in the same sector. Leaving column `i` out keeps its own noise out of the
regressor. Throws `ArgumentError` for a sector with one name.
"""
function loo_loadings(E::AbstractMatrix, sector::Vector{String})::Vector{Float64}
    size(E, 2) == length(sector) ||
        throw(ArgumentError("E has $(size(E, 2)) columns but sector has $(length(sector)) labels"))
    δ = zeros(Float64, size(E, 2))
    for name in sort(unique(sector))
        members = findall(==(name), sector)
        length(members) >= 2 ||
            throw(ArgumentError("sector $(name) has one name; a leave-one-out mean needs at least two"))
        for i in members
            others = filter(!=(i), members)
            x = vec(mean(view(E, :, others); dims=2))
            y = view(E, :, i)
            x̄ = mean(x)
            ȳ = mean(y)
            δ[i] = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
        end
    end
    return δ
end

"""
    sector_loadings(E, sector) → Vector{Float64}

Sector loadings `δᵢ` for the generator. Each sector's `loo_loadings` are multiplied by one scale `κ`,
chosen so that the shared co-movement the generator builds, `δᵢ·δⱼ·var(f_s)` averaged over the pairs
`i ≠ j` in the sector, equals the mean pairwise covariance of the columns of `E` in that sector.
`f_s` is the full equal-weight sector mean (`sector_factors`). Throws `ArgumentError` for a sector with
one name, or where either mean pairwise covariance is not positive (the message names the sector).
"""
function sector_loadings(E::AbstractMatrix, sector::Vector{String})::Vector{Float64}
    δ = loo_loadings(E, sector)
    sf = sector_factors(E, sector)
    for (s, name) in enumerate(sf.sectors)
        members = findall(==(name), sector)
        n = length(members)
        var_f = var(view(sf.F, :, s))
        # mean pairwise covariance: Σ_{i≠j} cov(eᵢ, eⱼ) = var(Σ eᵢ) − Σ var(eᵢ), with Σ eᵢ = n·f_s
        own = sum(var(view(E, :, i)) for i in members)
        real_cov = (n^2 * var_f - own) / (n * (n - 1))
        real_cov > 0 ||
            throw(ArgumentError("sector $(name) has mean pairwise residual covariance $(real_cov); it must be positive"))
        # the same mean for the leave-one-out loadings: Σ_{i≠j} δᵢδⱼ = (Σ δᵢ)² − Σ δᵢ²
        model_cov = (sum(δ[members])^2 - sum(δ[members] .^ 2)) * var_f / (n * (n - 1))
        model_cov > 0 ||
            throw(ArgumentError("sector $(name) leave-one-out loadings give mean pairwise covariance $(model_cov); it must be positive"))
        δ[members] .*= sqrt(real_cov / model_cov)
    end
    return δ
end

"""
    compose_name(α, β, δ, g_m, f, ε, σ²_m, σ²_s, σ²_gen, r²; ε_center, β²_mean,
                 floor=0.10, r2_threshold=0.80) → (g, β_eff, n_clipped)

Compose one name's daily growth `g[t] = α[t] + c·β[t]·g_m[t] + c·δ·f[t] + √s²·(ε[t] − ε_center)`.
`α` and `β` are per-day paths, `δ` the sector loading, and `g_m`, `f`, `ε` the market, sector, and own
draws. `σ²_m`, `σ²_s`, `σ²_gen` are the fixed variances of the market growth, the sector factor, and the
name's own growth; `r²` is the name's real single-index R²; `β²_mean` is the long-run mean of `β[t]²`.

The variance split is fixed for the whole path: `v̄ = β²_mean·σ²_m + δ²·σ²_s` and `ρ̄ = v̄/σ²_gen`.
If `r² ≥ r2_threshold`, `s² = v̄(1 − r²)/r²/σ²_gen` and `c = 1` (keep the real R²). Otherwise, if
`ρ̄ ≤ 1 − floor`, `s² = 1 − ρ̄` and `c = 1` (match the name's variance). Otherwise both loadings shrink
by `c = √((1 − floor)/ρ̄)`, `s² = floor`, and every day counts as clipped.
Returns the growth path `g`, the per-day `β_eff = c·β`, and `n_clipped` (0 or the path length).
"""
function compose_name(α::AbstractVector, β::AbstractVector, δ::Real, g_m::AbstractVector,
                      f::AbstractVector, ε::AbstractVector, σ²_m::Real, σ²_s::Real, σ²_gen::Real,
                      r²::Real; ε_center::Real, β²_mean::Real, floor::Real=0.10,
                      r2_threshold::Real=0.80)
    T = length(g_m)
    (length(α) == T && length(β) == T && length(f) == T && length(ε) == T) ||
        throw(ArgumentError("α, β, g_m, f and ε must share one length; got $(length(α)), $(length(β)), $(T), $(length(f)), $(length(ε))"))
    0 < floor < 1 || throw(ArgumentError("floor must lie in (0, 1); got $(floor)"))
    σ²_gen > 0 || throw(ArgumentError("σ²_gen must be positive; got $(σ²_gen)"))
    r² > 0 || throw(ArgumentError("r² must be positive; got $(r²)"))
    β²_mean >= 0 || throw(ArgumentError("β²_mean must be non-negative; got $(β²_mean)"))
    v̄ = β²_mean * σ²_m + δ^2 * σ²_s
    ρ̄ = v̄ / σ²_gen
    c = 1.0
    n_clipped = 0
    if r² >= r2_threshold
        s² = v̄ * (1 - r²) / r² / σ²_gen
    elseif ρ̄ <= 1 - floor
        s² = 1 - ρ̄
    else
        c = sqrt((1 - floor) / ρ̄)
        s² = floor
        n_clipped = T
    end
    β_eff = c .* β
    g = α .+ β_eff .* g_m .+ (c * δ) .* f .+ sqrt(s²) .* (ε .- ε_center)
    return (g=collect(Float64, g), β_eff=collect(Float64, β_eff), n_clipped=n_clipped)
end

"""
    fit(::Type{SectorFactorModel}, tickers, prices, market, sector; kwargs...) → (model, marginals)

Fit a `SectorFactorModel`. `tickers` and the columns of `prices` include the market ticker `market`;
`sector` holds one label per non-market ticker, in the order those tickers appear in `tickers`.
Every sector needs at least two names.

Steps: growth rates `G = excess_growth_rates(prices; rf, dt)`; single-index OLS of each name on the
market → `α̂`, `β̂`, `r²`, residuals `E`; `sector_factors` and `sector_loadings` of `E`; intercepts
`α = ᾱ + intercept_pull·(α̂ − ᾱ)`; a Jump-HMM for the market (its prices), for each name (its prices),
and for each sector (the compounded sector factor, `100·exp(cumsum(F·dt))`, fit with `rf = 0`).
Variance targets are the real variances. `ε_center[i]` is the mean of one `center_steps`-day draw of
name model `i` seeded with `center_seed + i`.

Keywords: `rf=0.0`, `N=100`, `ν=5.0`, `dt=1/252`, `min_obs=2` (as for `JumpHiddenMarkovModel`);
`intercept_pull=1.0` (1 keeps `α̂`, 0 sets every α to the mean); `β_halflife=126.0` (trading days);
`β_scale=nothing` (zeros: fixed β); `floor=0.10`; `r2_threshold=0.80`; `center_steps=1_000_000`;
`center_seed=4_000_000`.

Returns the model and a `Dict` of the name models keyed by ticker.
"""
function fit(::Type{SectorFactorModel}, tickers::Vector{String}, prices::AbstractMatrix{<:Real},
             market::String, sector::Vector{String};
             rf::Float64=0.0, N::Int=100, ν::Float64=5.0, dt::Float64=1/252, min_obs::Int=2,
             intercept_pull::Float64=1.0, β_halflife::Float64=126.0,
             β_scale::Union{Nothing,Vector{Float64}}=nothing,
             floor::Float64=0.10, r2_threshold::Float64=0.80,
             center_steps::Int=1_000_000, center_seed::Int=4_000_000)

    size(prices, 2) == length(tickers) ||
        throw(ArgumentError("tickers length $(length(tickers)) must match columns of prices $(size(prices, 2))"))
    m = findfirst(==(market), tickers)
    m === nothing && throw(ArgumentError("market ticker '$(market)' not found in tickers"))
    names_idx = [j for j in eachindex(tickers) if j != m]
    names = tickers[names_idx]
    length(sector) == length(names) ||
        throw(ArgumentError("sector has $(length(sector)) labels but there are $(length(names)) non-market tickers"))

    # growth rates and the single-index fit
    G_all = excess_growth_rates(prices; rf=rf, dt=dt)
    g_m = G_all[:, m]
    G = G_all[:, names_idx]
    X = hcat(ones(length(g_m)), g_m)
    α̂ = zeros(length(names)); β̂ = zeros(length(names)); r² = zeros(length(names))
    E = similar(G)
    for i in eachindex(names)
        coef = X \ G[:, i]
        α̂[i], β̂[i] = coef[1], coef[2]
        E[:, i] = G[:, i] .- X * coef
        r²[i] = 1 - var(E[:, i]) / var(G[:, i])
    end

    # sector factors and loadings (throws for one-name sectors or non-positive covariance)
    sf = sector_factors(E, sector)
    δ = sector_loadings(E, sector)
    sector_index = [findfirst(==(s), sf.sectors) for s in sector]

    # intercepts, optionally pulled toward the universe mean
    ᾱ = mean(α̂)
    α = ᾱ .+ intercept_pull .* (α̂ .- ᾱ)

    # Jump-HMMs: market, names, sectors
    market_model = fit(JumpHiddenMarkovModel, prices[:, m]; rf=rf, N=N, ν=ν, dt=dt, min_obs=min_obs)
    name_models = [fit(JumpHiddenMarkovModel, prices[:, j]; rf=rf, N=N, ν=ν, dt=dt, min_obs=min_obs)
                   for j in names_idx]
    sector_levels = 100.0 .* exp.(vcat(zeros(1, size(sf.F, 2)), cumsum(sf.F .* dt; dims=1)))
    sector_models = [fit(JumpHiddenMarkovModel, sector_levels[:, s]; rf=0.0, N=N, ν=ν, dt=dt, min_obs=min_obs)
                     for s in eachindex(sf.sectors)]

    # variance targets and the long-run mean of each name model
    σ²_m = var(g_m)
    σ²_s = vec(var(sf.F; dims=1))
    σ²_gen = vec(var(G; dims=1))
    ε_center = [mean(simulate(name_models[i], center_steps; n_paths=1, seed=center_seed + i).paths[1].observations)
                for i in eachindex(names)]

    scale = β_scale === nothing ? zeros(length(names)) : β_scale
    length(scale) == length(names) ||
        throw(ArgumentError("β_scale has $(length(scale)) entries but there are $(length(names)) names"))

    model = SectorFactorModel(names, sector, sf.sectors, sector_index, α, β̂, δ, r², σ²_m, σ²_s, σ²_gen,
                              ε_center, β_halflife, scale, floor, r2_threshold, market_model,
                              sector_models, name_models, sector_levels)
    marginals = Dict{String,JumpHiddenMarkovModel}(names[i] => name_models[i] for i in eachindex(names))
    return (model, marginals)
end
