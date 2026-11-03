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
