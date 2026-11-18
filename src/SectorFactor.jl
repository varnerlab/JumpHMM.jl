"""
    ema_rate(h::Float64) -> Float64

Compute the weight `a = 1 - 2^(-1/h)` of an exponential moving average with half-life `h`, in trading days.
After `h` days, a shock keeps half of its weight in the average.

### Arguments
- `h::Float64`: The half-life, in trading days. Must be above `0`

### Returns
- `Float64`: The weight `a` of the newest value in the moving average

### Example
```julia
a = JumpHMM.ema_rate(126.0);
```
"""
function ema_rate(h::Float64)::Float64

    # check: the half-life is positive -
    if (h > 0) == false
        throw(ArgumentError("the half-life h = $(h) must be positive"));
    end

    # return -
    return 1.0 - 2.0^(-1.0 / h);
end

"""
    drift_path(z::Array{Float64,1}, h::Float64, k::Float64) -> Array{Float64,1}

Compute the drift of `β` from the standard normal shocks `z`: an exponential moving average of the shocks, scaled by `k`, with half-life `h`.
With `a = ema_rate(h)`, the first value is `x[1] = k sqrt(a/(2 - a)) z[1]`, and each later value is `x[t] = (1 - a) x[t-1] + a k z[t]`.
The first value is drawn from the stationary distribution. Every value then has the variance `drift_variance(h, k)`.
The drift is proportional to `k`.

### Arguments
- `z::Array{Float64,1}`: The standard normal shocks, one per day
- `h::Float64`: The half-life of the drift, in trading days. Must be above `0`
- `k::Float64`: The drift scale. A value of `0.0` gives a drift of zero

### Returns
- `Array{Float64,1}`: The drift on each day. The array has the length of `z`

### Example
```julia
x = JumpHMM.drift_path(randn(252), 126.0, 0.3);
```
"""
function drift_path(z::Array{Float64,1}, h::Float64, k::Float64)::Array{Float64,1}

    # initialize -
    a = ema_rate(h);
    x = zeros(Float64, length(z));

    # no shocks: return the empty drift -
    if isempty(z) == true
        return x;
    end

    # day 1: draw the drift from its stationary distribution -
    x[1] = k * sqrt(a / (2 - a)) * z[1];

    # main loop: each day, the drift keeps (1 - a) of its old value and adds a times the scaled shock -
    for t ∈ 2:length(z)
        x[t] = (1 - a) * x[t-1] + a * k * z[t];
    end

    # return -
    return x;
end

"""
    drift_variance(h::Float64, k::Float64) -> Float64

Compute the stationary variance `k^2 a/(2 - a)` of `drift_path` with half-life `h` and scale `k`, where `a = ema_rate(h)`.

### Arguments
- `h::Float64`: The half-life of the drift, in trading days. Must be above `0`
- `k::Float64`: The drift scale

### Returns
- `Float64`: The variance of the drift on any day

### Example
```julia
v = JumpHMM.drift_variance(126.0, 0.3);
```
"""
function drift_variance(h::Float64, k::Float64)::Float64

    # the weight of the moving average -
    a = ema_rate(h);

    # return -
    return k^2 * a / (2 - a);
end

"""
    _sector_members(sector::Array{String,1}, label::String) -> Array{Int64,1}

Find the positions of the assets in the sector `label`.

### Arguments
- `sector::Array{String,1}`: The sector label of each asset
- `label::String`: The sector to look for

### Returns
- `Array{Int64,1}`: The positions `i` with `sector[i] == label`, in increasing order

### Example
```julia
members = JumpHMM._sector_members(["Energy", "Utilities", "Energy"], "Energy"); # [1, 3]
```
"""
function _sector_members(sector::Array{String,1}, label::String)::Array{Int64,1}

    # initialize -
    members = Array{Int64,1}();

    # main loop: keep each asset whose label matches -
    for i ∈ eachindex(sector)
        if sector[i] == label
            push!(members, i);
        end
    end

    # return -
    return members;
end

"""
    sector_factors(E::Array{Float64,2}, sector::Array{String,1}) -> NamedTuple

Compute the equal-weight factor of each sector from the single-index residuals `E`. The factor of a sector on a day
is the mean residual of the assets in that sector.

### Arguments
- `E::Array{Float64,2}`: The single-index residuals, one row per day and one column per asset
- `sector::Array{String,1}`: The sector label of each asset, one per column of `E`

### Returns
- `NamedTuple`: The sectors and their factors.
    - `sectors::Array{String,1}`: The sorted unique sector labels
    - `F::Array{Float64,2}`: The sector factors, one row per day and one column per sector, in the order of `sectors`

### Example
```julia
sf = JumpHMM.sector_factors(E, ["Energy", "Energy", "Utilities", "Utilities"]);
```
"""
function sector_factors(E::Array{Float64,2}, sector::Array{String,1})::NamedTuple

    # check: one label for each column -
    if size(E, 2) != length(sector)
        throw(ArgumentError("E has $(size(E, 2)) columns but sector has $(length(sector)) labels"));
    end

    # initialize: the sorted unique labels -
    sectors = sort(unique(sector));
    F = zeros(Float64, size(E, 1), length(sectors));

    # main loop: the factor of each sector is the mean of the residuals of its assets -
    for s ∈ eachindex(sectors)
        members = _sector_members(sector, sectors[s]);
        F[:, s] = vec(mean(view(E, :, members); dims = 2));
    end

    # return -
    return (sectors = sectors, F = F);
end

"""
    loo_loadings(E::Array{Float64,2}, sector::Array{String,1}) -> Array{Float64,1}

Compute the leave-one-out sector loading of each asset. The loading `δ[i]` is the slope of an ordinary least squares fit,
with an intercept, of column `i` of `E` on the mean of the other columns in the same sector. We leave column `i` out of the mean,
so that its own noise does not enter the regressor. We throw an `ArgumentError` for a sector with one asset.

### Arguments
- `E::Array{Float64,2}`: The single-index residuals, one row per day and one column per asset
- `sector::Array{String,1}`: The sector label of each asset, one per column of `E`

### Returns
- `Array{Float64,1}`: The leave-one-out sector loading of each asset

### Example
```julia
δ = JumpHMM.loo_loadings(E, ["Energy", "Energy", "Utilities", "Utilities"]);
```
"""
function loo_loadings(E::Array{Float64,2}, sector::Array{String,1})::Array{Float64,1}

    # check: one label for each column -
    if size(E, 2) != length(sector)
        throw(ArgumentError("E has $(size(E, 2)) columns but sector has $(length(sector)) labels"));
    end

    # initialize -
    δ = zeros(Float64, size(E, 2));

    # main loop: one sector at a time, in sorted label order -
    for label ∈ sort(unique(sector))

        # check: the sector has at least two assets -
        members = _sector_members(sector, label);
        if length(members) < 2
            throw(ArgumentError("sector $(label) has one asset; a leave-one-out mean needs at least two"));
        end

        # the loading of each asset in the sector -
        for i ∈ members

            # the other assets in the sector -
            others = Array{Int64,1}();
            for k ∈ members
                if k != i
                    push!(others, k);
                end
            end

            # regress the residual of asset i on the mean residual of the others -
            x = vec(mean(view(E, :, others); dims = 2));
            y = view(E, :, i);
            x̄ = mean(x);
            ȳ = mean(y);
            δ[i] = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2);
        end
    end

    # return -
    return δ;
end

"""
    sector_loadings(E::Array{Float64,2}, sector::Array{String,1}) -> Array{Float64,1}

Compute the sector loading `δ[i]` of each asset for the sector factor model. We multiply the `loo_loadings` of each sector by one scale,
chosen so that the co-movement the model builds inside the sector matches the data. The model gives two assets `i` and `j`
of a sector the covariance `δ[i] δ[j] var(f_s)`, where `f_s` is the sector factor from `sector_factors`. The scale makes the mean of
this covariance over the pairs `i ≠ j` equal the mean pairwise covariance of the residuals of the sector.
We throw an `ArgumentError` (naming the sector) for a sector with one asset, or when either mean pairwise covariance is not positive.

### Arguments
- `E::Array{Float64,2}`: The single-index residuals, one row per day and one column per asset
- `sector::Array{String,1}`: The sector label of each asset, one per column of `E`

### Returns
- `Array{Float64,1}`: The sector loading of each asset

### Example
```julia
δ = JumpHMM.sector_loadings(E, ["Energy", "Energy", "Utilities", "Utilities"]);
```
"""
function sector_loadings(E::Array{Float64,2}, sector::Array{String,1})::Array{Float64,1}

    # initialize: the leave-one-out loadings and the sector factors -
    δ = loo_loadings(E, sector);
    sf = sector_factors(E, sector);

    # main loop: scale the loadings of one sector at a time -
    for s ∈ eachindex(sf.sectors)
        label = sf.sectors[s];
        members = _sector_members(sector, label);
        n = length(members);
        var_f = var(view(sf.F, :, s));

        # the sum of the residual variances of the sector. We add them one at a time, in member order,
        # so the sum equals the earlier version -
        sum_of_variances = 0.0;
        for i ∈ members
            sum_of_variances += var(view(E, :, i));
        end

        # the mean pairwise covariance of the residuals. The sum of the residuals is n f_s, and the covariances
        # over the pairs i ≠ j add up to the variance of that sum minus the sum of the variances -
        real_cov = (n^2 * var_f - sum_of_variances) / (n * (n - 1));
        if (real_cov > 0) == false
            throw(ArgumentError("sector $(label) has mean pairwise residual covariance $(real_cov); it must be positive"));
        end

        # the same mean for the leave-one-out loadings: the products δ[i] δ[j] over the pairs i ≠ j add up to
        # the square of the sum of the loadings minus the sum of their squares -
        model_cov = (sum(δ[members])^2 - sum(δ[members] .^ 2)) * var_f / (n * (n - 1));
        if (model_cov > 0) == false
            throw(ArgumentError("sector $(label) leave-one-out loadings give mean pairwise covariance $(model_cov); it must be positive"));
        end

        # multiply every loading of the sector by one scale, so that the two means agree -
        scale = sqrt(real_cov / model_cov);
        for i ∈ members
            δ[i] = δ[i] * scale;
        end
    end

    # return -
    return δ;
end

"""
    own_residuals(E::Array{Float64,2}, F::Array{Float64,2}, δ::Array{Float64,1}, sector_index::Array{Int64,1}) -> Array{Float64,2}

Compute the residual each asset has left after its sector term: column `i` is `E[:, i] - δ[i] F[:, sector_index[i]]`.
The own model of each asset is fit to this series.

### Arguments
- `E::Array{Float64,2}`: The single-index residuals, one row per day and one column per asset
- `F::Array{Float64,2}`: The sector factors, one row per day and one column per sector
- `δ::Array{Float64,1}`: The sector loading of each asset
- `sector_index::Array{Int64,1}`: The column of `F` that holds the sector factor of each asset

### Returns
- `Array{Float64,2}`: The leftover residuals, one row per day and one column per asset

### Example
```julia
U = JumpHMM.own_residuals(E, sf.F, δ, [1, 1, 2, 2]);
```
"""
function own_residuals(E::Array{Float64,2}, F::Array{Float64,2}, δ::Array{Float64,1},
    sector_index::Array{Int64,1})::Array{Float64,2}

    # check: one loading and one sector position for each column of E -
    if size(E, 2) != length(δ) || size(E, 2) != length(sector_index)
        throw(ArgumentError("E has $(size(E, 2)) columns, δ $(length(δ)) entries and sector_index $(length(sector_index))"));
    end

    # initialize -
    U = zeros(Float64, size(E));

    # main loop: subtract the sector term of each asset -
    for i ∈ 1:size(E, 2)
        U[:, i] = E[:, i] .- δ[i] .* F[:, sector_index[i]];
    end

    # return -
    return U;
end

"""
    own_variances(E::Array{Float64,2}, F::Array{Float64,2}, U::Array{Float64,2}, δ::Array{Float64,1},
        sector_index::Array{Int64,1}; tickers::Array{String,1}) -> NamedTuple

Compute the two variances that size the own draw of each asset. The own variance `σ²_own[i] = var(E[:, i]) - δ[i]^2 var(F[:, s])` is the part of
the single-index residual variance that the sector term leaves to the own term, where `s = sector_index[i]`.
The leftover variance `σ²_u[i] = var(U[:, i])` is the variance of the leftover residual from `own_residuals`. We throw an `ArgumentError` (naming the ticker)
when the sector term carries the whole residual variance of an asset.

### Arguments
- `E::Array{Float64,2}`: The single-index residuals, one row per day and one column per asset
- `F::Array{Float64,2}`: The sector factors, one row per day and one column per sector
- `U::Array{Float64,2}`: The leftover residuals from `own_residuals`
- `δ::Array{Float64,1}`: The sector loading of each asset
- `sector_index::Array{Int64,1}`: The column of `F` that holds the sector factor of each asset
- `tickers::Array{String,1}`: A required keyword. The ticker of each asset, used in the error message

### Returns
- `NamedTuple`: The two variances of each asset.
    - `σ²_own::Array{Float64,1}`: The residual variance left to the own term
    - `σ²_u::Array{Float64,1}`: The variance of the leftover residual

### Example
```julia
variances = JumpHMM.own_variances(E, sf.F, U, δ, [1, 1, 2, 2]; tickers = ["XOM", "CVX", "DUK", "SO"]);
```
"""
function own_variances(E::Array{Float64,2}, F::Array{Float64,2}, U::Array{Float64,2}, δ::Array{Float64,1},
    sector_index::Array{Int64,1}; tickers::Array{String,1})::NamedTuple

    # initialize -
    number_of_assets = size(E, 2);
    σ²_own = zeros(Float64, number_of_assets);
    σ²_u = zeros(Float64, number_of_assets);

    # main loop: the two variances of each asset -
    for i ∈ 1:number_of_assets

        # the residual variance the sector term leaves to the own term -
        σ²_own[i] = var(view(E, :, i)) - δ[i]^2 * var(view(F, :, sector_index[i]));
        if σ²_own[i] <= 0
            throw(ArgumentError("asset $(tickers[i]): the sector term carries the whole residual variance (left over: $(σ²_own[i]))"));
        end

        # the variance of the leftover residual -
        σ²_u[i] = var(view(U, :, i));
    end

    # return -
    return (σ²_own = σ²_own, σ²_u = σ²_u);
end

"""
    own_scale(σ²_own::Float64, σ²_u::Float64, σ²_drift::Float64; ticker::String = "") -> Float64

Compute the scale `ψ` of the own draw of one asset: `ψ^2 = (σ²_own - σ²_drift) / σ²_u`, with `σ²_own` and `σ²_u` from `own_variances`.
The drift variance `σ²_drift` is the variance that a drifting `β` adds to the residual of the asset: `drift_variance(β_halflife, β_scale)` times the
variance of the market growth rate. The drift variance is zero for a fixed `β`.

The single-index residual comes from a fit with a fixed `β`. That residual already holds the movement of the stock's actual `β` over time. We subtract
`σ²_drift`, so that the synthetic drift does not add that variance a second time. Then the drift, sector, and own terms together
carry the variance of the residual. The scale `ψ` also differs from `1`, because the sector factor averages the residuals of its assets,
asset `i` included. As a result, part of the leftover residual of asset `i` still moves with the sector factor. Independent draws lose that
shared part, and `ψ` puts that variance back. We throw an `ArgumentError` (naming `ticker`) when the variance of the drift fills the whole own variance.

### Arguments
- `σ²_own::Float64`: The residual variance left to the own term
- `σ²_u::Float64`: The variance of the leftover residual
- `σ²_drift::Float64`: The variance that the drift of `β` adds to the residual

### Returns
- `Float64`: The scale `ψ` of the own draw

### Optional Arguments
- `ticker::String`: The ticker of the asset, used in the error message. Default: `""`

### Example
```julia
ψ = JumpHMM.own_scale(0.04, 0.03, 0.0; ticker = "XOM");
```
"""
function own_scale(σ²_own::Float64, σ²_u::Float64, σ²_drift::Float64; ticker::String = "")::Float64

    # check: the drift leaves some variance to the own draw -
    room = σ²_own - σ²_drift;
    if room <= 0
        throw(ArgumentError("asset $(ticker): the β drift's variance $(σ²_drift) fills the own variance $(σ²_own)"));
    end

    # return -
    return sqrt(room / σ²_u);
end

"""
    own_scales(model::MySectorFactorModel; β_halflife::Float64 = model.β_halflife,
        β_scale::Array{Float64,1} = model.β_scale) -> Array{Float64,1}

Compute the `own_scale` of every asset of `model` under the given drift of `β`. With the default drift, the result equals `model.ψ`.

### Arguments
- `model::MySectorFactorModel`: The sector factor model

### Returns
- `Array{Float64,1}`: The scale `ψ` of the own draw of each asset

### Optional Arguments
- `β_halflife::Float64`: The half-life of the drift of `β`, in trading days. Default: `model.β_halflife`
- `β_scale::Array{Float64,1}`: The drift scale of `β` for each asset. Default: `model.β_scale`

### Example
```julia
ψ = JumpHMM.own_scales(model; β_scale = fill(0.3, length(model.tickers)));
```
"""
function own_scales(model::MySectorFactorModel; β_halflife::Float64 = model.β_halflife,
    β_scale::Array{Float64,1} = model.β_scale)::Array{Float64,1}

    # initialize -
    number_of_assets = length(model.tickers);
    ψ = zeros(Float64, number_of_assets);

    # main loop: the variance the drift adds to each asset, then the scale of its own draw -
    for i ∈ 1:number_of_assets
        σ²_drift = drift_variance(β_halflife, β_scale[i]) * model.σ²_m;
        ψ[i] = own_scale(model.σ²_own[i], model.σ²_u[i], σ²_drift; ticker = model.tickers[i]);
    end

    # return -
    return ψ;
end

"""
    sample_draws(model::MySectorFactorModel, number_of_steps::Int64; seed::Int64) -> NamedTuple

Draw every random part of one path of the universe, before we compose the growth rates of the assets: one market path, one path of
each sector model, one path of each own model, and the standard normal shocks of the drift of `β`.
The path that `sample_dependence` draws is `compose(model, sample_draws(model, number_of_steps; seed = seed))`.
We keep the draws, so that we can compose the same random draws again under other parameters (`with_parameters`, `compose_name`).

Each part has its own seed: the market uses `seed`, sector `s` uses `10^12 + 10^4 * seed + s`, and asset `j` uses
`2 * 10^12 + 10^4 * seed + j`. The shocks of the drift come from their own generator, `Xoshiro(3 * 10^12 + seed)`.
These streams stay apart for `0 ≤ seed < 10^8` and fewer than `10^4` assets and sectors. Outside that range, we throw an
`ArgumentError`. Like `simulate`, this function reseeds the global random number generator.

### Arguments
- `model::MySectorFactorModel`: The sector factor model
- `number_of_steps::Int64`: The number of days to draw. Must be at least `1`
- `seed::Int64`: A required keyword. The seed of the draws, with `0 ≤ seed < 10^8`

### Returns
- `NamedTuple`: The draws, one row per day.
    - `g_m::Array{Float64,1}`: The market growth rates
    - `m_states::Array{Int64,1}`: The state of the market model on each day
    - `episodes::Array{Bool,1}`: A flag that is `true` if the day is part of a market jump episode
    - `F::Array{Float64,2}`: The sector factors, one column per sector
    - `sector_episodes::Array{Bool,2}`: A flag that is `true` if the day is part of a jump episode of the sector model, one column per sector
    - `E::Array{Float64,2}`: The own draws, one column per asset
    - `own_episodes::Array{Bool,2}`: A flag that is `true` if the day is part of a jump episode of the own model, one column per asset
    - `Z::Array{Float64,2}`: The standard normal shocks of the drift of `β`, one column per asset

### Example
```julia
draws = JumpHMM.sample_draws(model, 252; seed = 5660);
```
"""
function sample_draws(model::MySectorFactorModel, number_of_steps::Int64; seed::Int64)::NamedTuple

    # check: the number of days, and the range that keeps the random streams apart -
    if number_of_steps < 1
        throw(ArgumentError("number_of_steps must be at least 1, got $(number_of_steps)"));
    end
    if seed < 0 || seed >= 10^8
        throw(ArgumentError("seed must satisfy 0 ≤ seed < 10^8 so the random streams stay apart; got $(seed)"));
    end
    if length(model.tickers) >= 10^4 || length(model.sectors) >= 10^4
        throw(ArgumentError("the seeds stay apart only for fewer than 10^4 assets and sectors"));
    end

    # the market path, seeded with seed -
    market_path = simulate(model.market_model, number_of_steps; number_of_paths = 1, seed = seed);

    # initialize -
    number_of_sectors = length(model.sectors);
    number_of_assets = length(model.tickers);
    F = zeros(Float64, number_of_steps, number_of_sectors);
    sector_episodes = fill(false, number_of_steps, number_of_sectors);

    # one path of each sector model: sector s uses the seed 10^12 + 10^4 * seed + s -
    for s ∈ 1:number_of_sectors
        sector_path = simulate(model.sector_models[s], number_of_steps; number_of_paths = 1,
            seed = 10^12 + 10^4 * seed + s);
        F[:, s] = sector_path.growth_rates[:, 1];
        sector_episodes[:, s] = sector_path.episodes[:, 1];
    end

    # the standard normal shocks of the drift of β, from their own random number generator -
    Z = randn(Xoshiro(3 * 10^12 + seed), number_of_steps, number_of_assets);

    # one path of each own model: asset j uses the seed 2 * 10^12 + 10^4 * seed + j -
    E = zeros(Float64, number_of_steps, number_of_assets);
    own_episodes = fill(false, number_of_steps, number_of_assets);
    for j ∈ 1:number_of_assets
        own_path = simulate(model.own_models[j], number_of_steps; number_of_paths = 1,
            seed = 2 * 10^12 + 10^4 * seed + j);
        E[:, j] = own_path.growth_rates[:, 1];
        own_episodes[:, j] = own_path.episodes[:, 1];
    end

    # return -
    return (g_m = market_path.growth_rates[:, 1], m_states = market_path.states[:, 1], episodes = market_path.episodes[:, 1],
        F = F, sector_episodes = sector_episodes, E = E, own_episodes = own_episodes, Z = Z);
end

"""
    compose_name(α::Array{Float64,1}, β::Array{Float64,1}, δ::Float64, g_m::Array{Float64,1}, f::Array{Float64,1},
        ε::Array{Float64,1}; ψ::Float64) -> NamedTuple

Compose the daily growth rate of one asset: `g[t] = α[t] + β[t] g_m[t] + δ f[t] + ψ ε[t]`.
The own draw `ε` comes from the model of the residual of the asset after its sector term. That residual averages zero in the data,
so the own draw is not centered.

### Arguments
- `α::Array{Float64,1}`: The intercept on each day, in 1/year
- `β::Array{Float64,1}`: The sensitivity to the market on each day
- `δ::Float64`: The sector loading of the asset
- `g_m::Array{Float64,1}`: The market draw, in 1/year
- `f::Array{Float64,1}`: The draw of the sector factor of the asset, in 1/year
- `ε::Array{Float64,1}`: The own draw of the asset, in 1/year
- `ψ::Float64`: A required keyword. The scale of the own draw, at least `0` (`own_scale`)

### Returns
- `NamedTuple`: The composed asset.
    - `g::Array{Float64,1}`: The growth rate on each day, in 1/year
    - `β_path::Array{Float64,1}`: A copy of `β`, the sensitivity to the market on each day

### Example
```julia
out = JumpHMM.compose_name(fill(0.05, 252), fill(1.1, 252), 0.8, g_m, f, ε; ψ = 1.2);
```
"""
function compose_name(α::Array{Float64,1}, β::Array{Float64,1}, δ::Float64, g_m::Array{Float64,1},
    f::Array{Float64,1}, ε::Array{Float64,1}; ψ::Float64)::NamedTuple

    # check: every path has one value per day, and the scale is not negative -
    T = length(g_m);
    if length(α) != T || length(β) != T || length(f) != T || length(ε) != T
        throw(ArgumentError("α, β, g_m, f and ε must share one length; got $(length(α)), $(length(β)), $(T), $(length(f)), $(length(ε))"));
    end
    if (ψ >= 0) == false
        throw(ArgumentError("ψ must be non-negative; got $(ψ)"));
    end

    # compose the growth rate of each day -
    g = α .+ β .* g_m .+ δ .* f .+ ψ .* ε;

    # return -
    return (g = g, β_path = copy(β));
end

"""
    compose_name(model::MySectorFactorModel, draws::NamedTuple, j::Int64; α::Union{Float64,Nothing} = nothing,
        β_halflife::Float64 = model.β_halflife, β_scale::Union{Float64,Nothing} = nothing) -> NamedTuple

Compose asset `j` alone from `draws` (`sample_draws`). The result is column `j` of `compose(model, draws)`, computed without composing
the other assets. The keywords replace the intercept, the half-life of the drift, and the drift scale of asset `j` for this call only,
as `with_parameters` would for the whole model. The own scale follows the drift (`own_scale`). We use `compose_name` to fit the parameters of
one asset many times on fixed draws.

### Arguments
- `model::MySectorFactorModel`: The sector factor model
- `draws::NamedTuple`: The draws. We read the keys `g_m`, `F`, `E`, and `Z`
- `j::Int64`: The position of the asset in `model.tickers`

### Returns
- `NamedTuple`: The composed asset.
    - `g::Array{Float64,1}`: The growth rate on each day, in 1/year
    - `β_path::Array{Float64,1}`: The sensitivity to the market on each day

### Optional Arguments
- `α::Union{Float64,Nothing}`: The intercept of asset `j`, in 1/year. If `nothing`, we use `model.α[j]`. Default: `nothing`
- `β_halflife::Float64`: The half-life of the drift of `β`, in trading days. Default: `model.β_halflife`
- `β_scale::Union{Float64,Nothing}`: The drift scale of `β` for asset `j`. If `nothing`, we use `model.β_scale[j]`. Default: `nothing`

### Example
```julia
draws = JumpHMM.sample_draws(model, 252; seed = 5660);
out = JumpHMM.compose_name(model, draws, 1; β_scale = 0.5);
```
"""
function compose_name(model::MySectorFactorModel, draws::NamedTuple, j::Int64; α::Union{Float64,Nothing} = nothing,
    β_halflife::Float64 = model.β_halflife, β_scale::Union{Float64,Nothing} = nothing)::NamedTuple

    # check: j is one of the assets -
    number_of_assets = length(model.tickers);
    if j < 1 || j > number_of_assets
        throw(ArgumentError("asset index $(j) is outside 1:$(number_of_assets)"));
    end

    # use the intercept and the drift scale of the model, unless we were given others -
    α_j = model.α[j];
    if isnothing(α) == false
        α_j = α;
    end
    β_scale_j = model.β_scale[j];
    if isnothing(β_scale) == false
        β_scale_j = β_scale;
    end

    # initialize -
    number_of_steps = length(draws.g_m);
    s = model.sector_index[j];

    # the β of each day: the fitted β plus the drift built from the shocks of asset j -
    β = model.β[j] .+ drift_path(draws.Z[:, j], β_halflife, β_scale_j);

    # the scale of the own draw under this drift -
    σ²_drift = drift_variance(β_halflife, β_scale_j) * model.σ²_m;
    ψ = own_scale(model.σ²_own[j], model.σ²_u[j], σ²_drift; ticker = model.tickers[j]);

    # compose the market, sector, and own draws of asset j -
    return compose_name(fill(α_j, number_of_steps), β, model.δ[j], draws.g_m, draws.F[:, s], draws.E[:, j];
        ψ = ψ);
end

"""
    compose(model::MySectorFactorModel, draws::NamedTuple) -> NamedTuple

Compose the growth rates of every asset of one path from its `draws` (`sample_draws`), with the intercepts, the `β` and its drift,
the sector loadings, and the own scales of `model`. The same draws and model always give the same result.
Because we read only the keys `g_m`, `episodes`, `F`, `sector_episodes`, `E`, `own_episodes`, and `Z`, `draws` can mix the parts of
different draws, for example the market from one draw and the sectors from another.

### Arguments
- `model::MySectorFactorModel`: The sector factor model
- `draws::NamedTuple`: The draws, as from `sample_draws`

### Returns
- `NamedTuple`: The composed universe, one row per day and one column per asset.
    - `G::Array{Float64,2}`: The growth rates, in 1/year
    - `β_path::Array{Float64,2}`: The sensitivity to the market on each day
    - `name_episodes::Array{Bool,2}`: A flag that is `true` if the market, the sector, or the own draw of the asset was in a jump episode that day

### Example
```julia
draws = JumpHMM.sample_draws(model, 252; seed = 5660);
universe = JumpHMM.compose(model, draws);
```
"""
function compose(model::MySectorFactorModel, draws::NamedTuple)::NamedTuple

    # check: the draws have one column per asset -
    (number_of_steps, number_of_assets) = size(draws.E);
    if number_of_assets != length(model.tickers)
        throw(ArgumentError("draws have $(number_of_assets) assets but the model has $(length(model.tickers))"));
    end

    # initialize -
    G = zeros(Float64, number_of_steps, number_of_assets);
    β_path = zeros(Float64, number_of_steps, number_of_assets);
    name_episodes = fill(false, number_of_steps, number_of_assets);

    # main loop: compose one asset at a time -
    for j ∈ 1:number_of_assets
        out = compose_name(model, draws, j);
        s = model.sector_index[j];
        G[:, j] = out.g;
        β_path[:, j] = out.β_path;

        # a day is part of an episode for asset j if its market, sector, or own draw was in a jump episode -
        name_episodes[:, j] = draws.episodes .| draws.sector_episodes[:, s] .| draws.own_episodes[:, j];
    end

    # return -
    return (G = G, β_path = β_path, name_episodes = name_episodes);
end

"""
    with_parameters(model::MySectorFactorModel; α::Array{Float64,1} = model.α, β_halflife::Float64 = model.β_halflife,
        β_scale::Array{Float64,1} = model.β_scale) -> MySectorFactorModel

Make a new sector factor model with new intercepts `α`, a new half-life of the drift of `β`, and new drift scales. We recompute the
own scales `ψ` for the new drift (`own_scales`). The new model shares every other field with `model`, including the fitted hidden
Markov models and the level series. Changing one of those shared fields in place changes both models. The input model is not changed.

Make a `deepcopy` of `model` before a study that edits its fitted hidden Markov models, because the new model shares them.

### Arguments
- `model::MySectorFactorModel`: The sector factor model

### Returns
- `MySectorFactorModel`: The new model

### Optional Arguments
- `α::Array{Float64,1}`: The intercept of each asset, in 1/year. Default: `model.α`
- `β_halflife::Float64`: The half-life of the drift of `β`, in trading days, above `0`. Default: `model.β_halflife`
- `β_scale::Array{Float64,1}`: The drift scale of `β` for each asset, each at least `0`. Zeros keep `β` fixed. Default: `model.β_scale`

### Example
```julia
drifting = JumpHMM.with_parameters(model; β_scale = fill(0.3, length(model.tickers)), β_halflife = 126.0);
```
"""
function with_parameters(model::MySectorFactorModel; α::Array{Float64,1} = model.α, β_halflife::Float64 = model.β_halflife,
    β_scale::Array{Float64,1} = model.β_scale)::MySectorFactorModel

    # check: one α and one drift scale per asset, a positive half-life, and no negative drift scale -
    number_of_assets = length(model.tickers);
    if length(α) != number_of_assets
        throw(ArgumentError("α has $(length(α)) entries but there are $(number_of_assets) assets"));
    end
    if length(β_scale) != number_of_assets
        throw(ArgumentError("β_scale has $(length(β_scale)) entries but there are $(number_of_assets) assets"));
    end
    if (β_halflife > 0) == false
        throw(ArgumentError("β_halflife must be positive, got $(β_halflife)"));
    end
    for i ∈ eachindex(β_scale)
        if (β_scale[i] >= 0) == false
            throw(ArgumentError("β_scale entries must be ≥ 0, got $(β_scale[i]) at position $(i)"));
        end
    end

    # build an empty model -
    m = MySectorFactorModel();

    # the fit of the model: shared with the input model -
    m.tickers = model.tickers;
    m.sector = model.sector;
    m.sectors = model.sectors;
    m.sector_index = model.sector_index;
    m.β = model.β;
    m.δ = model.δ;
    m.r² = model.r²;
    m.σ²_own = model.σ²_own;
    m.σ²_u = model.σ²_u;
    m.σ²_m = model.σ²_m;

    # the new parameters: copies of the given intercepts and drift scales, and the new half-life -
    m.α = copy(α);
    m.β_halflife = β_halflife;
    m.β_scale = copy(β_scale);

    # the own scales under the new drift -
    m.ψ = own_scales(model; β_halflife = β_halflife, β_scale = β_scale);

    # the fitted hidden Markov models and the level series: shared with the input model, not copied -
    m.market_model = model.market_model;
    m.sector_models = model.sector_models;
    m.own_models = model.own_models;
    m.sector_levels = model.sector_levels;
    m.own_levels = model.own_levels;

    # return -
    return m;
end

"""
    sample_dependence(model::MySectorFactorModel, number_of_steps::Int64; seed::Int64) -> NamedTuple

Draw one synthetic path of the whole universe: `compose(model, sample_draws(model, number_of_steps; seed = seed))`.
The seeds are those of `sample_draws`. The same `seed` repeats the path exactly.

### Arguments
- `model::MySectorFactorModel`: The sector factor model
- `number_of_steps::Int64`: The number of days to draw. Must be at least `1`
- `seed::Int64`: A required keyword. The seed of the draws, with `0 ≤ seed < 10^8`

### Returns
- `NamedTuple`: The path, one row per day.
    - `g_m::Array{Float64,1}`: The market growth rates
    - `m_states::Array{Int64,1}`: The state of the market model on each day
    - `episodes::Array{Bool,1}`: A flag that is `true` if the day is part of a market jump episode
    - `F::Array{Float64,2}`: The sector factors, one column per sector
    - `G::Array{Float64,2}`: The growth rates of the assets, one column per asset
    - `β_path::Array{Float64,2}`: The sensitivity of each asset to the market on each day
    - `name_episodes::Array{Bool,2}`: A flag that is `true` if the market, the sector, or the own draw of the asset was in a jump episode that day

### Example
```julia
path = JumpHMM.sample_dependence(model, 252; seed = 5660);
```
"""
function sample_dependence(model::MySectorFactorModel, number_of_steps::Int64; seed::Int64)::NamedTuple

    # draw every random part, then compose the assets -
    draws = sample_draws(model, number_of_steps; seed = seed);
    composed = compose(model, draws);

    # return -
    return (g_m = draws.g_m, m_states = draws.m_states, episodes = draws.episodes, F = draws.F, G = composed.G,
        β_path = composed.β_path, name_episodes = composed.name_episodes);
end
