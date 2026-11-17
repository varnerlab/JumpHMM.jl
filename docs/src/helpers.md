# Helpers

```@meta
CurrentModule = JumpHMM
```

This page describes the helper functions for growth rates, for the distance between two samples, and for the drift of the market sensitivity ``\beta`` in a sector factor model.

## Growth rates
The [`excess_growth_rates`](@ref excess_growth_rates(::Array{Float64,1})) function turns share prices into excess growth rates. For the share prices ``S_t`` and ``S_{t+1}`` on two days, one time step ``\Delta t`` apart, the excess growth rate is

```math
g_t = \frac{1}{\Delta t} \ln\left(\frac{S_{t+1}}{S_t}\right) - r_f
```

where ``\Delta t`` is in years (`Δt`, default `1/252`) and ``r_f`` is the risk-free rate in 1/year (`rf`, default `0.0`). The function has three methods:

- __Vector:__ An `Array{Float64,1}` of ``n + 1`` share prices gives ``n`` growth rates.
- __Matrix:__ An `Array{Float64,2}` of share prices, with one row per day and one column per asset, gives a matrix with one fewer row.
- __Dictionary:__ A `Dict{String,DataFrame}` of price tables and a list of tickers gives a matrix with one column per ticker, in the order of the list. The keyword `price_col` names the column of share prices (default `:close`).

The share prices must be `Float64` values. Convert an integer array with `Float64.(prices)` first.

```julia
import JumpHMM
import DataFrames

data = Dict("AAA" => DataFrames.DataFrame(close = [100.0, 101.0, 99.5]), "BBB" => DataFrames.DataFrame(close = [50.0, 50.5, 51.0]));
G = JumpHMM.excess_growth_rates(data, ["AAA", "BBB"]; rf = 0.0, Δt = 1/252);   # 2 days × 2 assets (1/year)
```

The [`prices_from_growth_rates`](@ref) function reverses the computation. It starts from a share price ``S_1`` and computes ``S_{t+1} = S_t \exp\left((g_t + r_f)\,\Delta t\right)``. With the same `rf` and `Δt`, it gives back the share prices that made the growth rates.

## Distances between two samples
Two functions measure how far apart the distributions of two samples are. The [`validate`](@ref validate(::MyJumpHiddenMarkovModel, ::Array{Float64,1})) function uses both.

The [`wasserstein_distance`](@ref) function computes the Wasserstein-1 distance, the mean absolute difference between the sorted samples. If the two samples have different lengths, we compare them at the same number of evenly spaced quantiles. The distance has the units of the samples, for example 1/year for growth rates.

The [`hellinger_distance`](@ref) function puts both samples into the same histogram bins and scales each histogram to sum to one. With bin probabilities ``p_b`` and ``q_b``, the distance is

```math
H = \sqrt{\frac{1}{2} \sum_{b} \left(\sqrt{p_b} - \sqrt{q_b}\right)^{2}}
```

The bins split the range of both samples into equal widths, with the larger of `30` and the square root of the larger sample size as the number of bins. The distance is `0.0` for the same histogram and `1.0` for histograms with no bin in common.

## Drift of β
These three functions compute the drift of ``\beta`` in a [sector factor model](sector_factor.md). They are not exported, and we call them with the package name:

- __Moving-average weight:__ The [`ema_rate`](@ref) function computes the weight ``a = 1 - 2^{-1/h}`` of an exponential moving average with a half-life of ``h`` trading days. After ``h`` days, a shock keeps half of its weight.
- __Drift path:__ The [`drift_path`](@ref) function computes the drift on each day from standard normal shocks, a half-life, and a drift scale.
- __Drift variance:__ The [`drift_variance`](@ref) function computes the variance ``k^2 a/(2 - a)`` that the drift has on every day, for the drift scale ``k``.

```julia
z = randn(252);                                  # one standard normal shock per trading day
x = JumpHMM.drift_path(z, 126.0, 0.3);           # the drift of β on each day, half-life 126.0 trading days, scale 0.3
v = JumpHMM.drift_variance(126.0, 0.3);          # the variance of the drift on any day
```

The half-life and the scale are `Float64` values. Write `126.0`, not `126`, because an `Int64` argument throws a `MethodError`.
