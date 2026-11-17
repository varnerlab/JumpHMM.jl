# Simulation

```@meta
CurrentModule = JumpHMM
```

The [`simulate`](@ref simulate(::MyJumpHiddenMarkovModel, ::Int64)) function draws synthetic growth-rate paths from a [`MyJumpHiddenMarkovModel`](@ref). The [Jump Episodes](jump_episodes.md) page describes the steps of one path.

## Usage
The second argument is the number of days in each path, an `Int64` of at least `1`:

```julia
number_of_days = length(g);   # each path as long as the data
paths = JumpHMM.simulate(model, number_of_days;
    number_of_paths = 100,    # the number of paths (default 1000)
    start = nothing,          # the state of day 1, or nothing to draw it from the stationary distribution (default nothing)
    seed = 5660               # the seed of the random number generator, or nothing to skip reseeding (default nothing)
);
```

The result is a NamedTuple of three matrices, with one row for each day and one column for each path:

- __Growth rates:__ The `Array{Float64,2}` matrix `paths.growth_rates` holds the simulated growth rates, in 1/year.
- __States:__ The `Array{Int64,2}` matrix `paths.states` holds the state on each day.
- __Episodes:__ The `Array{Bool,2}` matrix `paths.episodes` is `true` on the days of a jump episode.

Column `p` of each matrix is path `p`. A `seed` reseeds the global random number generator. The same seed gives the same paths.

## From growth rates to share prices
The [`prices_from_growth_rates`](@ref) function turns one path of growth rates back into share prices:

```math
S_{t+1} = S_t \exp\left((g_t + r_f)\,\Delta t\right)
```

where ``S_t`` is the share price on day ``t`` (USD/share), ``g_t`` is the excess growth rate (1/year), and ``r_f`` is the risk-free rate (1/year). Use the same `rf` and `Δt` that made the growth rates, so that the share prices come back on the same scale:

```julia
S₀ = 100.0;                                 # the first share price (USD/share)
number_of_paths = size(paths.growth_rates, 2);
P = Array{Float64,2}(undef, number_of_days + 1, number_of_paths);
for p ∈ 1:number_of_paths
    P[:, p] = JumpHMM.prices_from_growth_rates(paths.growth_rates[:, p], S₀; rf = 0.0, Δt = 1/252);
end
```

The [Portfolio](portfolio.md) page describes `simulate` for a model of many assets.
