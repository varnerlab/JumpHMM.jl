# Getting Started

```@meta
CurrentModule = JumpHMM
```

This page shows the single-asset workflow: compute the growth rates, build the model, tune the jump settings, simulate, and study other jump settings. The example makes its own synthetic share price series, so that the code runs without a data file.

## Make a share price series
The code below makes 1000 daily share prices, with a calm stretch and a volatile stretch. In your own work, use the closing prices of a stock here, as an `Array{Float64,1}`.

```@example workflow
import JumpHMM
using Random

# make a synthetic share price series: 1000 trading days, more volatile from day 601 to day 700 -
Random.seed!(1234);
number_of_days = 1000;
Δt = 1/252;                 # one trading day, in years
S = Array{Float64,1}(undef, number_of_days + 1);
S[1] = 100.0;               # the first share price (USD/share)
for t ∈ 1:number_of_days
    σ = 0.15;               # the volatility of a calm day (1/sqrt(year))
    if t > 600 && t <= 700
        σ = 0.45;           # the volatility of a volatile day (1/sqrt(year))
    end
    S[t+1] = S[t] * exp((0.08 - 0.5 * σ^2) * Δt + σ * sqrt(Δt) * randn());
end
nothing # hide
```

## Compute the growth rates
The [`excess_growth_rates`](@ref excess_growth_rates(::Array{Float64,1})) function turns ``n + 1`` share prices into ``n`` growth rates, in 1/year. A risk-free rate `rf = 0.0` keeps the total growth rate.

```@example workflow
g = JumpHMM.excess_growth_rates(S; rf = 0.0, Δt = Δt);  # the growth rates (1/year)
length(g)
```

## Build the model
The [`build`](@ref build(::Type{MyJumpHiddenMarkovModel}, ::NamedTuple)) function estimates the hidden Markov model from the growth rates. The new model has no jump episodes, because the probability `ϵ` that an episode starts on a day is `0.0` by default.

```@example workflow
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (
    growth_rates = g,       # the growth rates (1/year)
    number_of_states = 20,  # the number of hidden states
    ν = 5.0                 # the degrees of freedom of the Student-t emissions
));
model.ϵ
```

Every scalar setting is a `Float64` or an `Int64`. Write `ν = 5.0`, not `ν = 5`, because an integer where the package expects a `Float64` throws an error. The page [The Model](model.md) lists the steps of the build and the fields of the model.

## Tune the jump settings
The [`tune`](@ref tune(::MyJumpHiddenMarkovModel, ::Array{Float64,1})) function searches a grid of jump settings for the pair whose simulated paths best match the growth rates. It returns a new model that holds the best pair, and the input model is not changed.

```@example workflow
tuned_model = JumpHMM.tune(model, g;
    ϵ_range = range(1e-3, 5e-3, length = 3),  # the candidate probabilities that an episode starts on a day
    λ_range = range(10.0, 50.0, length = 3),  # the candidate means of the episode length (trading days)
    number_of_paths = 20,                     # the number of simulated paths for each pair
    seed = 5660                               # the seed of the random number generator
);
(tuned_model.ϵ, tuned_model.λ)
```

This small grid keeps the example fast. The [Tuning](tuning.md) page explains the score and the default grid.

## Simulate
The [`simulate`](@ref simulate(::MyJumpHiddenMarkovModel, ::Int64)) function draws synthetic paths from the model. Here, each path has as many days as the data.

```@example workflow
paths = JumpHMM.simulate(tuned_model, length(g); number_of_paths = 100, seed = 5660);
size(paths.growth_rates)
```

The result holds three matrices with one row for each day and one column for each path:

- __Growth rates:__ The matrix `paths.growth_rates` holds the simulated growth rates, in 1/year.
- __States:__ The matrix `paths.states` holds the state of the model on each day.
- __Episodes:__ The matrix `paths.episodes` is `true` on the days of a jump episode.

To turn a path back into share prices, we use [`prices_from_growth_rates`](@ref) with the same `rf` and `Δt`:

```@example workflow
S_path = JumpHMM.prices_from_growth_rates(paths.growth_rates[:, 1], S[1]; rf = 0.0, Δt = Δt);  # USD/share
length(S_path)
```

## Study other jump settings
To study other jump settings, we build again from the same growth rates, with the new settings in the NamedTuple. The build has no random steps. The new model has the same states, transition matrix, and emissions as `model`.

```@example workflow
study_model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (
    growth_rates = g,
    number_of_states = 20,
    ν = 5.0,
    ϵ = 5e-3,       # the probability that a jump episode starts on a day
    λ = 30.0,       # the mean of the Poisson episode length (trading days)
    N_tail = 3      # the number of states in each tail that an episode day can reach
));
study_paths = JumpHMM.simulate(study_model, length(g); number_of_paths = 100, seed = 5660);
sum(study_paths.episodes) / length(study_paths.episodes)  # the fraction of days in a jump episode
```

The [Jump Episodes](jump_episodes.md) page explains each jump setting, and the [Validation](validation.md) page compares simulated paths with the data.
