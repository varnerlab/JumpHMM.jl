# The Model

```@meta
CurrentModule = JumpHMM
```

The [`MyJumpHiddenMarkovModel`](@ref) type holds a hidden Markov model of the daily growth rate of one asset. This page describes the model, the steps that [`build`](@ref build(::Type{MyJumpHiddenMarkovModel}, ::NamedTuple)) uses to estimate the model, and the fields of the model.

## The hidden Markov model
On each day ``t``, the asset is in one of ``N`` states, ``s_t \in \{1, \dots, N\}``, where ``N`` is `number_of_states`. The states are ordered by growth rate. State ``1`` holds the most negative growth rates, and state ``N`` holds the most positive ones.

The state on the next day depends only on the state today. The entry ``T_{ij}`` of the transition matrix ``T`` is the probability of moving from state ``i`` today to state ``j`` tomorrow.

In state ``k``, the model emits the growth rate

```math
g_t = \mu_k + \sigma_k\, \xi_t
```

where ``\mu_k`` is the location of state ``k`` (1/year), ``\sigma_k`` is its scale (1/year), and ``\xi_t`` is a draw from a Student-t distribution with ``\nu`` degrees of freedom. A smaller ``\nu`` gives heavier tails. The jump settings `ϵ`, `λ`, `p_neg`, and `N_tail` are stored in the same model, and the [Jump Episodes](jump_episodes.md) page explains them.

## How build estimates the model
The [`build`](@ref build(::Type{MyJumpHiddenMarkovModel}, ::NamedTuple)) function estimates the model from the growth rates in five steps:

1. __States:__ We fit a Laplace distribution to the growth rates by maximum likelihood ([`laplace_boundaries`](@ref)). Then we cut the fitted distribution at its quantiles into ``N`` intervals of equal probability. The ``N + 1`` interval edges are the `boundaries`, from `-Inf` to `Inf`.
2. __State of each day:__ Each growth rate gets the state whose interval holds it ([`assign_states`](@ref)).
3. __Transition matrix:__ We count how often state ``i`` is followed by state ``j``, and divide each row by its total ([`estimate_transition`](@ref)). A state with no next day in the data gets a row of equal probabilities.
4. __Emissions:__ The location ``\mu_k`` and the scale ``\sigma_k`` are the mean and the standard deviation of the growth rates in state ``k`` ([`estimate_emissions`](@ref)). A state with fewer than `min_obs` growth rates uses the mean and the standard deviation of all the growth rates. A state whose growth rates have a standard deviation below `1e-12` keeps its mean and uses the standard deviation of all the growth rates.
5. __Stationary distribution:__ We solve ``\pi T = \pi``, with entries of ``\pi`` that add to one, for the long-run fraction of days in each state ([`stationary_distribution`](@ref)).

The build has no random steps. The same growth rates always give the same model.

## The build call
The required keys are `growth_rates`, `number_of_states`, and `ν`. Every other key is optional, and the comments give its default:

```julia
import JumpHMM

g = JumpHMM.excess_growth_rates(prices; rf = 0.0, Δt = 1/252);  # the growth rates (1/year)
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (
    growth_rates = g,         # the growth rates (1/year), at least two values, all finite
    number_of_states = 100,   # the number of states, an Int64 of at least 1
    ν = 5.0,                  # the degrees of freedom of the Student-t emissions
    ϵ = 0.0,                  # the probability that a jump episode starts on a day (default 0.0)
    λ = 1.0,                  # the mean of the Poisson episode length, in trading days (default 1.0)
    p_neg = 0.52,             # the probability that an episode day is in the negative tail (default 0.52)
    N_tail = 5,               # the number of states in each tail that an episode day can reach (default 5)
    Δt = 1/252,               # the time step of the growth rates, in years (default 1/252)
    min_obs = 2               # the fewest growth rates a state needs to use its own emission (default 2)
));
```

The build checks `growth_rates`, `number_of_states`, `ϵ`, `λ`, `p_neg`, and `N_tail`, and throws an `ArgumentError` that names the key, for example for `ϵ = 1.5` or `N_tail = 0`. Each step above is an exported function that you can also call on its own.

The model holds no risk-free rate. We pass excess growth rates to `build`, `tune`, and `validate`. The risk-free rate is subtracted once, when [`excess_growth_rates`](@ref excess_growth_rates(::Array{Float64,1})) computes them.

## Fields
The fields of a [`MyJumpHiddenMarkovModel`](@ref) are:

| Field | What it holds |
|:---|:---|
| `number_of_states` | The number of states ``N`` |
| `laplace_μ`, `laplace_b` | The location and the scale of the fitted Laplace distribution (1/year) |
| `boundaries` | The ``N + 1`` edges of the state intervals (1/year) |
| `transition` | The transition matrix ``T``. Row ``i`` holds the probabilities of moving from state ``i`` to each state |
| `stationary` | The stationary distribution ``\pi`` |
| `emission_μ`, `emission_σ` | The location ``\mu_k`` and the scale ``\sigma_k`` of each state (1/year) |
| `emission_n` | The number of growth rates in each state |
| `emission_fallback` | A flag that is `true` for a state that uses the standard deviation of all the growth rates |
| `ν` | The degrees of freedom of the Student-t emissions |
| `ϵ`, `λ`, `p_neg`, `N_tail` | The jump settings, explained on the [Jump Episodes](jump_episodes.md) page |
| `Δt` | The time step of the growth rates, in years |
