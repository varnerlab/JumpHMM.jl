# Jump Episodes

```@meta
CurrentModule = JumpHMM
```

A jump episode is a run of days on which the asset is in the tail states of the model, where the growth rates are large and negative or large and positive. Jump episodes give the simulated paths volatility clustering. This page explains the four jump settings and how [`simulate`](@ref simulate(::MyJumpHiddenMarkovModel, ::Int64)) uses them.

## The jump settings
The jump settings are fields of the [`MyJumpHiddenMarkovModel`](@ref) and keys of its `build`:

| Setting | What it measures | Default |
|:---|:---|:---|
| `ϵ` | The probability that a jump episode starts on a given day | `0.0`, no jump episodes |
| `λ` | The mean of the Poisson episode length, in trading days | `1.0` |
| `p_neg` | The probability that a day of an episode is in the negative tail | `0.52` |
| `N_tail` | The number of states in each tail that a day of an episode can reach | `5` |

The negative tail is the lowest `N_tail` states, and the positive tail is the highest `N_tail` states.

The length ``K`` of an episode, in days, is drawn from a Poisson distribution with mean ``\lambda``:

```math
P(K = k) = \frac{\lambda^{k} e^{-\lambda}}{k!}, \qquad k = 0, 1, 2, \dots
```

## How simulate uses the jump settings
We'll follow one path. The [`simulate`](@ref simulate(::MyJumpHiddenMarkovModel, ::Int64)) function takes these steps:

1. __Day 1:__ The path starts in a state drawn from the stationary distribution, or in the state `start` if one is given, and the state emits a growth rate. Day 1 is never part of an episode.
2. __Episode check:__ On each later day with no episode under way, a jump episode starts with probability ``\epsilon``.
3. __Ordinary day:__ If no episode starts, the next state is drawn from the row of the transition matrix for the current state, and the new state emits a growth rate.
4. __Episode days:__ If an episode starts, we draw its length ``K``. On each of the next ``K`` days, starting today, the state is drawn from the negative tail with probability ``p_{\text{neg}}`` (`p_neg`) and from the positive tail otherwise, with each state of the tail equally likely. Each day of the episode picks its tail on its own, and the state of that day emits a growth rate.
5. __After the episode:__ The next day is checked for a new episode. On an ordinary day, the path moves on from the last state of the episode with the transition matrix.

An episode length of zero makes the day an ordinary day. An episode that would run past the last day stops at the last day. The `episodes` matrix of the result is `true` on the days of an episode.

## Changing the jump settings
To change the jump settings, we build again from the same growth rates, with the new settings in the NamedTuple:

```julia
study_model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (
    growth_rates = g,
    number_of_states = 100,
    ν = 5.0,
    ϵ = 1e-3,       # the probability that a jump episode starts on a given day
    λ = 90.0,       # the mean of the Poisson episode length (trading days)
    p_neg = 0.52,   # the probability that an episode day is in the negative tail
    N_tail = 5      # the number of states in each tail that an episode day can reach
));
```

The build has no random steps. The new model has the same states, transition matrix, and emissions as a model built from the same growth rates without jump episodes. To choose `ϵ` and `λ` from the data, use [`tune`](@ref tune(::MyJumpHiddenMarkovModel, ::Array{Float64,1})), which the [Tuning](tuning.md) page describes.

The decoding functions [`decode`](@ref), [`forward_filter`](@ref), and [`log_likelihood`](@ref) do not use the jump settings.
