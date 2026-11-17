# Decoding

```@meta
CurrentModule = JumpHMM
```

Three functions use a model to analyze a series of growth rates. Each function starts from the stationary distribution on day 1, moves with the transition matrix, and uses the Student-t emission density of each state. The jump settings `ϵ`, `λ`, `p_neg`, and `N_tail` are not used.

The emission density of state ``k`` at the growth rate ``g`` is

```math
f_k(g) = \frac{1}{\sigma_k}\, f_{\nu}\!\left(\frac{g - \mu_k}{\sigma_k}\right)
```

where ``\mu_k`` and ``\sigma_k`` are the location and the scale of state ``k``, and ``f_{\nu}`` is the density of a Student-t distribution with ``\nu`` degrees of freedom. Each function works with logarithms, so that the product of many small probabilities does not round to zero. Each function also throws an `ArgumentError` for an empty array of growth rates.

## Most likely states
The [`decode`](@ref) function finds the most likely sequence of states for the growth rates, with the Viterbi algorithm. It returns an `Array{Int64,1}` with the state of each day, from `1` to `number_of_states`.

```julia
states = JumpHMM.decode(model, g);   # the most likely state on each day
```

## State probabilities
The [`forward_filter`](@ref) function computes the probability of each state on each day, given the growth rates up to that day, ``P(s_t = k \mid g_1, \dots, g_t)``, where ``s_t`` is the state on day ``t``. It returns a matrix with one row for each day and one column for each state, and each row adds to one.

```julia
probabilities = JumpHMM.forward_filter(model, g);   # days × states
```

If every state gives a day a likelihood of zero, that day gets the same probability for every state.

## Log-likelihood
The [`log_likelihood`](@ref) function computes the log of the probability of the whole series of growth rates under the model, with the forward algorithm. It returns `-Inf` if every state gives some day a likelihood of zero.

```julia
ll = JumpHMM.log_likelihood(model, g);
```
