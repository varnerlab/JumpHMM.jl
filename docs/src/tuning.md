# Tuning

```@meta
CurrentModule = JumpHMM
```

The [`tune`](@ref tune(::MyJumpHiddenMarkovModel, ::Array{Float64,1})) function chooses the jump settings `ϵ` and `λ` from a grid. For each pair on the grid, we simulate paths and compare them with the growth rates of the data on two features. The autocorrelation of the absolute growth rate measures volatility clustering, and the excess kurtosis measures the heavy tails.

## The score
The score of one simulated path is

```math
J = \sum_{\ell=1}^{L} \left(a_{\ell} - \hat{a}_{\ell}\right)^{2} + w_{\kappa} \left(\kappa - \hat{\kappa}\right)^{2}
```

where ``a_{\ell}`` and ``\hat{a}_{\ell}`` are the autocorrelations of the absolute growth rate at a lag of ``\ell`` days in the data and in the path, and ``L`` is the number of lags (`acf_lags`). The excess kurtosis of the data is ``\kappa``, the excess kurtosis of the path is ``\hat{\kappa}``, and the weight ``w_{\kappa}`` (`w_κ`) sets how much the kurtosis error counts.

We score only the paths with at least one day in a jump episode, and we average the scores of those paths. The tuned model gets the pair with the lowest mean score. If no pair gives a path with an episode, `tune` warns and returns the first pair of the grid.

## Usage
The comments give the default of each keyword:

```julia
tuned_model = JumpHMM.tune(model, g;
    ϵ_range = range(1e-4, 2.5e-2, length = 20),  # the candidate values of ϵ (default shown)
    λ_range = range(10.0, 160.0, length = 16),   # the candidate values of the mean of the Poisson episode length, in trading days (default shown)
    number_of_paths = 200,   # the number of simulated paths for each pair (default 200)
    number_of_steps = 0,     # the number of days in each path, or 0 for the number of growth rates (default 0)
    w_κ = 0.20,              # the weight of the kurtosis error in the score (default 0.20)
    p_neg = 0.52,            # the probability that an episode day is in the negative tail (default 0.52)
    N_tail = 5,              # the number of states in each tail that an episode day can reach (default 5)
    acf_lags = 25,           # the number of lags L of the autocorrelation (default 25)
    seed = 5660              # the seed of the random number generator (default nothing)
);
```

The tuned model is a copy of the input model with the best `ϵ` and `λ`. The input model is not changed.

With a `seed`, the same grid gives the same result. When the data or the paths have too few days, we lower `acf_lags` to one less than the number of days in the shorter of the two.

The tuned model takes `p_neg` and `N_tail` from the keywords of `tune`, which default to `0.52` and `5`. When the model was built with other values of `p_neg` and `N_tail`, pass the same values to `tune`.

## Grid size
The default grid has ``20 \times 16 = 320`` pairs. With the default `200` paths for each pair, `tune` simulates 64,000 paths, each as long as the data. Start with a small grid, as in the [Getting Started](getting_started.md) example.
