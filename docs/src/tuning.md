# Tuning

## Overview
The `tune` function performs a grid search over jump parameters `(ϵ, λ)` to find values that best balance distributional fidelity (kurtosis) and temporal fidelity (autocorrelation of absolute returns).

The objective function is:

```math
J(\epsilon, \lambda) = \sum_{\ell=1}^{L} (\text{acf}_{\text{obs}}[\ell] - \text{acf}_{\text{sim}}[\ell])^2 + w_\kappa \cdot (\kappa_{\text{obs}} - \kappa_{\text{sim}})^2
```

Only simulated paths containing at least one jump event contribute to the objective, matching the paper's methodology.

## Usage
```julia
tuned_model = tune(model, prices;
    ϵ_range=range(1e-4, 2.5e-2, length=20),
    λ_range=range(10, 160, length=16),
    n_paths=200,
    w_κ=0.20)
```

The returned model is a new `JumpHiddenMarkovModel` with optimized `JumpParameters`. The partition, transition matrix, and emissions are carried forward unchanged.

## API
