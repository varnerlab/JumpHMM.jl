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

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ϵ_range` | `range(1e-4, 2.5e-2, length=20)` | Grid of jump probabilities to search |
| `λ_range` | `range(10.0, 160.0, length=16)` | Grid of mean jump durations to search |
| `n_paths` | `200` | Monte Carlo paths per candidate |
| `n_steps` | `0` | Simulation length per path (`0` = match empirical length) |
| `w_κ` | `0.20` | Weight on kurtosis error in the objective |
| `p_neg` | `0.52` | Probability of downward jumps |
| `N_tail` | `5` | Number of tail states used during jumps |
| `acf_lags` | `25` | Number of ACF lags (auto-clamped to sample length if needed) |
| `seed` | `nothing` | Random seed for reproducibility |

**Note**: If no candidate `(ϵ, λ)` produces any simulated paths with jumps (e.g., very small `ϵ_range` with few `n_paths`), `tune` emits a warning and returns the first grid point. Increase `n_paths` or `ϵ_range` if this occurs.

## API
