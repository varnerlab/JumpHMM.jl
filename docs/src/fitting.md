# Fitting

## Overview
The `fit` function constructs a `JumpHiddenMarkovModel` from a price time series. It performs the full pipeline:

1. Compute excess growth rates from prices
2. Fit a Laplace distribution via MLE → build `LaplacePartition` with `N` states
3. Assign observations to states
4. Estimate the transition matrix by frequency counting
5. Fit per-state Student-t emissions (with fallback for sparse states)
6. Compute the stationary distribution by solving the linear system `πT = π`
7. Initialize jump parameters with `ϵ=0` (no jumps until tuned)

**Note**: The input price series must have nonzero variance. A perfectly flat price series (all identical values) will raise an `ArgumentError`.

## Usage
```julia
model = fit(JumpHiddenMarkovModel, prices; rf=0.05, N=100, ν=5.0)
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `rf` | `0.0` | Annual risk-free rate |
| `N` | `100` | Number of discrete states |
| `ν` | `5.0` | Student-t degrees of freedom for emissions |
| `dt` | `1/252` | Time step (daily by default) |
| `min_obs` | `2` | Minimum observations per state before fallback to global stats |

## API

## State Partition

## Transition Matrix

## Emissions

## Types
