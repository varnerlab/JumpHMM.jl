# Fitting

## Overview
The `fit` function constructs a `JumpHiddenMarkovModel` from a price time series. It performs the full pipeline:

1. Compute excess growth rates from prices
2. Fit a Laplace distribution via MLE → build `LaplacePartition` with `N` states
3. Assign observations to states
4. Estimate the transition matrix by frequency counting
5. Fit per-state Student-t emissions (with fallback for sparse states)
6. Compute the stationary distribution via `T^50`
7. Initialize jump parameters with `ϵ=0` (no jumps until tuned)

## Usage
```julia
model = fit(JumpHiddenMarkovModel, prices; rf=0.05, N=100, ν=5.0)
```

## API

## State Partition

## Transition Matrix

## Emissions

## Types
