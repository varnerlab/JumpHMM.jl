# Code Review

## Findings

1. High: bootstrap sampling breaks cross-asset residual structure

   In `src/SIM.jl:70-75`, the bootstrap path draws a separate `idx` vector for each asset. That turns the stored residual matrix into independent per-asset marginal resamples instead of a row-wise bootstrap of residual vectors. Any contemporaneous residual dependence that exists in the training data is destroyed during simulation.

   Repro: constructing a `SingleIndexModel` with `β = 0` and residual rows `[1 1; -1 -1]` produces simulated asset correlation near `0` instead of near `1`.

   Suggested fix: draw one bootstrap index per simulated time step and reuse that row across all assets.

2. Medium: `residual_method` is not exposed through the main `PortfolioModel` API

   `src/SIM.jl:14` adds `residual_method`, but `src/Portfolio.jl:6-11` does not accept it and `src/Portfolio.jl:44-46` does not forward it. As a result, the new option cannot be used from the documented portfolio-level entrypoint for SIM portfolios.

   Repro: `fit(PortfolioModel, ["MKT", "A"], prices; dependence=SingleIndexModel, market="MKT", residual_method=:gaussian)` fails with an unsupported keyword error.

   Suggested fix: accept `residual_method` in `fit(::Type{PortfolioModel}, ...)` and pass it through when `dependence <: SingleIndexModel`.

3. Medium: unsupported residual methods silently fall back to Gaussian sampling

   In `src/SIM.jl:70-77`, anything other than `:bootstrap` is treated as Gaussian. That means typos or future unsupported values are accepted silently instead of failing fast.

   Repro: `fit(SingleIndexModel, ...; residual_method=:bogus)` succeeds, and with the same RNG seed `sample_dependence` produces identical output to `residual_method=:gaussian`.

   Suggested fix: validate `residual_method` during `fit` and throw an `ArgumentError` unless it is one of the supported modes.

## Verification

- `Pkg.test()` passes on the current branch.
