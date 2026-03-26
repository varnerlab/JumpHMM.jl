# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JumpHMM.jl is a Julia package implementing a Hybrid Hidden Markov Model with Poisson jump-diffusion for generating synthetic equity time series. It reproduces heavy-tailed distributions, negligible linear autocorrelation, and persistent volatility clustering. Based on the paper by Alswaidan & Varner (Cornell University).

## Build & Test Commands

```bash
# Run all tests
julia --project test/runtests.jl

# Run tests via Pkg REPL
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project -e 'using JumpHMM; include("test/test_simulate.jl")'

# Build documentation locally
julia --project=docs docs/make.jl
```

Julia compat: 1.9+

## Architecture

The package follows a pipeline: **Partition → Transition → Emission → Simulate → Validate**, with multi-asset support via copulas or factor models.

### Module inclusion order (order matters — types first, then utilities, then orchestrators):

1. **Types.jl** — All abstract and concrete type definitions (immutable structs throughout)
2. **Finance.jl** — `excess_growth_rates`, `prices_from_growth_rates` helpers
3. **Partition.jl** — Laplace MLE → equal-probability quantile bins for state discretization
4. **Transition.jl** — Frequency-counting transition matrix estimation (no EM/Baum-Welch)
5. **Emission.jl** — Per-state location-scale Student-t fitting with fallback to global stats
6. **Simulate.jl** — Forward simulation with Poisson-timed tail-state jump excursions
7. **Decode.jl** — Viterbi (log-space) and forward filtering algorithms
8. **Tune.jl** — Grid search over (ϵ, λ) minimizing ACF + kurtosis error
9. **Validate.jl** — Monte Carlo validation via KS, Anderson-Darling, Wasserstein, Hellinger, ACF-MAE
10. **Copula.jl** — Gaussian and Student-t copulas (PIT → correlation)
11. **Vine.jl** — C-vine with 5 bivariate families (Gaussian, Student-t, Clayton, Gumbel, Frank) and AIC selection
12. **SIM.jl** — Single-Index Model for large asset universes
13. **Portfolio.jl** — Multi-asset orchestration: fits marginal HMMs + dependence structure

### Key design decisions

- **All structs are immutable** — `fit`, `tune`, and similar return new instances; nothing is mutated.
- **No EM algorithm** — Transition estimation uses direct frequency counting; fitting is fast but tuning (grid search) is expensive.
- **Jump mechanism** — `JumpParameters(ϵ, λ, p_neg, N_tail)` forces the chain into tail states for `Poisson(λ)` steps with probability `ϵ` per timestep. This produces volatility clustering.
- **Dependence models are interchangeable** — `GaussianCopula`, `StudentTCopula`, `VineCopula`, and `SingleIndexModel` all work as the `dependence` argument to `PortfolioModel`.
- **All stochastic functions accept a `seed` parameter** for reproducibility.

### Core workflow

```julia
model = fit(JumpHiddenMarkovModel, prices; rf, N, ν)    # fit marginals
model = tune(model, prices; ϵ_range, λ_range, n_paths)  # grid search jump params
result = simulate(model, n_steps; n_paths, seed)         # Monte Carlo paths
report = validate(model, prices; n_paths, α)             # statistical validation
```

### Type hierarchy

- `AbstractMarkovModel` → `JumpHiddenMarkovModel`
- `AbstractDependenceModel` → `GaussianCopula`, `StudentTCopula`, `VineCopula`, `SingleIndexModel`
- `AbstractValidationResult` → `PathTestResult`, `ValidationReport`
- `AbstractBivariateCopula` → `GaussianBiCopula`, `StudentTBiCopula`, `ClaytonBiCopula`, `GumbelBiCopula`, `FrankBiCopula`

## Test Structure

Tests mirror source modules 1:1 (e.g., `test/test_simulate.jl` tests `src/Simulate.jl`). Each test file defines local helper functions to build test models. Tests use `@testset` blocks and the `Test` stdlib.

## Dependencies

Core: `Distributions`, `Statistics`, `StatsBase`, `LinearAlgebra`, `HypothesisTests`, `DataFrames`, `SpecialFunctions`, `PDMats`, `Random`.
