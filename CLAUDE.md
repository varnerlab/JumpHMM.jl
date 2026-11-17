# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

JumpHMM.jl builds a hidden Markov model with jump episodes from the daily growth rates of a stock, and it simulates synthetic growth rates from that model. It also builds models of many assets, with a dependence model that links the assets. The method is in the paper by Alswaidan and Varner (Cornell University). The package is written in the course style, so that the maintainer can read every line.

## Commands

```bash
# run the full test suite (bounds checking forced on) -
julia --project -e 'using Pkg; Pkg.test()'

# run the full test suite with the default bounds-check mode -
julia --project -e 'using Pkg; Pkg.test(julia_args=["--check-bounds=auto"])'

# build the documentation locally -
julia --project=docs docs/make.jl
```

Julia compat: 1.9+. The documentation build must finish without warnings.

## Layout

The module file `src/JumpHMM.jl` includes `src/Include.jl`, and it holds every export. The file `src/Include.jl` holds every `using` line and includes the topic files in this order:

- `Types.jl` holds every type: the abstract types and the mutable `My…Model` types.
- `Finance.jl` holds the growth-rate helpers.
- `Estimate.jl` holds the partition, transition matrix, and emission estimates.
- `Factory.jl` holds every `build` method, which constructs a model from a NamedTuple.
- `Simulate.jl`, `Tune.jl`, `Validate.jl`, and `Decode.jl` hold `simulate`, `tune`, `validate`, and the decoding functions.
- `Copula.jl`, `Vine.jl`, `SingleIndex.jl`, and `SectorFactor.jl` hold the dependence models.
- `Portfolio.jl` holds the multi-asset model.

## Course style rules for `src/`

- Every type is a `My…Model` mutable struct with a `build` method in `Factory.jl`. Results, such as reports and sample matrices, are filled inside the function that produces them.
- Write documented top-level functions with explicit `for` loops and `if` blocks.
- Do not write inner or nested functions, closures, `do` blocks, ternary chains, `x || throw` or `&& return` one-liners, or `kwargs...` splats.
- Check inputs with plain `if` blocks that throw an `ArgumentError` naming the input. Read optional keys with `if haskey(data, :key) == true` blocks.
- Write a comment above each step in the course voice (`# initialize -`, `# main loop -`, `# return -`).
- Write docstrings in the course order: signature, description, `### Arguments`, `### Returns`, `### Optional Arguments`, and a short example.
- Use `Array{Float64,1}` and `Array{Float64,2}` in signatures.
- Write qualified calls in every example: `JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (...))`.

## Same numbers as 0.4.1

Every result of the package is identical (`==`) to the result of version 0.4.1, except for three deliberate changes:

- The function `hellinger_distance` counts every value of the two samples, where the 0.4.1 histogram left out the largest value, so the `hellinger` and `mean_hellinger` fields of a validation report differ.
- The function `tune` keeps the `p_neg` and `N_tail` of the input model when the call does not pass them, where version 0.4.1 reset them to `0.52` and `5`.
- A Frank copula with `θ = 0.0` uses the formulas of the independence copula, where in version 0.4.1 the h-function returned `NaN`, its inverse a meaningless `3.8e-9`, and the log-density `-Inf`, so the vine fit never selected such an edge.

A change keeps the arithmetic, the operation order, and the order of the random draws.

The files `test/reference/reference-0.4.1.jld2` and `test/reference/reference-0.4.1-checkbounds.jld2` hold the 0.4.1 results. Bounds checking changes the last bit of some sums, so the test loader picks the checkbounds file when the tests run with bounds checking forced on (`Pkg.test()`), and the other file when they run with `--check-bounds=auto`. Run the tests in both modes before a commit. The file `test/reference/README.md` explains how to regenerate the reference files.

The reference keys that end in `hellinger_corrected` or `mean_hellinger_corrected` hold the Hellinger distances of the 0.4.1 validation runs with the corrected histogram. The equivalence tests compare the Hellinger fields with these keys, and every other field with the 0.4.1 keys.

The equivalence tests run only when the Julia version and the loaded Distributions, StatsBase, and SpecialFunctions versions equal the versions in `test/reference/Manifest.toml`. With any other version, `test/runtests.jl` prints one line that names each version that differs, skips the equivalence tests, and runs the unit tests.

## Tests

The file `test/runtests.jl` loads the frozen inputs and references, and it runs every `test/equivalence/test_*.jl` file and then every `test/test_*.jl` file. The equivalence tests compare results with the 0.4.1 references. The unit tests mirror the source files.

## Documentation

The pages are in `docs/src` and follow the writing rules above: plain sentences, qualified calls, and no course lectures or weeks named. The docs build with Documenter.jl, and the CI server deploys them.

## Dependencies

The package depends on `Distributions`, `Statistics`, `StatsBase`, `LinearAlgebra`, `HypothesisTests`, `DataFrames`, `SpecialFunctions`, `PDMats`, and `Random`.
