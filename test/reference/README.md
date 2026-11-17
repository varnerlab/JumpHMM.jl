# Reference values from JumpHMM 0.4.1

These files freeze what JumpHMM 0.4.1 returns on a fixed set of prices. Tests for later versions load the two JLD2 files and compare the new results with `==`.

- `inputs.jld2` holds `prices` (days x 13, SPY first), `tickers`, and `sectors` (one label for each stock).
- `reference-0.4.1.jld2` holds the 0.4.1 results. Every value is a plain array, number, `Bool`, or `String`. Keys use `/` to group related values, for example `simulate/jump/growth_rates`.
- `reference-0.4.1-checkbounds.jld2` holds the same results from a run with bounds checking forced on (`--check-bounds=yes`, the mode `Pkg.test()` uses). Bounds checking changes the last bit of some sums, so the tests load this file in that mode and `reference-0.4.1.jld2` otherwise.
- The `bicop/...` keys hold each bivariate copula family fit to a positive (`pos`) and a negative (`neg`) dependence pair, with its log-density, h-function, and inverse h-function on a 5 x 5 grid, and the family that AIC selects (`bicop/select/<case>`).
- The `hybrid_branches/...` keys hold a second hybrid single-index fit with `r2_preserve_threshold = 0.45` and `idiosyncratic_floor = 0.7`, so that the R²-preserving branch (7 tickers) and the beta clip (3 tickers) both run.
- `Project.toml` and `Manifest.toml` pin the environment to the 0.4.1 commit `e5821a8`.

## Regenerate

Run these commands from the package root:

```bash
julia --project=test/reference -e 'using Pkg; Pkg.instantiate()'
julia --project=test/reference test/reference/make_inputs.jl
julia --project=test/reference test/reference/make_reference.jl
julia --check-bounds=yes --project=test/reference test/reference/make_reference.jl test/reference/inputs.jld2 test/reference/reference-0.4.1-checkbounds.jld2
```

`make_inputs.jl` reads the paper repository's universe and sector files. The default paths are at the top of the script. It picks the four sectors with the most tickers, and the first three tickers of each sector in alphabetical order.

## Check that the reference is reproducible

Write a second copy and compare every key with `==`:

```bash
julia --project=test/reference test/reference/make_reference.jl test/reference/inputs.jld2 second.jld2
julia --project=test/reference test/reference/compare_references.jl test/reference/reference-0.4.1.jld2 second.jld2
```
