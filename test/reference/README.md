# Reference values from JumpHMM 0.4.1

These files freeze what JumpHMM 0.4.1 returns on a fixed set of prices. Tests for later versions load the two JLD2 files and compare the new results with `==`.

- `inputs.jld2` holds `prices` (days x 13, SPY first), `tickers`, and `sectors` (one label for each stock).
- `reference-0.4.1.jld2` holds the 0.4.1 results. Every value is a plain array, number, `Bool`, or `String`. Keys use `/` to group related values, for example `simulate/jump/growth_rates`.
- `Project.toml` and `Manifest.toml` pin the environment to the 0.4.1 commit `e5821a8`.

## Regenerate

Run these commands from the package root:

```bash
julia --project=test/reference -e 'using Pkg; Pkg.instantiate()'
julia --project=test/reference test/reference/make_inputs.jl
julia --project=test/reference test/reference/make_reference.jl
```

`make_inputs.jl` reads the paper repository's universe and sector files. The default paths are at the top of the script. It picks the four sectors with the most tickers, and the first three tickers of each sector in alphabetical order.

## Check that the reference is reproducible

Write a second copy and compare every key with `==`:

```bash
julia --project=test/reference test/reference/make_reference.jl test/reference/inputs.jld2 second.jld2
julia --project=test/reference test/reference/compare_references.jl test/reference/reference-0.4.1.jld2 second.jld2
```
