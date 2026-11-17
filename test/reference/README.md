# Reference values from JumpHMM 0.4.1

These files freeze what JumpHMM 0.4.1 returns on a fixed set of prices. Tests for later versions load the two JLD2 files and compare the new results with `==`.

- `inputs.jld2` holds `prices` (days x 13, SPY first), `tickers`, and `sectors` (one label for each stock).
- `reference-0.4.1.jld2` holds the 0.4.1 results. Every value is a plain array, number, `Bool`, or `String`. Keys use `/` to group related values, for example `simulate/jump/growth_rates`.
- `reference-0.4.1-checkbounds.jld2` holds the same results from a run with bounds checking forced on (`--check-bounds=yes`, the mode `Pkg.test()` uses). Bounds checking changes the last bit of some sums, so the tests load this file in that mode and `reference-0.4.1.jld2` otherwise.
- The `bicop/...` keys hold each bivariate copula family fit to a positive (`pos`) and a negative (`neg`) dependence pair, with its log-density, h-function, and inverse h-function on a 5 x 5 grid, and the family that AIC selects (`bicop/select/<case>`).
- The `hybrid_branches/...` keys hold a second hybrid single-index fit with `r2_preserve_threshold = 0.45` and `idiosyncratic_floor = 0.7`, so that the R²-preserving branch (7 tickers) and the beta clip (3 tickers) both run.
- The `portfolio/simulate/SPY/...` keys hold the market paths (`growth_rates`, `states`, `episodes`) of the same sector-factor simulation as `portfolio/simulate/<ticker>/...`. The first version of the files left them out.
- The `portfolio_once/...` keys hold a Gaussian copula portfolio (`gauss`, the first 301 days) and a hybrid portfolio (`hybrid`, the first five columns) that were tuned and validated after one `Random.seed!(7)`, with no `seed` keyword. These keys pin the order in which a portfolio visits its models. The hybrid keys hold the portfolio's models (`marginals`), the hybrid model's own models (`dependence`), and the market model.
- The `portfolio_sector_dt/...` keys hold the validation of a sector-factor portfolio built with `sector_options = (dt = 1/12, center_steps = 1000)`, for two tickers.
- The `copula_one/...` and `hybrid_one/...` keys hold the Gaussian, Student-t, and vine copulas of one column, and a hybrid model of the market and one stock with its portfolio paths.
- The keys that end in `hellinger_corrected` or `mean_hellinger_corrected` hold the Hellinger distances of the same validation runs, computed with a corrected histogram that counts the largest value. Version 0.4.1 left the largest value out of its histogram, and version 0.5.0 counts it. The script patches the corrected function into 0.4.1 at the end of the run, after every other key is written, so that the other keys keep the 0.4.1 numbers.
- `Project.toml` and `Manifest.toml` pin the environment to the 0.4.1 commit `e5821a8`.

The equivalence tests compare with `==`, so they run only when the Julia version and the loaded Distributions, StatsBase, and SpecialFunctions versions equal the versions in `Manifest.toml`. Otherwise `test/runtests.jl` prints one line that names each version that differs, skips the equivalence tests, and runs the unit tests.

## Regenerate

Run these commands from the package root:

```bash
julia --project=test/reference -e 'using Pkg; Pkg.instantiate()'
julia --project=test/reference test/reference/make_inputs.jl <universe.jld2> <sectors.csv>
julia --project=test/reference test/reference/make_reference.jl
julia --check-bounds=yes --project=test/reference test/reference/make_reference.jl test/reference/inputs.jld2 test/reference/reference-0.4.1-checkbounds.jld2
```

The script `make_inputs.jl` reads a universe file of price tables and a CSV file of GICS sectors, given as its first two arguments. It picks the four sectors with the most tickers, and the first three tickers of each sector in alphabetical order.

## Check that the reference is reproducible

Write a second copy and compare every key with `==`:

```bash
julia --project=test/reference test/reference/make_reference.jl test/reference/inputs.jld2 second.jld2
julia --project=test/reference test/reference/compare_references.jl test/reference/reference-0.4.1.jld2 second.jld2
```
