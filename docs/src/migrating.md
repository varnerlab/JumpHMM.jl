# Migrating from 0.4

```@meta
CurrentModule = JumpHMM
```

Version 0.5.0 renames the types, functions, and settings of version 0.4.1, and every model is now made with one `build` function. The old names are gone, with no aliases. With the same inputs and seeds, version 0.5.0 gives the same numbers as version 0.4.1.

## General rules
Most changes follow a few rules:

- __Build, not fit:__ Every type is made with `build(Type, data)`, where the NamedTuple `data` holds the inputs. The `fit` function and the positional constructors are gone.
- __Growth rates, not share prices:__ The single-asset `build`, `tune`, and `validate` take excess growth rates, which [`excess_growth_rates`](@ref excess_growth_rates(::Array{Float64,1})) computes. The portfolio functions still take share prices.
- __No risk-free rate in the model:__ The single-asset model has no `rf` field. The portfolio stores `rf` and `Δt` and uses them when it computes growth rates.
- __Names:__ The setting `N` is now `number_of_states`, `n_paths` is `number_of_paths`, `n_steps` and `n` are `number_of_steps`, and `dt` is `Δt`. The word `returns` is now `growth_rates`, and `jumps` is now `episodes`.
- __Required settings:__ The keys `number_of_states` and `ν` have no default. Version 0.4.1 used `N = 100` and `ν = 5.0` when they were left out.
- __Exact types:__ Scalars are `Float64` or `Int64`, and arrays are `Array{Float64,1}` or `Array{Float64,2}`. A function argument such as `126` where the function expects a `Float64`, or a view of an array, throws an error. Write `126.0` everywhere, including the keys of `build`.

The tables below write the names without the `JumpHMM.` prefix to save space. Write the prefix in your code.

## Types

| 0.4.1 | 0.5.0 |
|:---|:---|
| `AbstractMarkovModel` | `AbstractMarkovModel` |
| `AbstractDependenceModel` | `AbstractDependenceModel` |
| `AbstractValidationResult` | `AbstractReportModel` |
| `AbstractBivariateCopula` | `AbstractBivariateCopulaModel` |
| `JumpHiddenMarkovModel` | `MyJumpHiddenMarkovModel` |
| `LaplacePartition` | The fields `laplace_μ`, `laplace_b`, and `boundaries` of `MyJumpHiddenMarkovModel` |
| `StudentTEmission` | The fields `emission_μ`, `emission_σ`, `emission_n`, `emission_fallback`, and `ν` of `MyJumpHiddenMarkovModel` |
| `JumpParameters` | The fields `ϵ`, `λ`, `p_neg`, and `N_tail` of `MyJumpHiddenMarkovModel` |
| `SimulationPath`, `SimulationResult` | The NamedTuple `(growth_rates, states, episodes)` of days × paths matrices that `simulate` returns |
| `ValidationReport` | `MyValidationReportModel` |
| `PathTestResult` | The per-path vector fields of `MyValidationReportModel`, except `kurtosis_emp`, which is now the single value `report.kurtosis_emp` |
| `GaussianCopula` | `MyGaussianCopulaModel` |
| `StudentTCopula` | `MyStudentTCopulaModel` |
| `GaussianBiCopula` | `MyGaussianBivariateCopulaModel` |
| `StudentTBiCopula` | `MyStudentTBivariateCopulaModel` |
| `ClaytonBiCopula` | `MyClaytonBivariateCopulaModel` |
| `GumbelBiCopula` | `MyGumbelBivariateCopulaModel` |
| `FrankBiCopula` | `MyFrankBivariateCopulaModel` |
| `VineEdge` | `MyVineEdgeModel` |
| `VineCopula` | `MyVineCopulaModel` |
| `SingleIndexModel` | `MySingleIndexDependenceModel` |
| `HybridSingleIndexModel` | `MyHybridSingleIndexDependenceModel` |
| `SectorFactorModel` | `MySectorFactorModel` |
| `PortfolioModel` | `MyPortfolioModel` |
| `PortfolioSimulationResult` | A `Dict{String,NamedTuple}` keyed by ticker, which the portfolio `simulate` returns |

## Fields

| 0.4.1 | 0.5.0 |
|:---|:---|
| `model.partition.N` | `model.number_of_states` |
| `model.partition.μ`, `model.partition.b` | `model.laplace_μ`, `model.laplace_b` |
| `model.partition.boundaries` | `model.boundaries` |
| `model.emissions[k].μ`, `model.emissions[k].σ` | `model.emission_μ[k]`, `model.emission_σ[k]` |
| `model.emissions[k].n_obs` | `model.emission_n[k]` |
| `model.emissions[k].is_fallback` | `model.emission_fallback[k]` |
| `model.emissions[k].ν` | `model.ν` |
| `model.jump.ϵ`, `model.jump.λ`, `model.jump.p_neg`, `model.jump.N_tail` | `model.ϵ`, `model.λ`, `model.p_neg`, `model.N_tail` |
| `model.rf` | No field. A portfolio stores `portfolio.rf` |
| `model.dt` | `model.Δt`. A portfolio also stores `portfolio.Δt` |
| `result.paths[p].observations` | `paths.growth_rates[:, p]` |
| `result.paths[p].states` | `paths.states[:, p]` |
| `result.paths[p].jumps` | `paths.episodes[:, p]` |
| `portfolio_result.results[ticker].paths[p].observations` | `paths[ticker].growth_rates[:, p]` |
| `report.path_results[i].ks_pvalue`, and the other per-path values | `report.ks_pvalue[i]`, and the other per-path vector fields, except `kurtosis_emp`, which is now the single value `report.kurtosis_emp` |
| `edge.var1`, `edge.var2` | `edge.first`, `edge.second` |
| `vine.d` | `vine.number_of_assets` |

## Functions and calls

| 0.4.1 | 0.5.0 |
|:---|:---|
| `fit(JumpHiddenMarkovModel, prices; rf, N, ν, dt, min_obs)` | `build(MyJumpHiddenMarkovModel, (growth_rates = g, number_of_states, ν, Δt, min_obs))`, with `g = excess_growth_rates(prices; rf, Δt)` |
| A new `JumpHiddenMarkovModel(...)` with `JumpParameters(ϵ, λ; p_neg, N_tail)` | `build(MyJumpHiddenMarkovModel, (growth_rates = g, number_of_states, ν, ϵ, λ, p_neg, N_tail))`, built again from the same growth rates |
| `fit(LaplacePartition, observations; N)` | `laplace_boundaries(growth_rates, number_of_states)`, which returns `(μ, b, boundaries)` |
| `assign_states(partition, observations)` | `assign_states(boundaries, growth_rates)` |
| `estimate_transition(states, N)` | `estimate_transition(states, number_of_states)` |
| `stationary_distribution(T; power)` | `stationary_distribution(T)`. The `power` keyword is gone |
| `fit_emissions(states, observations, N; ν, min_obs)` | `estimate_emissions(states, growth_rates, number_of_states; ν, min_obs)`, which returns `(μ, σ, n, fallback)` |
| `sample_emission(emission)` | `sample_emission(model, k)` for state `k` |
| `simulate(model, n_steps; n_paths, start = :stationary, seed)` | `simulate(model, number_of_steps; number_of_paths, start = nothing, seed)` |
| `tune(model, prices; n_paths, n_steps, ...)` | `tune(model, growth_rates; number_of_paths, number_of_steps, ...)` |
| `validate(model, prices; n_paths, α, acf_lags, rf, seed)` | `validate(model, growth_rates; number_of_paths, α, acf_lags, seed)` |
| `decode(model, observations)` | `decode(model, growth_rates)` |
| `forward_filter(model, observations)` | `forward_filter(model, growth_rates)` |
| `log_likelihood(model, observations)` | `log_likelihood(model, growth_rates)` |
| `JumpHMM._wasserstein1(x, y)` | `wasserstein_distance(x, y)` |
| `JumpHMM._hellinger(x, y)` | `hellinger_distance(x, y)` |
| `excess_growth_rates(prices; rf, dt)`, all three methods | `excess_growth_rates(prices; rf, Δt)` |
| `prices_from_growth_rates(G, P0; rf, dt)` | `prices_from_growth_rates(growth_rates, P₀; rf, Δt)` |
| `fit(GaussianCopula, returns)` | `build(MyGaussianCopulaModel, (growth_rates = R,))` |
| `fit(StudentTCopula, returns)` | `build(MyStudentTCopulaModel, (growth_rates = R,))` |
| `fit(VineCopula, returns)` | `build(MyVineCopulaModel, (growth_rates = R,))` |
| `GaussianBiCopula(ρ)`, `StudentTBiCopula(ρ, ν)` | `build(MyGaussianBivariateCopulaModel, (ρ = ρ,))`, `build(MyStudentTBivariateCopulaModel, (ρ = ρ, ν = ν))` |
| `ClaytonBiCopula(θ)`, `GumbelBiCopula(θ)`, `FrankBiCopula(θ)` | `build(MyClaytonBivariateCopulaModel, (θ = θ,))`, and the same for the Gumbel and Frank types |
| `VineEdge(var1, var2, conditioned, copula)` | `build(MyVineEdgeModel, (first, second, conditioned, copula))` |
| `sample_dependence(copula, n)` | `sample_dependence(copula, number_of_steps)` |
| `fit(SingleIndexModel, returns, market_returns; market_prices, rf, N, ν, dt, min_obs, residual_method)` | `build(MySingleIndexDependenceModel, (growth_rates, market_growth_rates, number_of_states, ν, market_prices, rf, Δt, min_obs, residual_method))` |
| `fit(HybridSingleIndexModel, tickers, prices, market_ticker; copula_type, rf, N, ν, dt, min_obs, r2_preserve_threshold, idiosyncratic_floor)` | `build(MyHybridSingleIndexDependenceModel, (tickers, prices, market, number_of_states, ν, copula, rf, Δt, min_obs, r2_preserve_threshold, idiosyncratic_floor))` |
| `fit(SectorFactorModel, tickers, prices, market, sector; rf, N, ν, dt, ...)`, which returned `(model, own_models)` | `build(MySectorFactorModel, (tickers, prices, market, sector, number_of_states, ν, rf, Δt, ...))`, which returns the model. The own models are in `model.own_models` |
| `JumpHMM.sample_draws(model, n; seed)` | `sample_draws(model, number_of_steps; seed)`, now exported |
| `JumpHMM.compose(model, draws)` | `compose(model, draws)`, now exported |
| `JumpHMM.compose_name(...)` | `compose_name(...)`, now exported |
| `JumpHMM.with_parameters(model; ...)` | `with_parameters(model; ...)`, now exported |
| `sample_dependence(sector_model, n; seed)` | `sample_dependence(sector_model, number_of_steps; seed)` |
| `fit(PortfolioModel, tickers, prices; dependence, market, rf, N, ν, dt, min_obs, residual_method, sectors, sector_options)` | `build(MyPortfolioModel, (tickers, prices, dependence, number_of_states, ν, market, rf, Δt, min_obs, residual_method, sector, sector_options))`. The key `dependence` is required, where 0.4.1 defaulted to `StudentTCopula` |
| `fit(PortfolioModel, …; sectors = labels)` | The key `sector = labels`, the same name as in the sector factor build |
| `tune(portfolio, prices; tickers_map, market_col, ...)` | `tune(portfolio, prices; tickers_map, market_col, ...)`, with the 0.5.0 keyword names |
| `simulate(portfolio, n_steps; n_paths, seed)` | `simulate(portfolio, number_of_steps; number_of_paths, seed)`, which returns a `Dict{String,NamedTuple}` |
| `validate(portfolio, prices; tickers_map, ...)` | `validate(portfolio, prices; tickers_map, ...)`, with the 0.5.0 keyword names |

In the two tables above, a bare key such as `number_of_states` stands for `number_of_states = ...`.

## Result keys of the sector factor model

| Function | 0.4.1 key | 0.5.0 key |
|:---|:---|:---|
| `sample_draws` | `jumps`, `sector_jumps`, `own_jumps` | `episodes`, `sector_episodes`, `own_episodes` |
| `compose` | `name_jumps` | `name_episodes` |
| `sample_dependence` | `jumps`, `name_jumps` | `episodes`, `name_episodes` |

The other keys, `g_m`, `m_states`, `F`, `E`, `Z`, `G`, and `β_path`, keep their names. The flags are now `Array{Bool,2}` matrices and `Array{Bool,1}` vectors instead of `BitMatrix` values, and the two compare equal with `==`.

## Example
A 0.4.1 script and its 0.5.0 form:

```julia
# 0.4.1 -
model = JumpHMM.fit(JumpHMM.JumpHiddenMarkovModel, prices; N = 100, ν = 5.0);
tuned_model = JumpHMM.tune(model, prices; n_paths = 100, seed = 5660);
result = JumpHMM.simulate(tuned_model, 252; n_paths = 100, seed = 5660);
g_first_path = result.paths[1].observations;

# 0.5.0 -
g = JumpHMM.excess_growth_rates(prices; rf = 0.0, Δt = 1/252);
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = g, number_of_states = 100, ν = 5.0));
tuned_model = JumpHMM.tune(model, g; number_of_paths = 100, seed = 5660);
paths = JumpHMM.simulate(tuned_model, 252; number_of_paths = 100, seed = 5660);
g_first_path = paths.growth_rates[:, 1];
```
