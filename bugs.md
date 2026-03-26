# JumpHMM Code Review Findings

Review date: 2026-03-26

I ran the full test suite with `julia --project -e 'using Pkg; Pkg.test()'`. It passes, so the findings below are issues that are either untested or only visible under edge cases.

## Second Review Pass

Re-check date: 2026-03-26

- Current test status: `julia --project -e 'using Pkg; Pkg.test()'` passes.
- Resolved since the previous re-check: the all-`-Inf` `forward_filter` / `log_likelihood` repro now returns a uniform posterior and `-Inf` log-likelihood instead of `NaN`.
- Resolved since the previous re-check: `SingleIndexModel` portfolio `validate`/`tune` now use the stored original ticker-to-column mapping by default, and market-model tuning now works without requiring an explicit `market_col`.
- Resolved since the previous re-check: the Gaussian-copula portfolio repro that previously collapsed dependence now produced strong simulated dependence (`Pearson ≈ 0.877`, `Spearman ≈ 0.874`) on the same `ρ = 0.9` training setup.

### New Finding: flat price paths crash fitting

- File: `src/Partition.jl:6-9`
- Impact: any perfectly flat price path, or any path whose excess-growth rates are all identical, crashes `fit(JumpHiddenMarkovModel, ...)` with a raw `DomainError` from `fit_mle(Laplace, ...)`. This also cascades into `SingleIndexModel` fitting when the market-return series is constant.
- Repro: `fit(JumpHiddenMarkovModel, fill(100.0, 20); N=5)` throws `DomainError(0.0, "Laplace: the condition θ > zero(θ) is not satisfied.")`.
- Why it is a bug: a flat series is a valid input shape for the public API, but the package currently treats zero-variance data as an internal distribution-constructor failure instead of handling it deliberately or surfacing a package-level error.
- Suggested fix: detect zero-variance observations before `fit_mle(Laplace, ...)` and either regularize the scale to a small positive value or throw a clear domain-specific `ArgumentError`.

### New Finding: `validate` and `tune` crash on short histories

- File: `src/Validate.jl:14-18`
- File: `src/Validate.jl:38-40`
- File: `src/Tune.jl:25-31`
- File: `src/Tune.jl:53-56`
- Impact: small input datasets can be fit successfully but cannot be validated or tuned with default settings.
- Repro: using a 12-point price vector, both `validate(model, prices; n_paths=2)` and `tune(model, prices; n_paths=2, ϵ_range=[1e-3], λ_range=[5.0])` throw `ErrorException("lags must be less than the sample length.")`.
- Why it is a bug: both functions hard-code `acf_lags=25` by default and never clamp it to the available sample length before calling `autocor`.
- Suggested fix: replace `acf_lags` with `min(acf_lags, length(series) - 1)` in both empirical and simulated ACF calculations, or reject undersized inputs explicitly with a clear error.

### New Finding: copula models can fit singular correlation matrices but cannot sample from them

- File: `src/Copula.jl:6-9`
- File: `src/Copula.jl:17-33`
- File: `src/Copula.jl:41-58`
- Impact: fitting succeeds on perfectly collinear assets, but later sampling and portfolio simulation fail with `PosDefException`.
- Repro: with `R = [1 1; 2 2; 3 3; 4 4]`, both `fit(GaussianCopula, R)` and `fit(StudentTCopula, R)` return `Σ = [1 1; 1 1]`, but both `sample_dependence(...)` calls fail with `PosDefException(2)`.
- Why it is a bug: the fit step returns a model that the package cannot actually use for simulation.
- Suggested fix: project `Σ` onto a positive-definite correlation matrix during fitting, or reject singular fits immediately with a package-level error message.

### New Finding: `_hellinger` returns `NaN` for degenerate identical samples

- File: `src/Validate.jl:75-90`
- Impact: validation metrics can become `NaN` when both samples collapse to a single repeated value.
- Repro: `JumpHMM._hellinger(fill(1.0, 10), fill(1.0, 10))` returns `NaN`.
- Why it is a bug: when `lo == hi`, the histogram edges collapse to a zero-width range and the normalized weights become undefined.
- Suggested fix: handle the `lo == hi` case explicitly and return `0.0` when both samples are identical degenerate distributions.

### New Finding: zero-step simulation throws an unguarded `BoundsError`

- File: `src/Simulate.jl:32-42`
- Impact: `simulate(model, 0; ...)` crashes with a low-level bounds error instead of returning an empty path or rejecting the input clearly.
- Repro: calling `simulate(model, 0; n_paths=1)` throws `BoundsError(Int64[], (1,))`.
- Why it is a bug: the function allocates zero-length output vectors and then unconditionally writes `states_out[1]`, `obs_out[1]`, and `jumps_out[1]`.
- Suggested fix: require `n_steps >= 1` with a clear `ArgumentError`, or define zero-step simulation semantics and return empty paths.

## Final Review Pass

Re-check date: 2026-03-26

- Current test status: `julia --project -e 'using Pkg; Pkg.test()'` passes.

### New Finding: decoding APIs crash on empty observation sequences

- File: `src/Decode.jl:6-52`
- File: `src/Decode.jl:61-119`
- File: `src/Decode.jl:128-187`
- Impact: `decode`, `forward_filter`, and `log_likelihood` all throw low-level `BoundsError`s on `Float64[]` instead of rejecting empty inputs clearly or returning a defined empty result.
- Repro: each of `decode(model, Float64[])`, `forward_filter(model, Float64[])`, and `log_likelihood(model, Float64[])` fails with `BoundsError(Float64[], (1,))`.
- Why it is a bug: all three functions index `observations[1]` without validating that the sequence is non-empty.
- Suggested fix: require `length(observations) ≥ 1` with a clear `ArgumentError`, or define explicit empty-sequence semantics and implement them consistently across the three APIs.

### New Finding: `SingleIndexModel` silently produces `NaN` coefficients when the market factor has zero variance

- File: `src/SIM.jl:34-45`
- Impact: `fit(SingleIndexModel, ...)` can return a model containing `NaN` betas, alphas, and residual scales when `market_returns` is constant or nearly constant.
- Repro: fitting with `market_returns = fill(0.01, 50)` produced `β = [NaN, NaN]` because `var(market_returns) == 0.0`.
- Why it is a bug: the OLS step divides by `G_m_var` with no guard, so the API silently returns an invalid model instead of failing fast or handling the degenerate factor.
- Suggested fix: reject zero-variance market factors with a clear error, or switch to a guarded regression path that handles constant factors explicitly.

### New Finding: `tune` returns an arbitrary grid point when no simulated path contains a jump

- File: `src/Tune.jl:34-36`
- File: `src/Tune.jl:50-70`
- Impact: on sparse-jump grids or small Monte Carlo budgets, `tune` can return parameters determined only by the iteration order of `ϵ_range` and `λ_range`, not by the data.
- Repro: with `n_paths=1`, `n_steps=10`, and tiny jump probabilities, `tune(...; ϵ_range=[1e-10, 2e-10], λ_range=[5.0, 10.0])` returned `(1e-10, 5.0)`, while reversing the grid order returned `(2e-10, 10.0)` on the same fitted model and seed.
- Why it is a bug: if `n_valid == 0` for every candidate, `best_J` stays `Inf` and the function falls back to `first(ϵ_range)` / `first(λ_range)`.
- Suggested fix: detect the “no candidate produced any valid jump paths” case and either increase simulation effort automatically or throw a clear error instead of returning an order-dependent result.

## Historical Follow-up Verification After Attempted Fixes

This section records an earlier intermediate re-check. Current status is reflected in the `Second Review Pass` section above.

Re-check date: 2026-03-26

- Current test status: `julia --project -e 'using Pkg; Pkg.test()'` still passes.
- Resolved: the sticky-chain stationary-distribution repro from item 2 now returns `[0.4999999999999998, 0.5000000000000002]`, so the old `(T^50)[1, :]` bug appears fixed.
- Resolved: the `simulate(...; start=1)` repro from item 4 now returns `[1, 2, 2, 2, 2]`, so the first emitted state now honors `start`.

### Still Open: `forward_filter` / `log_likelihood` numerical failure on all-`-Inf` paths

- File: `src/Decode.jl:61-107`
- File: `src/Decode.jl:116-165`
- Status: still broken after the attempted log-space rewrite.
- Why it is still a bug: the new implementation computes `m = maximum(log_alpha)` and then normalizes with `exp.(log_alpha .- m)`. When every candidate log-probability is `-Inf`, `m == -Inf`, and `log_alpha .- m` becomes `NaN`.
- Repro: with the same 2-state model and `obs = fill(1e300, 3)`, `forward_filter` still returned `[NaN NaN; NaN NaN; NaN NaN]` and `log_likelihood` still returned `NaN`.
- Suggested fix: explicitly handle the `all(isinf, log_alpha)` / `all(isinf, log_alpha_new)` case before normalization and return a defined fallback instead of subtracting `-Inf`.

### Still Open: `SingleIndexModel` default portfolio helpers still use the wrong columns

- File: `src/Portfolio.jl:75-106`
- File: `src/Portfolio.jl:192-210`
- Status: still broken by default; the change added optional arguments but did not fix the default behavior.
- Why it is still a bug: the default `tickers_map` is still derived from `portfolio.tickers`, which for SIM portfolios excludes the market ticker. Passing the original price matrix `[market, asset1, asset2, ...]` still maps `"A"` to column `1` instead of its actual column.
- Repro: after fitting a SIM portfolio on `[MKT, A, B]`, `validate(portfolio, prices)` still matched `validate(portfolio.marginals["A"], prices[:, 1])`, while `validate(..., prices[:, 2])` produced different numbers.
- Additional open issue: `src/Portfolio.jl:97-103` only tunes the SIM market model when `market_col` is supplied explicitly. With defaults, `tuned_default.dependence.market_model.jump.ϵ` stayed `0.0`; with `market_col=1`, the same repro changed it to `0.002`.
- Suggested fix: store the original input column mapping inside the portfolio object, or derive it automatically during `fit`, so `tune` and `validate` work correctly without requiring the caller to reconstruct the mapping manually.

### Partially Improved But Not Confirmed: copula portfolio simulation

- File: `src/Portfolio.jl:126-156`
- Status: improved, but not yet confirmed as correct.
- What changed: the code now consumes `U = sample_dependence(dep, n_steps)` and uses copula quantiles when generating observations, so the original "completely ignored" bug is no longer present verbatim.
- Remaining concern: a focused repro with training correlation `0.9` produced simulated cross-asset correlation `0.035671217544176806`, which is far below the original dependence. This may indicate that the current construction still fails to preserve practical cross-asset dependence, or that it needs a better validation metric than raw Pearson correlation.
- Why this remains a finding: `test/test_portfolio.jl` still only checks shapes and object types and does not assert any nontrivial dependence preservation, so the fix is not actually verified.

## 1. High: copula-based portfolio simulation never applies the copula

- File: `src/Portfolio.jl:117-132`
- Why it is a bug: `simulate(portfolio, ...)` samples `U = sample_dependence(dep, n_steps)` and then never uses `U`. Each marginal is simulated independently and pushed into the result unchanged, so `GaussianCopula`, `StudentTCopula`, and `VineCopula` have no effect on the generated joint paths.
- Impact: multi-asset simulations are effectively independent across assets, which breaks the main promise of the portfolio API.
- Repro: fitting a 2-asset Gaussian-copula portfolio on data with training correlation `0.9` produced simulated cross-asset return correlation `-0.008`.
- Suggested fix: use the copula draws to reorder or transform marginal draws before storing each path, as described in `SPECIFICATION.md`.

## 2. High: `stationary_distribution` is not actually stationary for sticky chains

- File: `src/Transition.jl:23-31`
- Why it is a bug: the implementation returns `(T^50)[1, :]`, which is only an approximation after an arbitrary number of steps. For slowly mixing chains this can be far from the true stationary distribution.
- Impact: the fitted model can start simulations from the wrong distribution and initialize decoders/filters with biased state probabilities.
- Repro: for `T = [0.999 0.001; 0.001 0.999]`, the true stationary distribution is `[0.5, 0.5]`, but the current function returns `[0.9523734090020178, 0.04762659099798212]`.
- Suggested fix: solve for the left eigenvector of `T` associated with eigenvalue `1`, or solve the linear system for `πT = π` plus `sum(π) = 1`.

## 3. High: `tune` and `validate` use the wrong price columns for `SingleIndexModel` portfolios by default

- File: `src/Portfolio.jl:30-55`
- File: `src/Portfolio.jl:66-84`
- File: `src/Portfolio.jl:163-179`
- Why it is a bug: `fit(...; dependence=SingleIndexModel, market=...)` removes the market ticker and stores only non-market `asset_tickers` in `portfolio.tickers`. Later, the default `tickers_map` in both `tune` and `validate` assumes the input `prices` matrix columns match `portfolio.tickers` in order. If the caller passes the original price matrix `[market, asset1, asset2, ...]`, ticker `"A"` gets mapped to column `1` instead of its real column.
- Impact: validation and tuning can silently run against the wrong assets. In the common case where the original price matrix still includes the market column, asset reports are computed against unrelated price series.
- Repro: after fitting a SIM portfolio on `[MKT, A, B]`, `validate(portfolio, prices)` produced the same `A` report as `validate(portfolio.marginals["A"], prices[:, 1])`, while `validate(..., prices[:, 2])` gave different numbers.
- Additional issue in the same path: `src/Portfolio.jl:86-95` explicitly leaves `dep.market_model` unchanged (`market_tuned = market_model`), so SIM tuning does not tune the market model at all.
- Suggested fix: persist the original column mapping when building the portfolio, or require/derive an explicit mapping for SIM portfolios; then actually tune the market model using the market column.

## 4. Medium: `simulate(...; start=k)` does not start the returned path in state `k`

- File: `src/Simulate.jl:38-73`
- Why it is a bug: the code chooses `s = start`, but before writing `states_out[1]` it immediately performs either a normal transition or a jump assignment. That means the first recorded state is a successor of `start`, not `start` itself.
- Impact: the `start` keyword is misleading and unusable when callers need deterministic initial conditions.
- Repro: with a transition matrix whose rows always move to state `2`, calling `simulate(model, 5; n_paths=1, start=1)` returns states `[2, 2, 2, 2, 2]`.
- Suggested fix: record and emit from the chosen starting state at `t = 1`, then transition on later steps.

## 5. Medium: `forward_filter` and `log_likelihood` can return `NaN` because they operate in probability space

- File: `src/Decode.jl:71-90`
- File: `src/Decode.jl:113-134`
- Why it is a bug: both functions multiply raw densities and then normalize by `Z = sum(alpha...)`. For sufficiently extreme observations, `pdf(...)` underflows to `0.0`, `Z` becomes `0.0`, and the code divides by zero or takes `log(0.0)`.
- Impact: the decoding API is numerically unstable on extreme data or long horizons and can return `NaN` instead of a usable posterior or likelihood.
- Repro: with a simple 2-state model and `obs = fill(1e300, 3)`, `forward_filter` returned a matrix of `NaN` values and `log_likelihood` returned `NaN`.
- Suggested fix: implement both routines in log-space with log-sum-exp normalization, similar to how `decode` already uses log-space Viterbi recursion.

## Coverage gaps

- `test/test_portfolio.jl` only checks shapes and object types; it never asserts that cross-asset dependence survives simulation.
- There are no tests for SIM helper methods using the original price matrix shape after the market ticker is removed.
- There are no tests for `simulate(...; start=...)` semantics.
- There are no numerical-stability tests for `forward_filter` or `log_likelihood`.
- There are no tests that verify `stationary_distribution` is actually stationary.
