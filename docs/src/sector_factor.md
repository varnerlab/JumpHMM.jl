# Sector Factor Model

## Overview
The sector factor model builds a synthetic universe from three layers of Jump-HMMs: one for the market,
one for each sector, and one for each name's own residual. The daily growth rate of name ``i`` in sector
``s`` is

```math
g_i(t) = \alpha_i + \beta_i(t)\, g_m(t) + \delta_i\, f_s(t) + \psi_i\, \tilde{\varepsilon}_i(t)
```

where ``g_m(t)`` is one draw of the market model, ``f_s(t)`` one draw of the model of the name's sector
factor, and ``\tilde{\varepsilon}_i(t)`` one draw of the model of the name's residual after its sector term,
centered on that model's long-run mean. Each term is a model of the piece of the growth rate it stands for.
The Jump-HMM itself is described by Alswaidan and Varner (arXiv:2603.10202).

## What each piece is fit to
- ``\alpha_i``, ``\beta_i``: an ordinary least squares single-index fit of the name's growth on the market's,
  with residual ``\varepsilon_i(t)``.
- ``f_s``: the equal-weight mean of the single-index residuals of the names in sector ``s``, compounded into
  a price-like series and fit with its own Jump-HMM.
- ``\delta_i``: the leave-one-out slope of the name's residual on the mean of the other residuals in its
  sector, scaled once per sector so the generator reproduces the sector's mean pairwise residual covariance.
- ``\tilde{\varepsilon}_i``: a Jump-HMM fit to the residual left after the sector term,
  ``u_i(t) = \varepsilon_i(t) - \delta_i f_s(t)`` (`own_residuals`), compounded the same way.
- ``\psi_i``: the scale of the own draw, set so the β drift, the sector term and the own term together carry
  the variance of the name's single-index residual (`own_variances`, `own_scale`):

```math
\psi_i^2 = \frac{\operatorname{Var}(\varepsilon_i) - \delta_i^2 \operatorname{Var}(f_s) - \sigma_{x,i}^2 \operatorname{Var}(g_m)}{\operatorname{Var}(u_i)}
```

Here ``\sigma_{x,i}^2`` is the variance of the β drift (`drift_variance(β_halflife, β_scale[i])`), zero for a
fixed β. The single-index residual comes from a fit with a fixed β, so it already holds the real β's movement,
and subtracting the drift's variance keeps the synthetic drift from adding it a second time. The sector factor
averages the residuals of its names, name ``i`` included, so part of ``u_i`` can still move with ``f_s``.
Independent draws drop that shared part, and ``\psi_i`` puts it back. With two names per sector and a fixed β,
``\psi_i`` is well above 1; with many names per sector, it is close to 1, and a drifting β lowers it.
`with_parameters` recomputes ``\psi`` when the drift changes.

## Why no copula
Names are linked only through the market and sector draws they share on the same day. A copula imposes a
fitted dependence by reordering each name's draws across days, which would scramble the volatility
clustering and the timing of jump episodes that the Jump-HMM produces. Here every name keeps its own
day-by-day dynamics, and correlation across sectors comes only through the market term.

## Fixed or drifting β
By default ``\alpha_i`` and ``\beta_i`` are fixed at their single-index estimates. Two options relax this:
- `β_scale` (one value per name) makes β drift as an exponential moving average of normal shocks with
  half-life `β_halflife` trading days, started from its stationary distribution.
- `intercept_pull` shrinks each ``\alpha_i`` toward the universe mean (1 keeps the estimates, 0 sets every
  α to the mean).

## Usage
```julia
using JumpHMM

tickers = ["SPY", "JPM", "BAC", "XOM", "CVX"]      # market first or anywhere; prices has one column per ticker
sectors = ["Financials", "Financials", "Energy", "Energy"]   # one label per non-market ticker, in order

portfolio = fit(PortfolioModel, tickers, prices;
                dependence=SectorFactorModel, market="SPY", sectors=sectors, N=50)
portfolio = tune(portfolio, prices)                 # tunes the market, sector, and own models
result = simulate(portfolio, 252; n_paths=100, seed=1)
result.results["SPY"]                              # the synthetic market paths are returned too

# options are forwarded with `sector_options`
drifting = fit(PortfolioModel, tickers, prices; dependence=SectorFactorModel, market="SPY",
               sectors=sectors, sector_options=(β_scale=fill(0.3, 4), β_halflife=126.0))

# one path of the whole universe, with the sector draws and the per-day β
d = sample_dependence(portfolio.dependence, 252; seed=1)
```

Every sector needs at least two names. The same `seed` reproduces a path exactly. For a
`SectorFactorModel` portfolio, `portfolio.marginals` holds the own models, and `validate` checks each one
against its residual series.

## Calibrating the generator yourself
A caller that fits its own component models, or calibrates the drift against its own estimator, can
use three tools:
- `fit(SectorFactorModel, ...; market_model, own_models, sector_models)` stores models fitted or tuned
  elsewhere instead of fitting them; the noise centers come from the own models it stores. Own models must
  describe each name's residual after its sector term (`model.own_levels` holds those series, compounded).
- `sample_draws(model, n; seed)` returns every random component of a path, and `compose(model, draws)`
  turns them into name growth rates. `sample_dependence(model, n; seed)` is the two together.
- `with_parameters(model; α, β_halflife, β_scale)` copies the model with new intercepts and β drift.

Together they let a fit hold the randomness fixed and change only the parameters:

```julia
bank = [JumpHMM.sample_draws(model, n_days; seed=p) for p in 1:20]   # fixed draws
for k in (2.0, 4.0, 6.0)
    trial = JumpHMM.with_parameters(model; β_scale=fill(k, length(model.tickers)))
    G = [JumpHMM.compose(trial, d).G for d in bank]                # same draws, new drift
    # ... compare a statistic of G with its real value and keep the best k
end
```
