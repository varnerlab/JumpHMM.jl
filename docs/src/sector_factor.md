# Sector Factor Model

## Overview
The sector factor model builds a synthetic universe from three layers of Jump-HMMs: one for the market,
one for each sector, and one for each name. The daily growth rate of name ``i`` in sector ``s`` is

```math
g_i(t) = \alpha_i + \beta_i(t)\, g_m(t) + \delta_i\, f_s(t) + s_i\, \tilde{\varepsilon}_i(t)
```

where ``g_m(t)`` is one draw of the market model, ``f_s(t)`` one draw of the model of the name's sector,
and ``\tilde{\varepsilon}_i(t)`` one draw of the name's own model, centered on that model's long-run mean.
It is the composition used in the INFORMS 2026 study of interpretable allocation and model-predictive
rebalancing (Varner, arXiv:2611.03022), built on the Jump-HMM of Alswaidan and Varner (arXiv:2603.10202).

## What each piece is fit to
- ``\alpha_i``, ``\beta_i``: an ordinary least squares single-index fit of the name's growth on the market's.
- ``f_s``: the equal-weight mean of the single-index residuals of the names in sector ``s``, compounded into
  a price-like series and fit with its own Jump-HMM.
- ``\delta_i``: the leave-one-out slope of the name's residual on the mean of the other residuals in its
  sector, scaled once per sector so the generator reproduces the sector's mean pairwise residual covariance.
- ``s_i``: chosen so each name keeps its real variance; if the market and sector terms would explain more
  than `1 - floor` of it, both loadings shrink and ``s_i^2`` = `floor`. Names whose real R² is at least
  `r2_threshold` keep their real R² instead.

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
portfolio = tune(portfolio, prices)                 # tunes the market, sector, and name models
result = simulate(portfolio, 252; n_paths=100, seed=1)
result.results["SPY"]                              # the synthetic market paths are returned too

# options are forwarded with `sector_options`
drifting = fit(PortfolioModel, tickers, prices; dependence=SectorFactorModel, market="SPY",
               sectors=sectors, sector_options=(β_scale=fill(0.3, 4), β_halflife=126.0))

# one path of the whole universe, with the sector draws and the per-day β
d = sample_dependence(portfolio.dependence, 252; seed=1)
```

Every sector needs at least two names. The same `seed` reproduces a path exactly.
