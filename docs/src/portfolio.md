# Portfolio

```@meta
CurrentModule = JumpHMM
```

The [`MyPortfolioModel`](@ref) type holds a model of many assets: a hidden Markov model of each asset and a dependence model that links the assets. One `build` call fits both from a matrix of share prices. The `tune`, `simulate`, and `validate` functions then work on the whole portfolio.

## Build
The required keys are `tickers`, `prices`, `dependence`, `number_of_states`, and `ν`:

```julia
import JumpHMM

portfolio = JumpHMM.build(JumpHMM.MyPortfolioModel, (
    tickers = tickers,                            # one ticker for each column of prices
    prices = prices,                              # the share prices (USD/share), one row per day
    dependence = JumpHMM.MyStudentTCopulaModel,   # the type of the dependence model
    number_of_states = 100,
    ν = 5.0,
    rf = 0.0,                                     # optional: the risk-free rate, in 1/year (default 0.0)
    Δt = 1/252                                    # optional: the time step, in years (default 1/252)
));
```

We turn the prices into excess growth rates with `rf` and `Δt`. The portfolio stores both, and `tune` and `validate` use them again. Pass share prices, not growth rates, to `build`, `tune`, and `validate`.

The dependence model sets which other keys the build needs, and which tickers the portfolio models:

| `dependence` | Needs `market` | Needs `sectors` | Assets in `portfolio.tickers` |
|:---|:---|:---|:---|
| `MyGaussianCopulaModel` | no | no | every ticker |
| `MyStudentTCopulaModel` | no | no | every ticker |
| `MyVineCopulaModel` | no | no | every ticker |
| `MySingleIndexDependenceModel` | yes | no | every ticker except the market |
| `MyHybridSingleIndexDependenceModel` | yes | no | every ticker except the market |
| `MySectorFactorModel` | yes | yes | every ticker except the market |

The key `market` names the ticker of the market, which must be in `tickers`. For a sector factor portfolio, `sectors` holds one sector label for each ticker except the market.

The sector factor build itself calls this key `sector`. The key `sector_options` passes more keys to the sector factor build, for example `(center_steps = 20_000, β_scale = fill(0.3, 4))`.

The single-index model also takes `residual_method`, which the [Single-Index Models](single_index.md) page explains. Every portfolio takes `min_obs`, as the single-asset build does.

## Tune
The [`tune`](@ref tune(::MyPortfolioModel, ::Array{Float64,2})) function tunes each hidden Markov model of the portfolio with the single-asset [`tune`](@ref tune(::MyJumpHiddenMarkovModel, ::Array{Float64,1})), and returns a new portfolio. The input portfolio is not changed. What we tune depends on the dependence model:

- __Copula:__ We tune the model of each asset on the growth rates of that asset.
- __Single-index and hybrid:__ We tune the model of each asset and the market model.
- __Sector factor:__ We tune the market model, each sector model, and each own model. Then we draw the long-run mean of each own model (`ε_center`) again. The own scales `ψ` stay as they are.

The keywords are those of the single-asset `tune`. With a `seed`, we reseed the global random number generator before each model is tuned.

Each model searches the whole grid, and a portfolio has many models. Start with a small grid.

## Simulate
The [`simulate`](@ref simulate(::MyPortfolioModel, ::Int64)) function returns a `Dict{String,NamedTuple}`, keyed by ticker. Each entry holds the same three matrices as the single-asset `simulate` (`growth_rates`, `states`, and `episodes`), with one row for each day and one column for each path. How we draw the paths depends on the dependence model:

- __Copula:__ For each path, we draw uniform numbers from the copula and simulate one path of each asset from its own model. Then we sort the growth rates of each asset and put them in the order of the ranks of its uniform numbers. The `states` and `episodes` stay in the order of the path before this reorder.
- __Single-index and hybrid:__ The growth rates come from `sample_dependence`. Because these paths have no states, `states` is `0` and `episodes` is `false`. A hybrid portfolio gives `number_of_steps - 1` days, because its `sample_dependence` drops the first day.
- __Sector factor:__ Each path gets its own seed, drawn from `seed`, and comes from `sample_dependence`. An asset has `states` equal to `0`, and `episodes` is `true` on the days when the market, the sector, or the own draw of the asset is in a jump episode. The result also has an entry for the market ticker, with the states and episodes of the market model.

```julia
paths = JumpHMM.simulate(portfolio, 252; number_of_paths = 1000, seed = 5660);
paths["JPM"].growth_rates     # the growth rates of JPM (1/year), days × paths
```

## Validate
The [`validate`](@ref validate(::MyPortfolioModel, ::Array{Float64,2})) function returns a `Dict{String,MyValidationReportModel}`, keyed by ticker. It validates the model of each asset against the growth rates of that asset, with the single-asset [`validate`](@ref validate(::MyJumpHiddenMarkovModel, ::Array{Float64,1})). A sector factor portfolio validates each own model against its own level series instead, because the own model describes the residual of the asset after its sector term.

```julia
reports = JumpHMM.validate(portfolio, prices; number_of_paths = 100, seed = 5660);
reports["JPM"].ks_pass_rate   # the fraction of paths with a Kolmogorov-Smirnov p-value above α
```

## Price columns
The portfolio stores the column of each ticker in the price matrix, the market included, in `tickers_map`. The `tune` and `validate` functions read the prices by these columns. To pass a price matrix with other columns, give `tickers_map` to both functions, and give `market_col` to `tune`.
