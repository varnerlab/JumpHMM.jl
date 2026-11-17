# Single-Index Models

```@meta
CurrentModule = JumpHMM
```

A single-index model links the assets through one market factor. For each asset ``i``, we regress its growth rate on the growth rate of the market,

```math
g_i(t) = \alpha_i + \beta_i\, g_m(t) + \varepsilon_i(t)
```

where ``\alpha_i`` is the intercept (1/year), ``\beta_i`` is the sensitivity of the asset to the market, and ``\varepsilon_i(t)`` is the residual (1/year). The package has two single-index models, which draw the residuals in different ways.

## Single-index model
The [`MySingleIndexDependenceModel`](@ref) type holds the intercepts, the sensitivities, the residuals, and a hidden Markov model of the market. Its [`build`](@ref build(::Type{MySingleIndexDependenceModel}, ::NamedTuple)) takes the growth rates of the assets and of the market:

```julia
G = JumpHMM.excess_growth_rates(prices; rf = 0.0, Δt = 1/252);   # the growth rates (1/year), days × tickers, market in column 1
model = JumpHMM.build(JumpHMM.MySingleIndexDependenceModel, (
    growth_rates = G[:, 2:end],       # the growth rates of the assets
    market_growth_rates = G[:, 1],    # the growth rates of the market
    market_prices = prices[:, 1],     # optional: the share prices of the market (default nothing)
    number_of_states = 100,
    ν = 5.0,
    residual_method = :bootstrap      # optional: :bootstrap or :gaussian (default :bootstrap)
));
```

We estimate ``\beta_i = \operatorname{cov}(g_i, g_m)/\operatorname{var}(g_m)`` and ``\alpha_i = \bar{g}_i - \beta_i\, \bar{g}_m``, where a bar marks the mean over the days, and keep the residuals. The market model is fit to the excess growth rates of `market_prices`, made with the `rf` and `Δt` keys. Without `market_prices`, we rebuild market prices from `market_growth_rates`, starting at `100.0`, and fit the market model to their growth rates.

The [`sample_dependence`](@ref sample_dependence(::MySingleIndexDependenceModel, ::Int64)) function simulates one market path from the market model and adds ``\alpha_i + \beta_i\, g_m(t)`` to a residual for each asset. It returns the growth rates of the assets, with one row per day and one column per asset. The key `residual_method` sets how we draw the residuals:

- __Bootstrap:__ With `:bootstrap`, each simulated day takes the residuals of one training day, drawn at random. All the assets use the same training day, which keeps the correlation between their residuals.
- __Gaussian:__ With `:gaussian`, each residual is a normal draw with the standard deviation of the residuals of its asset (`σ_ε`). The residuals of two assets are independent, and all of their co-movement comes from the market term.

## Hybrid single-index model
The [`MyHybridSingleIndexDependenceModel`](@ref) type keeps a hidden Markov model of each asset, so that each asset keeps its own heavy tails and volatility. Its [`build`](@ref build(::Type{MyHybridSingleIndexDependenceModel}, ::NamedTuple)) takes a matrix of share prices:

```julia
model = JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel, (
    tickers = tickers,                    # one ticker for each column of prices, the market included
    prices = prices,                      # the share prices (USD/share), one row per day
    market = "SPY",                       # the ticker of the market
    number_of_states = 100,
    ν = 5.0,
    copula = JumpHMM.MyStudentTCopulaModel,   # optional: the copula type for the residuals (default shown)
    r2_preserve_threshold = 0.80,         # optional: the R² at or above which we keep the R² of an asset (default 0.80)
    idiosyncratic_floor = 0.10            # optional: the smallest fraction of the variance of an asset left to its residual (default 0.10)
));
```

The build fits a hidden Markov model to the market and to each asset. It regresses each asset on the market for ``\alpha_i``, ``\beta_i``, and the R² of the asset, ``R_i^2``, and it fits the copula to the residuals.

The [`sample_dependence`](@ref sample_dependence(::MyHybridSingleIndexDependenceModel, ::Int64)) function simulates a market path and one path of each asset from its own model, and drops the first day of every path. The result has `number_of_steps - 1` rows.

Let ``\sigma_m^2`` be the variance of the growth rate of the market in the data, and ``\sigma_i^2`` the variance of the simulated path of asset ``i``. We scale the path of each asset into its residual term in one of two ways:

- __High R²:__ If ``R_i^2`` is at least `r2_preserve_threshold`, we scale the path to the variance ``\beta_i^2 \sigma_m^2 (1 - R_i^2)/R_i^2``. A simulated asset would have the R² of the data if the simulated market had the variance ``\sigma_m^2``.
- __Lower R²:__ Otherwise, we keep the variance ``\sigma_i^2`` of the asset. The market term carries the fraction ``q_i = \beta_i^2 \sigma_m^2/\sigma_i^2`` of that variance, and the scaled path carries the rest. If ``q_i`` is above ``1 - \phi``, where ``\phi`` is `idiosyncratic_floor`, we lower ``\beta_i`` so that the scaled path keeps the fraction ``\phi``.

Last, we reorder the scaled path of each asset so that its ranks follow the ranks of a copula draw, and we add ``\alpha_i + \beta_i\, g_m(t)``.
