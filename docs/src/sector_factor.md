# Sector Factor Model

```@meta
CurrentModule = JumpHMM
```

The [`MySectorFactorModel`](@ref) type holds a model of many assets, built from three layers of hidden Markov models: one model of the market, one model of each sector, and one own model of each asset. On day ``t``, the growth rate of asset ``i`` in sector ``s`` is

```math
g_i(t) = \alpha_i + \beta_i(t)\, g_m(t) + \delta_i\, f_s(t) + \psi_i \left(\varepsilon_i(t) - \bar{\varepsilon}_i\right)
```

where ``g_m(t)`` is a draw of the market model, ``f_s(t)`` is a draw of the model of sector ``s``, and ``\varepsilon_i(t)`` is a draw of the own model of asset ``i``, all in 1/year. The long-run mean ``\bar{\varepsilon}_i`` of the own model (`ε_center`) centers the own draw. The intercept ``\alpha_i`` (1/year), the market sensitivity ``\beta_i(t)``, the sector loading ``\delta_i``, and the own scale ``\psi_i`` come from the fit below.

The assets are linked only through the market draw and the sector draw that they have in common on each day. A copula would reorder the draws of each asset across days, which would break up the volatility clustering and the timing of the jump episodes that each model produces.

## What each term is fit to
The [`build`](@ref build(::Type{MySectorFactorModel}, ::NamedTuple)) function fits each term to its part of the data:

- __Market term:__ We regress the growth rate of each asset on the growth rate of the market by ordinary least squares, ``g_i(t) = \hat{\alpha}_i + \beta_i\, g_m(t) + e_i(t)``. The fit gives the intercept ``\hat{\alpha}_i``, the sensitivity ``\beta_i``, the R² of each asset (`r²`), and the residual ``e_i(t)``.
- __Sector factor:__ The factor ``f_s(t)`` of sector ``s`` is the mean residual of the assets in that sector on day ``t`` ([`sector_factors`](@ref)).
- __Sector loading:__ The loading ``\delta_i`` starts as the slope of the residual of asset ``i`` on the mean residual of the other assets in its sector ([`loo_loadings`](@ref)). We then scale the loadings of each sector by one number, so that the mean of ``\delta_i \delta_j \operatorname{Var}(f_s)`` over the pairs of assets in the sector equals the mean covariance of their residuals ([`sector_loadings`](@ref)).
- __Own term:__ The own model of asset ``i`` is fit to the residual left after the sector term, ``u_i(t) = e_i(t) - \delta_i f_s(t)`` ([`own_residuals`](@ref)).
- __Hidden Markov models:__ The market model is fit to the growth rates of the market. The sector and own models are fit to the growth rates of level series that compound ``f_s`` and ``u_i`` from a start of `100.0` (`sector_levels`, `own_levels`). These growth rates equal ``f_s`` and ``u_i`` up to rounding.

Every sector needs at least two assets, because the loading of an asset uses the other assets in its sector.

## Intercepts and the drift of β
By default, the intercept ``\alpha_i`` is the fitted ``\hat{\alpha}_i``, and ``\beta_i(t)`` stays at the fitted ``\beta_i``. The optional keys `intercept_pull`, `β_scale`, and `β_halflife` change these defaults.

The key `intercept_pull` moves each intercept toward the mean intercept ``\bar{\alpha}`` of all the assets, ``\alpha_i = \bar{\alpha} + c\,(\hat{\alpha}_i - \bar{\alpha})``, where ``c`` is `intercept_pull`. A value of `1.0`, the default, keeps the fitted intercepts, and `0.0` gives every asset the mean intercept.

The keys `β_scale`, with one value per asset, and `β_halflife` make the sensitivity drift, ``\beta_i(t) = \beta_i + x_i(t)``. The drift ``x_i`` is an exponential moving average of standard normal shocks ``z_i(t)`` ([`drift_path`](@ref)):

```math
x_i(t) = (1 - a)\, x_i(t-1) + a\, k_i\, z_i(t), \qquad a = 1 - 2^{-1/h}
```

where ``k_i`` is the drift scale of asset ``i`` (`β_scale[i]`) and ``h`` is the half-life of the drift, in trading days (`β_halflife`, default `126.0`). The first value is drawn from the stationary distribution of the drift. The drift then has the same variance ``\sigma_{x,i}^{2} = k_i^{2}\, a/(2 - a)`` on every day ([`drift_variance`](@ref)).

The default `β_scale` is all zeros, which keeps every ``\beta_i`` fixed.

## The own scale
The own scale ``\psi_i`` makes the drift of ``\beta_i``, the sector term, and the own term together carry the variance of the residual ``e_i`` ([`own_variances`](@ref), [`own_scale`](@ref)):

```math
\psi_i^{2} = \frac{\operatorname{Var}(e_i) - \delta_i^{2} \operatorname{Var}(f_s) - \sigma_{x,i}^{2} \operatorname{Var}(g_m)}{\operatorname{Var}(u_i)}
```

The residual ``e_i`` comes from a fit with a fixed ``\beta_i``. That residual already holds the movement of the stock's actual ``\beta`` over time. We subtract ``\sigma_{x,i}^{2} \operatorname{Var}(g_m)``, so that a drifting ``\beta_i`` does not add that variance a second time.

The sector factor averages the residuals of its assets, asset ``i`` included. As a result, part of ``u_i`` still moves with ``f_s``. Independent draws of the sector model and the own model lose that shared part, and ``\psi_i`` puts that variance back.

## Usage
The `tickers` list includes the market, and `sector` has one label for each asset, without the market:

```julia
import JumpHMM

tickers = ["SPY", "JPM", "BAC", "XOM", "CVX"];             # one ticker for each column of prices
sector = ["Financials", "Financials", "Energy", "Energy"];  # one label for each asset, without the market

model = JumpHMM.build(JumpHMM.MySectorFactorModel, (
    tickers = tickers,
    prices = prices,          # the share prices (USD/share), one row per day and one column per ticker
    market = "SPY",           # the ticker of the market
    sector = sector,
    number_of_states = 100,
    ν = 5.0,
    β_scale = fill(0.3, 4),   # optional: the drift scale of β for each asset (default all 0.0)
    β_halflife = 126.0        # optional: the half-life of the drift, in trading days (default 126.0)
));
model.tickers                 # ["JPM", "BAC", "XOM", "CVX"]: the assets, without the market

path = JumpHMM.sample_dependence(model, 252; seed = 5660);
path.G                        # the growth rates of the assets (1/year), one row per day and one column per asset
```

The build draws one long path of each own model, `center_steps` days long (default `1_000_000`), and takes its mean as ``\bar{\varepsilon}_i``. A smaller `center_steps`, such as `20_000`, builds faster but gives a less precise ``\bar{\varepsilon}_i``.

The keys `market_model`, `sector_models`, and `own_models` pass in models fitted or tuned elsewhere, instead of the models the build would fit. Each own model must describe the residual of its asset after the sector term, the series that `own_levels` holds. The [`build`](@ref build(::Type{MySectorFactorModel}, ::NamedTuple)) docstring lists every optional key and its default.

## Draws and composition
A path of the universe comes from two steps:

- __Draw:__ The [`sample_draws`](@ref) function draws every random part of one path. These parts are one market path, one path of each sector model, one path of each own model, and the standard normal shocks of the drift of ``\beta``.
- __Compose:__ The [`compose`](@ref) function combines the draws with the parameters of the model into the growth rates of the assets.

The [`sample_dependence`](@ref sample_dependence(::MySectorFactorModel, ::Int64)) function does both steps. The table lists the keys of each result:

| Key | What it holds | Returned by |
|:---|:---|:---|
| `g_m` | The market growth rates (1/year), one per day | `sample_draws`, `sample_dependence` |
| `m_states` | The state of the market model on each day | `sample_draws`, `sample_dependence` |
| `episodes` | A flag that is `true` on the days of a market jump episode | `sample_draws`, `sample_dependence` |
| `F` | The sector factors (1/year), one column per sector | `sample_draws`, `sample_dependence` |
| `sector_episodes` | A flag that is `true` on the days of a jump episode of each sector model | `sample_draws` |
| `E` | The own draws (1/year), one column per asset | `sample_draws` |
| `own_episodes` | A flag that is `true` on the days of a jump episode of each own model | `sample_draws` |
| `Z` | The standard normal shocks of the drift of ``\beta``, one column per asset | `sample_draws` |
| `G` | The growth rates of the assets (1/year), one column per asset | `compose`, `sample_dependence` |
| `β_path` | The market sensitivity of each asset on each day | `compose`, `sample_dependence` |
| `name_episodes` | A flag that is `true` on the days when the market, the sector, or the own draw of an asset is in a jump episode | `compose`, `sample_dependence` |

Each part of the draw has its own seed, made from `seed`. The same `seed` gives the same path, and `seed` must be an `Int64` from `0` up to, but not including, `10^8`. Like [`simulate`](@ref simulate(::MyJumpHiddenMarkovModel, ::Int64)), the function `sample_draws` reseeds the global random number generator.

## Composing the same draws under new parameters
We keep the draws, so that we can compose them again under new parameters. The [`with_parameters`](@ref) function makes a new model with new intercepts, a new half-life, or new drift scales, and it recomputes ``\psi``. The new model shares the fitted hidden Markov models with the input model.

```julia
# draw 20 paths once, and keep the draws -
number_of_days = 252;
bank = Array{NamedTuple,1}();
for p ∈ 1:20
    push!(bank, JumpHMM.sample_draws(model, number_of_days; seed = p));
end

# compose the same draws under three drift scales -
for k ∈ [0.1, 0.3, 0.5]
    trial = JumpHMM.with_parameters(model; β_scale = fill(k, length(model.tickers)));
    for p ∈ 1:20
        G = JumpHMM.compose(trial, bank[p]).G;   # the growth rates (1/year), days × assets
        # compare a statistic of G with its value in the data -
    end
end
```

The [`compose_name`](@ref compose_name(::MySectorFactorModel, ::NamedTuple, ::Int64)) function composes one asset alone, with its own intercept, half-life, or drift scale for that call. Because `compose` reads only the keys it needs, the `draws` can mix the parts of two draws, for example the market from one draw and the sectors from another.

## Changing a fitted model
The build computes the long-run mean ``\bar{\varepsilon}_i`` (`ε_center`) from the fitted own models, and the own scale ``\psi_i`` (`ψ`) from the variances of the data and the drift of ``\beta``. Change the intercepts and the drift of ``\beta`` with [`with_parameters`](@ref), which recomputes ``\psi``.

The functions `compose` and `compose_name` compute the own scale again from the variances. A value written into `ψ` by hand therefore has no effect.

After you change the jump settings or the emissions of the market model, a sector model, or an own model, call the build again with the keys `market_model`, `sector_models`, and `own_models`. The new build computes `ε_center` from the changed models. The rebuild must repeat the settings of the first build, because every key it leaves out takes its default:

```julia
own_models = deepcopy(model.own_models);
own_models[1].ϵ = 2e-3;          # the probability that a jump episode starts on a given day, for the first asset
rebuilt = JumpHMM.build(JumpHMM.MySectorFactorModel, (
    tickers = tickers, prices = prices, market = "SPY", sector = sector, number_of_states = 100, ν = 5.0,
    β_scale = model.β_scale,              # the settings of the first build
    β_halflife = model.β_halflife,
    center_steps = model.center_steps,
    center_seed = model.center_seed,
    market_model = model.market_model, sector_models = model.sector_models, own_models = own_models
));
```

The model from `with_parameters` shares its fitted models with the input model. A change to one of those fitted models changes both sector factor models. Make a `deepcopy` of the model before a study that edits its fitted models.

## Tuning, simulating, and validating
To tune the market, sector, and own models, build a [portfolio](portfolio.md) with `dependence = JumpHMM.MySectorFactorModel`. The portfolio also simulates the universe, and it validates each own model against the residual series it was fit to. The portfolio takes the same `sector` key, and it takes the other keys of this build in `sector_options`.
