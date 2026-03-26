# Finance Helpers

## Overview
Built-in functions for computing excess growth rates and reconstructing price paths.

## Excess Growth Rates
Annualized excess log-growth rates:

```math
G_t = \frac{1}{\Delta t} \ln\left(\frac{P_t}{P_{t-1}}\right) - r_f
```

Supports three input formats:

```julia
# From a price vector → returns Vector{Float64} of length n-1
G = excess_growth_rates(prices; rf=0.05, dt=1/252)

# From a price matrix (n_obs × n_assets) → returns (n_obs-1 × n_assets) matrix
G = excess_growth_rates(price_matrix; rf=0.05, dt=1/252)

# From a Dict{String, DataFrame} → returns (n_obs-1 × n_tickers) matrix
G = excess_growth_rates(data, tickers; rf=0.05, price_col=:close)
```

## Price Reconstruction
Convert excess growth rates back to prices:

```julia
P = prices_from_growth_rates(G, P0; rf=0.05, dt=1/252)
```

This is the exact inverse: `prices_from_growth_rates(excess_growth_rates(p, rf=r), p[1], rf=r) ≈ p`.

## API
