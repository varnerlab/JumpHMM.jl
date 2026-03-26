# Single-Index Model

## Overview
The Single-Index Model (SIM) is an alternative to copulas for modeling cross-asset dependence. All correlation flows through a single market factor:

```math
G_i = \alpha_i + \beta_i \times G_{\text{market}} + \varepsilon_i
```

where `αᵢ` is the intercept, `βᵢ` is the market beta, and `εᵢ ~ N(0, σ_εᵢ)` is the idiosyncratic residual.

## When to Use
- **Large universes** (100+ assets): SIM needs only 2 parameters per asset vs `n(n-1)/2` for copulas
- **Paper reproducibility**: The paper uses SIM for the 424 S&P 500 constituent extension
- **Quick baseline**: Simpler to fit and understand

## Trade-offs vs Copulas
- Cannot capture sector-level co-movement independent of the market factor
- No explicit tail dependence control (inherited from market HMM)
- All pairwise correlations are determined by the betas: `cor(i,j) ≈ βᵢβⱼσ²_market`

## Usage
```julia
sim = fit(SingleIndexModel, asset_returns, market_returns;
          market_prices=market_prices, N=100, ν=5.0)

# Generate correlated asset returns
dep_returns = sample_dependence(sim, 252)
```

The `market_prices` parameter is optional. If omitted, prices are reconstructed from returns using a dummy initial price `P₀ = 100`.

**Note**: The market factor must have nonzero variance. Constant market returns will raise an `ArgumentError`.

## API
