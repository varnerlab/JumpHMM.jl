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

## Hybrid Single-Index Model

The `HybridSingleIndexModel` extends the basic SIM with a variance-correction trick that prevents the naive composition `G_i = α_i + β_i G_market + ε_i` from inflating asset variance by `β²σ²_market`. It combines:

- **Per-ticker HMM marginals** for idiosyncratic draws (preserves heavy tails, regime switching)
- **Variance correction** so that `Var[G_i] ≈ Var[marginal_i]`
- **Copula rank-reordering** to preserve cross-sectional dependence after scaling
- **β clipping** with an idiosyncratic floor to prevent degenerate paths

### Two construction branches

The model selects a branch per ticker based on the calibrated R²:

**Marginal-preserving** (`r² < r2_preserve_threshold`, default 0.80): scales ε so that the total variance matches the HMM marginal variance. If the market loading `ρ = β²σ²_m / σ²_gen` exceeds `1 - f` (where `f` is the idiosyncratic floor, default 0.10), β is clipped downward:

```math
s_i^2 = 1 - \rho_i, \quad \rho_i = \frac{\beta_i^2 \sigma_m^2}{\sigma_{\text{gen},i}^2}
```

**R²-preserving** (`r² ≥ threshold`): targets the residual variance that recovers the calibrated R² exactly. For index ETFs (SPY, QQQ) this ensures the synthetic data reproduces both β and R². The SPY limit (R² = 1) yields ε ≡ 0, making `G = α + β G_market` deterministic.

```math
\sigma_{\varepsilon,\text{target}}^2 = \beta_i^2 \sigma_m^2 \cdot \frac{1 - R^2}{R^2}
```

### Usage

```julia
# Standalone
model = fit(HybridSingleIndexModel, tickers, price_matrix, "SPY";
            copula_type=StudentTCopula, N=100, ν=5.0,
            r2_preserve_threshold=0.80, idiosyncratic_floor=0.10)

dep_returns = sample_dependence(model, 252)  # (251 × n_assets)

# Via PortfolioModel (recommended)
portfolio = fit(PortfolioModel, tickers, price_matrix;
                dependence=HybridSingleIndexModel, market="SPY")
result = simulate(portfolio, 252; n_paths=1000)
```

### Pluggable copula

The copula is fitted on OLS residuals (not raw returns) to avoid double-counting market correlation. Any `AbstractDependenceModel` can be used:

```julia
# Gaussian copula instead of default Student-t
model = fit(HybridSingleIndexModel, tickers, prices, "SPY";
            copula_type=GaussianCopula)
```

## API
