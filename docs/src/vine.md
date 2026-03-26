# Vine Copula

## Overview
A vine copula (C-vine) models cross-asset dependence by assigning a **separate bivariate copula** to each pair of assets. Unlike Gaussian or Student-t copulas that apply a single dependence structure uniformly, vine copulas allow heterogeneous dependence — tech stocks can crash together (Clayton, lower tail) while energy-finance pairs have symmetric dependence (Gaussian).

## Bivariate Copula Families
Five families are available, automatically selected per edge via AIC:

| Family | Tail Dependence | Symmetry | Use Case |
|--------|----------------|----------|----------|
| `GaussianBiCopula` | None | Symmetric | Weak/normal dependence |
| `StudentTBiCopula` | Both tails | Symmetric | Joint crashes + rallies |
| `ClaytonBiCopula` | Lower tail | Asymmetric | Crash-together, diverge in rallies |
| `GumbelBiCopula` | Upper tail | Asymmetric | Rally-together, diverge in crashes |
| `FrankBiCopula` | None | Symmetric | Moderate, no tail dependence |

## How It Works
The C-vine organizes variables into a tree structure:
- **Tree 1**: The most connected variable (highest sum of |Kendall's tau|) is the center. Bivariate copulas link it to all other variables.
- **Tree 2**: Conditional dependencies are modeled using pseudo-observations computed via h-functions from Tree 1.
- **Trees 3..d-1**: Higher-order conditional dependencies.

Each edge in each tree gets the bivariate family that minimizes AIC, so the model adapts to the specific dependence pattern of each pair.

## Usage
```julia
using JumpHMM

# Fit a vine copula from returns
vc = fit(VineCopula, returns_matrix)

# Sample correlated uniforms
U = sample_dependence(vc, 1000)

# Use in a portfolio model
portfolio = fit(PortfolioModel, tickers, price_matrix;
    dependence=VineCopula, rf=0.05, N=100)
```

## When to Use
- **20-100 assets**: Where heterogeneous dependence matters but copula estimation is still feasible
- **Sector-diverse portfolios**: Where different sectors have different dependence patterns
- **Tail-risk analysis**: Where asymmetric tail dependence (crashes vs rallies) is important
