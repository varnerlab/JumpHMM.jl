# Copulas

## Overview
Copulas capture cross-asset dependence without modifying the marginal distributions. Each asset gets its own Jump-HMM for marginal dynamics, while a copula models how the assets co-move.

Two copula types are available:

| Copula | Tail Dependence | Parameters | Best For |
|--------|----------------|------------|----------|
| `GaussianCopula` | None | Correlation matrix `Σ` | Simple baseline |
| `StudentTCopula` | Yes (via `ν`) | Correlation matrix `Σ` + `ν` | Realistic portfolios |

## Gaussian Copula
Captures linear dependence via a correlation matrix. Simple but underestimates the tendency for assets to crash together.

```julia
gc = fit(GaussianCopula, returns_matrix)
U = sample_dependence(gc, 1000)  # (1000 × n_assets) uniform matrix
```

## Student-t Copula
Adds a degrees-of-freedom parameter `ν` that controls tail dependence — the tendency for extreme co-movements. Lower `ν` means stronger tail dependence.

The `ν` parameter is automatically selected via profile maximum likelihood over the candidates `[3, 4, 5, 6, 7, 8, 10, 15, 20, 30]`.

```julia
tc = fit(StudentTCopula, returns_matrix)
tc.ν  # selected degrees of freedom
U = sample_dependence(tc, 1000)
```

## Positive-Definiteness
Both copula types automatically regularize the estimated correlation matrix to ensure positive-definiteness. This prevents `PosDefException` errors when sampling from nearly collinear assets.

## API
