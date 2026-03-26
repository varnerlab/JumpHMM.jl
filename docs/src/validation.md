# Validation

## Overview
The `validate` function performs statistical validation by simulating paths and comparing them against empirical data using multiple metrics.

## Metrics
For each simulated path, the following are computed:
- **KS p-value**: Kolmogorov-Smirnov two-sample test
- **AD p-value**: Anderson-Darling k-sample test
- **Wasserstein-1**: Earth mover's distance (interpolated to common quantile grid if lengths differ)
- **Hellinger**: Histogram-based Hellinger distance (bin count = `max(30, √n)`)
- **ACF-MAE**: Mean absolute error of autocorrelation of |returns|
- **Excess kurtosis**: Simulated vs empirical

## Usage
```julia
report = validate(model, prices; n_paths=1000, α=0.05)

report.ks_pass_rate      # fraction of paths with KS p-value > α
report.ad_pass_rate      # fraction of paths with AD p-value > α
report.mean_acf_mae      # mean ACF-MAE across paths
report.mean_wasserstein   # mean Wasserstein-1 distance
report.mean_hellinger     # mean Hellinger distance
report.mean_kurtosis      # mean excess kurtosis across paths
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_paths` | `1000` | Number of Monte Carlo paths to simulate |
| `α` | `0.05` | Significance level for KS/AD pass rates |
| `acf_lags` | `25` | Number of ACF lags (auto-clamped to sample length if needed) |
| `rf` | `model.rf` | Risk-free rate for computing growth rates |
| `seed` | `nothing` | Random seed for reproducibility |

## API

## Types
