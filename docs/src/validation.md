# Validation

```@meta
CurrentModule = JumpHMM
```

The [`validate`](@ref validate(::MyJumpHiddenMarkovModel, ::Array{Float64,1})) function simulates paths from a model and compares each path with the growth rates of the data. Each path has as many days as the data. The result is a [`MyValidationReportModel`](@ref) that holds one value per path for each measure, and the summaries over the paths.

## Usage
The comments give the default of each keyword:

```julia
report = JumpHMM.validate(tuned_model, g;
    number_of_paths = 1000,   # the number of simulated paths (default 1000)
    α = 0.05,                 # the significance level of the tests (default 0.05)
    acf_lags = 25,            # the number of lags of the autocorrelation (default 25)
    seed = 5660               # the seed of the random number generator (default nothing)
);
report.ks_pass_rate           # the fraction of paths with a Kolmogorov-Smirnov p-value above α
report.mean_acf_mae           # the mean autocorrelation error over the paths
```

The model holds no risk-free rate. Pass excess growth rates made with the same `rf` as the growth rates that built the model.

## What validate measures
For each path, `validate` computes six measures:

- __Kolmogorov-Smirnov test:__ The two-sample test compares the distribution of the path with the distribution of the data by the largest gap between their cumulative distribution functions. The p-value of each path is in `ks_pvalue`.
- __Anderson-Darling test:__ The k-sample test also compares the two distributions, with more weight on the tails than the Kolmogorov-Smirnov test. The p-value of each path is in `ad_pvalue`.
- __Wasserstein distance:__ The [`wasserstein_distance`](@ref) between the path and the data, in 1/year, is in `wasserstein`.
- __Hellinger distance:__ The [`hellinger_distance`](@ref) between the path and the data, from `0.0` for the same histogram to `1.0` for histograms with no bin in common, is in `hellinger`.
- __Autocorrelation error:__ The mean absolute difference between the autocorrelation of the absolute growth rate in the path and in the data, over the lags `1` to `acf_lags`, is in `acf_mae`.
- __Excess kurtosis:__ The excess kurtosis of each path is in `kurtosis_sim`, and the excess kurtosis of the data is in `kurtosis_emp`.

A path passes a test when its p-value is above the significance level `α`. The [Helpers](helpers.md) page explains the two distances.

## Summaries
The report also holds the summaries over the paths:

| Field | What it holds |
|:---|:---|
| `ks_pass_rate` | The fraction of paths with a Kolmogorov-Smirnov p-value above `α` |
| `ad_pass_rate` | The fraction of paths with an Anderson-Darling p-value above `α` |
| `mean_acf_mae` | The mean of `acf_mae` over the paths |
| `mean_wasserstein` | The mean of `wasserstein` over the paths (1/year) |
| `mean_hellinger` | The mean of `hellinger` over the paths |
| `mean_kurtosis` | The mean of `kurtosis_sim` over the paths |
| `α` | The significance level of the tests |
