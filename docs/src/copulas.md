# Copulas

```@meta
CurrentModule = JumpHMM
```

A copula is a joint distribution of uniform numbers, one for each asset. It describes how the assets move together, apart from the distribution of each asset. The package has two copulas for many assets, [`MyGaussianCopulaModel`](@ref) and [`MyStudentTCopulaModel`](@ref), and a vine copula, which the [Vine Copula](vine.md) page describes.

## Build
Both copulas are built from a matrix of growth rates, with one row per day and one column per asset. The matrix needs at least two rows and one column, and every value must be finite. A matrix of one column gives a copula of one asset.

```julia
R = JumpHMM.excess_growth_rates(prices; rf = 0.0, Δt = 1/252);   # the growth rates (1/year), days × assets
copula = JumpHMM.build(JumpHMM.MyStudentTCopulaModel, (growth_rates = R,));
copula.Σ   # the correlation matrix of the copula
copula.ν   # the degrees of freedom of the copula
```

We replace each column of growth rates by its ranks divided by the number of days plus one. The division gives uniform numbers strictly between `0` and `1` that keep only the order of the growth rates of each asset. The correlation matrix ``\Sigma`` of these uniform numbers is the correlation matrix of the copula.

If ``\Sigma`` is not positive definite, we raise each eigenvalue below `1e-8` to `1e-8` and rescale ``\Sigma`` to ones on the diagonal. The sampler needs a positive definite matrix, because it factors ``\Sigma``.

The Student-t copula also has the degrees of freedom ``\nu``. We pick ``\nu`` from `3, 4, 5, 6, 7, 8, 10, 15, 20, 30` as the value with the largest log-likelihood. A smaller ``\nu`` makes extreme moves of several assets on the same day more likely.

## Sample
The [`sample_dependence`](@ref sample_dependence(::MyStudentTCopulaModel, ::Int64)) function draws uniform numbers from a copula, with one row per day and one column per asset. The draws come from the global random number generator. To repeat a sample, call `Random.seed!` first.

```julia
import Random

Random.seed!(5660);
U = JumpHMM.sample_dependence(copula, 252);   # 252 days × assets, every value strictly between 0 and 1
```

In a [portfolio](portfolio.md), we put the simulated growth rates of each asset in the order of the ranks of its column of `U`. The reorder keeps the values of each asset and changes their order in time. As a result, a copula portfolio loses the volatility clustering of each path.
