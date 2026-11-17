# Vine Copula

```@meta
CurrentModule = JumpHMM
```

A vine copula builds the dependence of many assets from copulas of two assets, one on each edge of a sequence of trees. Each edge picks its own copula family. Different pairs of assets can then have different kinds of dependence.

The [`MyVineCopulaModel`](@ref) type holds a C-vine, in which each tree has one center asset linked to every later asset.

## Bivariate copula families
Each edge holds one of five bivariate copula families:

| Type | Parameters | Tail dependence |
|:---|:---|:---|
| [`MyGaussianBivariateCopulaModel`](@ref) | the correlation `ρ`, strictly between `-1` and `1` | none |
| [`MyStudentTBivariateCopulaModel`](@ref) | the correlation `ρ` and the degrees of freedom `ν`, above `0` | both tails |
| [`MyClaytonBivariateCopulaModel`](@ref) | the dependence parameter `θ`, above `0` | lower tail |
| [`MyGumbelBivariateCopulaModel`](@ref) | the dependence parameter `θ`, at least `1` | upper tail |
| [`MyFrankBivariateCopulaModel`](@ref) | the dependence parameter `θ`, finite | none |

Lower tail dependence means that two assets tend to fall together on their worst days. Upper tail dependence means that two assets tend to rise together on their best days. Each family can also be built from its parameters, for example `JumpHMM.build(JumpHMM.MyClaytonBivariateCopulaModel, (θ = 2.0,))`.

## How build fits the vine
The [`build`](@ref build(::Type{MyVineCopulaModel}, ::NamedTuple)) function fits the vine to a matrix of growth rates, with one row per day and one column per asset, in four steps:

1. __Uniform numbers:__ We replace each column of growth rates by its ranks divided by the number of days plus one.
2. __Order:__ We order the assets by the sum of the absolute Kendall's tau with every other asset, largest first.
3. __Tree ``k``:__ The ``k``-th asset in this order is the center of tree ``k``, and it is paired with each later asset. We fit each of the five families to the pair from its Kendall's tau. The edge keeps the family with the smallest Akaike information criterion, two times the number of parameters minus two times the log-likelihood.
4. __Next tree:__ We replace the uniform numbers of each later asset by their conditional values given the center. Tree ``k + 1`` then describes the dependence that is left after tree ``k``.

The Student-t family also picks its degrees of freedom by log-likelihood. A family whose log-density is not finite on some day is skipped. A vine of ``d`` assets has ``d - 1`` trees, and tree ``k`` has ``d - k`` edges.

```julia
vine = JumpHMM.build(JumpHMM.MyVineCopulaModel, (growth_rates = R,));   # R: the growth rates (1/year), days × assets
vine.order                    # the order of the assets in the vine
vine.edges[1][1].copula       # the bivariate copula of the first edge of tree 1
```

Each edge is a [`MyVineEdgeModel`](@ref) with the two assets `first` and `second`, the assets it is conditioned on (`conditioned`), and its bivariate copula (`copula`).

## Sample
The [`sample_dependence`](@ref sample_dependence(::MyVineCopulaModel, ::Int64)) function draws uniform numbers from the vine, with one row per day and one column per asset. For each day, it draws one independent uniform number per asset and passes these uniform numbers through the inverse conditional distributions of the edges, from the last tree back to the first.

The draws come from the global random number generator. A [portfolio](portfolio.md) uses the vine draws as it uses the uniform numbers of the other copulas.

```julia
U = JumpHMM.sample_dependence(vine, 252);   # 252 days × assets, every value strictly between 0 and 1
```
