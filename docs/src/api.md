# API Reference

```@meta
CurrentModule = JumpHMM
```

This page lists the docstring of every exported type and function. The last section lists the helpers of the sector factor model, which are not exported and are called with the package name.

## Module

```@docs
JumpHMM
```

## Abstract types

```@docs
AbstractMarkovModel
AbstractDependenceModel
AbstractBivariateCopulaModel
AbstractReportModel
```

## Single-asset model

```@docs
MyJumpHiddenMarkovModel
build(::Type{MyJumpHiddenMarkovModel}, ::NamedTuple)
laplace_boundaries
assign_states
estimate_transition
stationary_distribution
estimate_emissions
sample_emission
simulate(::MyJumpHiddenMarkovModel, ::Int64)
tune(::MyJumpHiddenMarkovModel, ::Array{Float64,1})
```

## Validation

```@docs
MyValidationReportModel
validate(::MyJumpHiddenMarkovModel, ::Array{Float64,1})
wasserstein_distance
hellinger_distance
```

## Decoding

```@docs
decode
forward_filter
log_likelihood
```

## Copulas

```@docs
MyGaussianCopulaModel
MyStudentTCopulaModel
build(::Type{MyGaussianCopulaModel}, ::NamedTuple)
build(::Type{MyStudentTCopulaModel}, ::NamedTuple)
sample_dependence(::MyGaussianCopulaModel, ::Int64)
sample_dependence(::MyStudentTCopulaModel, ::Int64)
```

## Vine copula

```@docs
MyGaussianBivariateCopulaModel
MyStudentTBivariateCopulaModel
MyClaytonBivariateCopulaModel
MyGumbelBivariateCopulaModel
MyFrankBivariateCopulaModel
MyVineEdgeModel
MyVineCopulaModel
build(::Type{MyGaussianBivariateCopulaModel}, ::NamedTuple)
build(::Type{MyStudentTBivariateCopulaModel}, ::NamedTuple)
build(::Type{MyClaytonBivariateCopulaModel}, ::NamedTuple)
build(::Type{MyGumbelBivariateCopulaModel}, ::NamedTuple)
build(::Type{MyFrankBivariateCopulaModel}, ::NamedTuple)
build(::Type{MyVineEdgeModel}, ::NamedTuple)
build(::Type{MyVineCopulaModel}, ::NamedTuple)
sample_dependence(::MyVineCopulaModel, ::Int64)
```

## Single-index models

```@docs
MySingleIndexDependenceModel
MyHybridSingleIndexDependenceModel
build(::Type{MySingleIndexDependenceModel}, ::NamedTuple)
build(::Type{MyHybridSingleIndexDependenceModel}, ::NamedTuple)
sample_dependence(::MySingleIndexDependenceModel, ::Int64)
sample_dependence(::MyHybridSingleIndexDependenceModel, ::Int64)
```

## Sector factor model

```@docs
MySectorFactorModel
build(::Type{MySectorFactorModel}, ::NamedTuple)
sample_draws
compose
compose_name(::MySectorFactorModel, ::NamedTuple, ::Int64)
compose_name(::Array{Float64,1}, ::Array{Float64,1}, ::Float64, ::Array{Float64,1}, ::Array{Float64,1}, ::Array{Float64,1})
with_parameters
sample_dependence(::MySectorFactorModel, ::Int64)
```

## Portfolio

```@docs
MyPortfolioModel
build(::Type{MyPortfolioModel}, ::NamedTuple)
tune(::MyPortfolioModel, ::Array{Float64,2})
simulate(::MyPortfolioModel, ::Int64)
validate(::MyPortfolioModel, ::Array{Float64,2})
```

## Growth rates

```@docs
excess_growth_rates(::Array{Float64,1})
excess_growth_rates(::Array{Float64,2})
excess_growth_rates(::Dict{String,DataFrame}, ::Array{String,1})
prices_from_growth_rates
```

## Sector factor helpers

```@docs
ema_rate
drift_path
drift_variance
sector_factors
loo_loadings
sector_loadings
own_residuals
own_variances
own_scale
own_scales
```
