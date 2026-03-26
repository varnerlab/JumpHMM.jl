# JumpHMM.jl

`JumpHMM.jl` is a [Julia](https://julialang.org) package implementing a Hybrid Hidden Markov Model with Poisson jump-diffusion for generating synthetic equity time series. The package implements the method described in:

> Alswaidan A. and Varner J. *Hybrid Hidden Markov Model for Modeling Equity Excess Growth Rate Dynamics: A Discrete-State Approach with Jump-Diffusion*. Cornell University.

The model reproduces three canonical stylized facts of financial returns:
- **Heavy-tailed distributions** (leptokurtosis)
- **Negligible linear autocorrelation** in raw returns
- **Persistent volatility clustering** (ARCH effect in absolute returns)

## Features
- Single-asset pipeline: `fit` → `tune` → `simulate` → `validate` → `decode`
- Four multi-asset dependence models: `GaussianCopula`, `StudentTCopula`, `VineCopula`, `SingleIndexModel`
- Immutable type system — all fitting/tuning operations return new instances
- Self-contained — all finance helpers built in

## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/varnerlab/JumpHMM.jl.git")
```

## Disclaimer and Risks
__This content is offered solely for training and informational purposes__. No offer or solicitation to buy or sell securities or derivative products or any investment or trading advice or strategy is made, given, or endorsed by the teaching team.

__Trading involves risk__. Carefully review your financial situation before investing in securities, futures contracts, options, or commodity interests. Past performance, whether actual or indicated by historical tests of strategies, is no guarantee of future performance or success. Trading is generally inappropriate for someone with limited resources, investment or trading experience, or a low-risk tolerance. Only risk capital that is not required for living expenses.

__You are fully responsible for any investment or trading decisions you make__. Such decisions should be based solely on evaluating your financial circumstances, investment or trading objectives, risk tolerance, and liquidity needs.
