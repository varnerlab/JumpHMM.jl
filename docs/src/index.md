# JumpHMM.jl

The JumpHMM.jl package builds a hidden Markov model with jump episodes from the daily growth rates of a stock, and it simulates synthetic growth rates from that model. It also builds models of many assets, with a dependence model that links the assets. The method is described in:

> Alswaidan A. and Varner J. *Hybrid Hidden Markov Model for Modeling Equity Excess Growth Rate Dynamics: A Discrete-State Approach with Jump-Diffusion*. Cornell University.

## Growth rates
We model growth rates, not share prices. The growth rate of a share price ``S`` over one time step ``\Delta t``, in years, is

```math
g = \frac{1}{\Delta t} \ln\left(\frac{S(t + \Delta t)}{S(t)}\right)
```

in units of 1/year. The excess growth rate subtracts the risk-free rate ``r_f``, in 1/year, from ``g``.

## Stylized facts
The model reproduces three stylized facts of the daily growth rates of stocks:

- __Heavy tails:__ Large moves happen more often than a normal distribution allows. The excess kurtosis of the growth rate is above zero.
- __Little linear autocorrelation:__ The growth rate on one day says almost nothing about the growth rate on the next day.
- __Volatility clustering:__ Large moves tend to follow large moves. The autocorrelation of the absolute growth rate stays positive for many days.

The hidden Markov model gives the heavy tails and the small linear autocorrelation. Synthetic growth-rate paths simulated without jump episodes show less volatility clustering than the market data. The jump episodes add the missing volatility clustering.

## Installation
The package can be installed by running the following command in the Julia REPL:

```julia
using Pkg
Pkg.add(url = "https://github.com/varnerlab/JumpHMM.jl.git")
```

## Calling the package
We load the package with `import JumpHMM`, and we write the package name in front of every type and function:

```julia
import JumpHMM

model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = g, number_of_states = 100, ν = 5.0));
```

The package name keeps these functions apart from functions with the same names in other packages, for example a `build` function in another package you load. Every code example on these pages follows this rule.

## Disclaimer and Risks
__This content is offered solely for training and informational purposes__. No offer or solicitation to buy or sell securities or derivative products or any investment or trading advice or strategy is made, given, or endorsed by the teaching team.

__Trading involves risk__. Carefully review your financial situation before investing in securities, futures contracts, options, or commodity interests. Past performance, whether actual or indicated by historical tests of strategies, is no guarantee of future performance or success. Trading is generally inappropriate for someone with limited resources, investment or trading experience, or a low-risk tolerance. Only risk capital that is not required for living expenses.

__You are fully responsible for any investment or trading decisions you make__. Such decisions should be based solely on evaluating your financial circumstances, investment or trading objectives, risk tolerance, and liquidity needs.
