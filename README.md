# JumpHMM.jl

[![CI](https://github.com/varnerlab/JumpHMM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/varnerlab/JumpHMM.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/varnerlab/JumpHMM.jl/actions/workflows/documentation.yml/badge.svg)](https://varnerlab.github.io/JumpHMM.jl/dev/)

The JumpHMM.jl package builds a hidden Markov model with jump episodes from the daily growth rates of a stock, and it simulates synthetic growth rates from that model. It also builds models of many assets, with a dependence model that links the assets. The method is described in:

> Alswaidan A. and Varner J. *Hybrid Hidden Markov Model for Modeling Equity Excess Growth Rate Dynamics: A Discrete-State Approach with Jump-Diffusion*. Cornell University.

## Installation
Install the package by running the following command in the Julia REPL:

```julia
using Pkg
Pkg.add(url = "https://github.com/varnerlab/JumpHMM.jl.git")
```

## Getting started
The example below makes a synthetic share price series, builds and tunes a model, and simulates paths. We write the package name in front of every type and function.

```julia
import JumpHMM
using Random

# make a synthetic share price series: 1000 trading days, more volatile from day 601 to day 700 -
Random.seed!(1234);
number_of_days = 1000;
Δt = 1/252;                 # one trading day, in years
S = Array{Float64,1}(undef, number_of_days + 1);
S[1] = 100.0;               # the first share price (USD/share)
for t ∈ 1:number_of_days
    σ = 0.15;               # the volatility of a calm day (1/sqrt(year))
    if t > 600 && t <= 700
        σ = 0.45;           # the volatility of a volatile day (1/sqrt(year))
    end
    # compute the next share price, where 0.08 is the drift of the share price (1/year) -
    S[t+1] = S[t] * exp((0.08 - 0.5 * σ^2) * Δt + σ * sqrt(Δt) * randn());
end

g = JumpHMM.excess_growth_rates(S; rf = 0.0, Δt = Δt);  # the growth rates (1/year)

model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (
    growth_rates = g,       # the growth rates (1/year)
    number_of_states = 20,  # the number of hidden states
    ν = 5.0                 # the degrees of freedom of the Student-t emissions
));

tuned_model = JumpHMM.tune(model, g;
    ϵ_range = range(1e-3, 5e-3, length = 3),  # the candidate probabilities that a jump episode starts on a given day
    λ_range = range(10.0, 50.0, length = 3),  # the candidate values of the mean of the Poisson episode length (trading days)
    number_of_paths = 20,                     # the number of simulated paths for each pair
    seed = 5660                               # the seed of the random number generator
);

paths = JumpHMM.simulate(tuned_model, length(g); number_of_paths = 100, seed = 5660);
```

The [documentation](https://varnerlab.github.io/JumpHMM.jl/dev/) explains each step and the models of many assets.

## Disclaimer and Risks
__This content is offered solely for training and informational purposes__. No offer or solicitation to buy or sell securities or derivative products or any investment or trading advice or strategy is made, given, or endorsed by the teaching team.

__Trading involves risk__. Carefully review your financial situation before investing in securities, futures contracts, options, or commodity interests. Past performance, whether actual or indicated by historical tests of strategies, is no guarantee of future performance or success. Trading is generally inappropriate for someone with limited resources, investment or trading experience, or a low-risk tolerance. Only risk capital that is not required for living expenses.

__You are fully responsible for any investment or trading decisions you make__. Such decisions should be based solely on evaluating your financial circumstances, investment or trading objectives, risk tolerance, and liquidity needs.

## License
MIT License. See `LICENSE`.
