# Simulation

## Overview
The `simulate` function generates synthetic paths from a fitted Jump-HMM.

### Algorithm
At each time step:
1. With probability `(1 - ϵ)`: normal Markovian transition via row of `T`
2. With probability `ϵ`: a jump is triggered
   - Duration `K ~ Poisson(λ)` consecutive steps
   - Each jump step lands in bottom `N_tail` states (probability `p_neg`) or top `N_tail` states (probability `1 - p_neg`), uniformly within the tail
3. For each state in the chain, an observation is sampled from the corresponding Student-t emission

## Usage
```julia
result = simulate(model, 252; n_paths=1000, seed=1234)

# Access individual paths
path = result.paths[1]
path.states        # Vector{Int} — hidden state indices
path.observations  # Vector{Float64} — sampled excess growth rates
path.jumps         # Vector{Bool} — true at jump steps
```

## API

## Types
