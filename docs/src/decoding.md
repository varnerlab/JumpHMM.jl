# Decoding

## Overview
Decoding recovers the most likely hidden state sequence from observed data, or computes filtered state probabilities. All three functions require non-empty observation sequences.

## Viterbi Decoding
Find the most likely state sequence given observed excess growth rates. Computed entirely in log-space for numerical stability.
```julia
G = excess_growth_rates(prices; rf=0.05)
states = decode(model, G)
# states is Vector{Int} of length length(G)
```

## Forward Filtering
Compute filtered state probabilities `P(S_t = k | G_{1:t})`. Implemented in log-space with log-sum-exp normalization.
```julia
alpha = forward_filter(model, G)
# alpha is (n_steps × N) matrix, each row sums to 1
```

For extreme observations where all emission likelihoods underflow, the filter falls back to a uniform distribution over states rather than returning `NaN`.

## Log-Likelihood
Compute the log-likelihood of an observation sequence under the model. Implemented in log-space with log-sum-exp.
```julia
ll = log_likelihood(model, G)
# ll is a scalar Float64
```

Returns `-Inf` (not `NaN`) when observations are so extreme that no state has nonzero likelihood.

## API
