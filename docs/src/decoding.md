# Decoding

## Overview
Decoding recovers the most likely hidden state sequence from observed data, or computes filtered state probabilities.

## Viterbi Decoding
Find the most likely state sequence given observed excess growth rates:
```julia
G = excess_growth_rates(prices; rf=0.05)
states = decode(model, G)
```

## Forward Filtering
Compute filtered state probabilities `P(S_t = k | G_{1:t})`:
```julia
alpha = forward_filter(model, G)
# alpha is (n_steps × N) matrix, each row sums to 1
```

## Log-Likelihood
Compute the log-likelihood of an observation sequence:
```julia
ll = log_likelihood(model, G)
```

## API
