"""
    decode(model, observations) → Vector{Int}

Viterbi algorithm: find the most likely state sequence. Computed in log-space.
"""
function decode(model::JumpHiddenMarkovModel,
                observations::AbstractVector{Float64})
    isempty(observations) && throw(ArgumentError("observations must be non-empty"))
    N = model.partition.N
    T_steps = length(observations)
    ν = model.ν

    # precompute log transition matrix and emission distributions
    log_T = log.(model.transition)
    dists = [model.emissions[k].μ + model.emissions[k].σ * TDist(ν)
             for k in 1:N]

    # Viterbi tables
    delta = Matrix{Float64}(undef, T_steps, N)
    psi = Matrix{Int}(undef, T_steps, N)

    # initialization
    for k in 1:N
        lp = model.stationary[k] > 0.0 ? log(model.stationary[k]) : -Inf
        delta[1, k] = lp + logpdf(dists[k], observations[1])
        psi[1, k] = 0
    end

    # recursion
    for t in 2:T_steps
        for k in 1:N
            best_val = -Inf
            best_j = 1
            for j in 1:N
                val = delta[t-1, j] + log_T[j, k]
                if val > best_val
                    best_val = val
                    best_j = j
                end
            end
            delta[t, k] = best_val + logpdf(dists[k], observations[t])
            psi[t, k] = best_j
        end
    end

    # backtrack
    states = Vector{Int}(undef, T_steps)
    states[T_steps] = argmax(delta[T_steps, :])
    for t in (T_steps-1):-1:1
        states[t] = psi[t+1, states[t+1]]
    end

    return states
end

"""
    forward_filter(model, observations) → Matrix{Float64}

Forward algorithm: compute P(S_t = k | G_{1:t}). Returns (n_steps × N) matrix.
Implemented in log-space with log-sum-exp normalization for numerical stability.
"""
function forward_filter(model::JumpHiddenMarkovModel,
                        observations::AbstractVector{Float64})
    isempty(observations) && throw(ArgumentError("observations must be non-empty"))
    N = model.partition.N
    T_steps = length(observations)
    ν = model.ν

    dists = [model.emissions[k].μ + model.emissions[k].σ * TDist(ν)
             for k in 1:N]
    log_T = log.(model.transition)

    alpha = Matrix{Float64}(undef, T_steps, N)
    log_alpha = Vector{Float64}(undef, N)

    # initialization (log-space)
    for k in 1:N
        lp = model.stationary[k] > 0.0 ? log(model.stationary[k]) : -Inf
        log_alpha[k] = lp + logpdf(dists[k], observations[1])
    end
    # normalize to probabilities via log-sum-exp
    m = maximum(log_alpha)
    if isfinite(m)
        alpha[1, :] .= exp.(log_alpha .- m) ./ sum(exp.(log_alpha .- m))
    else
        # all states have zero likelihood — fall back to uniform
        alpha[1, :] .= 1.0 / N
        log_alpha .= -log(N)
    end

    # recursion
    log_alpha_new = Vector{Float64}(undef, N)
    for t in 2:T_steps
        for k in 1:N
            # log-sum-exp: log(sum_j alpha[t-1,j] * T[j,k])
            max_val = -Inf
            for j in 1:N
                v = log_alpha[j] + log_T[j, k]
                if v > max_val
                    max_val = v
                end
            end
            s = 0.0
            for j in 1:N
                s += exp(log_alpha[j] + log_T[j, k] - max_val)
            end
            log_alpha_new[k] = max_val + log(s) + logpdf(dists[k], observations[t])
        end
        # normalize: store log-space for next iteration, probability-space in output
        m = maximum(log_alpha_new)
        if isfinite(m)
            log_alpha .= log_alpha_new
            alpha[t, :] .= exp.(log_alpha_new .- m) ./ sum(exp.(log_alpha_new .- m))
        else
            # all states have zero likelihood — fall back to uniform
            alpha[t, :] .= 1.0 / N
            log_alpha .= -log(N)
        end
    end

    return alpha
end

"""
    log_likelihood(model, observations) → Float64

Log-likelihood of an observation sequence under the model (forward algorithm).
Implemented in log-space with log-sum-exp for numerical stability.
"""
function log_likelihood(model::JumpHiddenMarkovModel,
                        observations::AbstractVector{Float64})
    isempty(observations) && throw(ArgumentError("observations must be non-empty"))
    N = model.partition.N
    T_steps = length(observations)
    ν = model.ν

    dists = [model.emissions[k].μ + model.emissions[k].σ * TDist(ν)
             for k in 1:N]
    log_T = log.(model.transition)

    log_alpha = Vector{Float64}(undef, N)
    ll = 0.0

    # initialization (log-space)
    for k in 1:N
        lp = model.stationary[k] > 0.0 ? log(model.stationary[k]) : -Inf
        log_alpha[k] = lp + logpdf(dists[k], observations[1])
    end
    # normalize and accumulate log-likelihood
    m = maximum(log_alpha)
    if isfinite(m)
        log_Z = m + log(sum(exp.(log_alpha .- m)))
        ll += log_Z
        log_alpha .-= log_Z
    else
        # all states have zero likelihood — return -Inf immediately
        return -Inf
    end

    # recursion
    log_alpha_new = Vector{Float64}(undef, N)
    for t in 2:T_steps
        for k in 1:N
            # log-sum-exp: log(sum_j alpha[j] * T[j,k])
            max_val = -Inf
            for j in 1:N
                v = log_alpha[j] + log_T[j, k]
                if v > max_val
                    max_val = v
                end
            end
            s = 0.0
            for j in 1:N
                s += exp(log_alpha[j] + log_T[j, k] - max_val)
            end
            log_alpha_new[k] = max_val + log(s) + logpdf(dists[k], observations[t])
        end
        # normalize and accumulate
        m = maximum(log_alpha_new)
        if isfinite(m)
            log_Z = m + log(sum(exp.(log_alpha_new .- m)))
            ll += log_Z
            log_alpha .= log_alpha_new .- log_Z
        else
            return -Inf
        end
    end

    return ll
end
