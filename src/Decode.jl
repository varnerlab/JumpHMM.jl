"""
    decode(model, observations) → Vector{Int}

Viterbi algorithm: find the most likely state sequence. Computed in log-space.
"""
function decode(model::JumpHiddenMarkovModel,
                observations::AbstractVector{Float64})
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
"""
function forward_filter(model::JumpHiddenMarkovModel,
                        observations::AbstractVector{Float64})
    N = model.partition.N
    T_steps = length(observations)
    ν = model.ν

    dists = [model.emissions[k].μ + model.emissions[k].σ * TDist(ν)
             for k in 1:N]

    alpha = Matrix{Float64}(undef, T_steps, N)

    # initialization
    for k in 1:N
        alpha[1, k] = model.stationary[k] * pdf(dists[k], observations[1])
    end
    Z = sum(alpha[1, :])
    alpha[1, :] ./= Z

    # recursion
    for t in 2:T_steps
        for k in 1:N
            pred = 0.0
            for j in 1:N
                pred += alpha[t-1, j] * model.transition[j, k]
            end
            alpha[t, k] = pred * pdf(dists[k], observations[t])
        end
        Z = sum(alpha[t, :])
        if Z > 0.0
            alpha[t, :] ./= Z
        end
    end

    return alpha
end

"""
    log_likelihood(model, observations) → Float64

Log-likelihood of an observation sequence under the model (forward algorithm).
"""
function log_likelihood(model::JumpHiddenMarkovModel,
                        observations::AbstractVector{Float64})
    N = model.partition.N
    T_steps = length(observations)
    ν = model.ν

    dists = [model.emissions[k].μ + model.emissions[k].σ * TDist(ν)
             for k in 1:N]

    alpha = Vector{Float64}(undef, N)
    ll = 0.0

    # initialization
    for k in 1:N
        alpha[k] = model.stationary[k] * pdf(dists[k], observations[1])
    end
    Z = sum(alpha)
    ll += log(Z)
    alpha ./= Z

    # recursion
    alpha_new = Vector{Float64}(undef, N)
    for t in 2:T_steps
        for k in 1:N
            pred = 0.0
            for j in 1:N
                pred += alpha[j] * model.transition[j, k]
            end
            alpha_new[k] = pred * pdf(dists[k], observations[t])
        end
        Z = sum(alpha_new)
        ll += log(Z)
        alpha_new ./= Z
        alpha .= alpha_new
    end

    return ll
end
