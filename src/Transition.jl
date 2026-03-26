"""
    estimate_transition(states, N) → Matrix{Float64}

Estimate an N×N row-stochastic transition matrix by frequency counting.
Rows with zero counts are filled with uniform 1/N.
"""
function estimate_transition(states::AbstractVector{Int}, N::Int)
    P = zeros(Float64, N, N)
    for i in 2:length(states)
        P[states[i-1], states[i]] += 1.0
    end
    for row in 1:N
        Z = sum(P[row, :])
        if Z > 0.0
            P[row, :] ./= Z
        else
            P[row, :] .= 1.0 / N
        end
    end
    return P
end

"""
    stationary_distribution(T; power=50) → Vector{Float64}

Compute the stationary distribution via matrix exponentiation: π̄ = (T^power)[1, :].
"""
function stationary_distribution(T::Matrix{Float64}; power::Int=50)
    π̄ = (T^power)[1, :]
    π̄ ./= sum(π̄)  # ensure normalization
    return π̄
end
