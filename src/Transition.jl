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
    stationary_distribution(T) → Vector{Float64}

Compute the stationary distribution by solving πT = π, i.e. the left eigenvector
of T associated with eigenvalue 1, normalized so that sum(π) = 1.
"""
function stationary_distribution(T::Matrix{Float64}; power::Int=0)
    N = size(T, 1)
    # Solve (T' - I) π = 0 with the constraint sum(π) = 1.
    # Replace last row of (T' - I) with the normalization constraint.
    A = transpose(T) - I
    A[N, :] .= 1.0
    b = zeros(N)
    b[N] = 1.0
    π̄ = A \ b
    # clamp any tiny negative values from numerical noise
    π̄ .= max.(π̄, 0.0)
    π̄ ./= sum(π̄)
    return π̄
end
