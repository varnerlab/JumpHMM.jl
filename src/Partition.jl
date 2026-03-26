"""
    fit(::Type{LaplacePartition}, observations; N=100) → LaplacePartition

Fit a Laplace distribution via MLE, then construct N equal-probability quantile bins.
"""
function fit(::Type{LaplacePartition}, observations::AbstractVector{Float64};
             N::Int=100)
    if length(observations) < 2 || all(x -> x == observations[1], observations)
        throw(ArgumentError("Cannot fit LaplacePartition: observations must have nonzero variance (got a constant series)."))
    end
    d = fit_mle(Laplace, observations)
    return LaplacePartition(params(d)..., N)
end

"""
    assign_states(partition, observations) → Vector{Int}

Map continuous observations to discrete state indices.

Convention: boundaries[k] ≤ x < boundaries[k+1] for k = 1..N-1.
Last state (k = N) is inclusive on the upper bound.
"""
function assign_states(partition::LaplacePartition,
                       observations::AbstractVector{Float64})
    boundaries = partition.boundaries
    N = partition.N
    states = Vector{Int}(undef, length(observations))
    for i in eachindex(observations)
        # searchsortedlast returns k such that boundaries[k] ≤ x
        k = searchsortedlast(boundaries, observations[i])
        states[i] = clamp(k, 1, N)
    end
    return states
end
