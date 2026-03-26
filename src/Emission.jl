"""
    fit_emissions(states, observations, N; ν=5.0, min_obs=2) → Vector{StudentTEmission}

Fit per-state Student-t emission distributions. States with fewer than `min_obs`
observations (or σ < 1e-12) fall back to global mean/std.
"""
function fit_emissions(states::AbstractVector{Int},
                       observations::AbstractVector{Float64}, N::Int;
                       ν::Float64=5.0, min_obs::Int=2)

    μ_global = mean(observations)
    σ_global = std(observations)
    emissions = Vector{StudentTEmission}(undef, N)

    for k in 1:N
        idxs = findall(==(k), states)
        n_obs = length(idxs)
        if n_obs ≥ min_obs
            μ_k = mean(observations[idxs])
            σ_k = std(observations[idxs])
            if σ_k < 1e-12
                emissions[k] = StudentTEmission(μ_k, σ_global, ν, n_obs, true)
            else
                emissions[k] = StudentTEmission(μ_k, σ_k, ν, n_obs, false)
            end
        else
            emissions[k] = StudentTEmission(μ_global, σ_global, ν, n_obs, true)
        end
    end

    return emissions
end

"""
    sample_emission(e::StudentTEmission) → Float64

Draw a single sample: x = μ + σ × rand(TDist(ν)).
"""
function sample_emission(e::StudentTEmission)
    return e.μ + e.σ * rand(TDist(e.ν))
end
