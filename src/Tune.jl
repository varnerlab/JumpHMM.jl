"""
    tune(model, prices; kwargs...) → JumpHiddenMarkovModel

Grid search over (ϵ, λ) to minimize a composite objective balancing
distributional fidelity (kurtosis) and temporal fidelity (ACF of |returns|).

Returns a NEW model with optimized JumpParameters.
"""
function tune(model::JumpHiddenMarkovModel, prices::AbstractVector{<:Real};
              ϵ_range=range(1e-4, 2.5e-2, length=20),
              λ_range=range(10.0, 160.0, length=16),
              n_paths::Int=200,
              n_steps::Int=0,
              w_κ::Float64=0.20,
              p_neg::Float64=0.52,
              N_tail::Int=5,
              acf_lags::Int=25,
              seed::Union{Int,Nothing}=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    # compute empirical targets
    G_emp = excess_growth_rates(prices; rf=model.rf, dt=model.dt)
    if n_steps == 0
        n_steps = length(G_emp)
    end
    lags = collect(1:acf_lags)
    acf_obs = autocor(abs.(G_emp), lags)
    κ_obs = kurtosis(G_emp)

    best_J = Inf
    best_ϵ = first(ϵ_range)
    best_λ = first(λ_range)

    for ϵ_candidate in ϵ_range
        for λ_candidate in λ_range

            jump_cand = JumpParameters(Float64(ϵ_candidate), Float64(λ_candidate);
                                       p_neg=p_neg, N_tail=N_tail)
            candidate = JumpHiddenMarkovModel(
                model.partition, model.transition, model.emissions,
                model.stationary, jump_cand, model.ν, model.rf, model.dt
            )

            result = simulate(candidate, n_steps; n_paths=n_paths)

            # accumulate objective over paths with at least one jump
            J_accum = 0.0
            n_valid = 0
            for path in result.paths
                any(path.jumps) || continue
                acf_sim = autocor(abs.(path.observations), lags)
                κ_sim = kurtosis(path.observations)
                acf_err = sum((acf_obs .- acf_sim).^2)
                κ_err = w_κ * (κ_obs - κ_sim)^2
                J_accum += acf_err + κ_err
                n_valid += 1
            end

            if n_valid > 0
                J_mean = J_accum / n_valid
                if J_mean < best_J
                    best_J = J_mean
                    best_ϵ = Float64(ϵ_candidate)
                    best_λ = Float64(λ_candidate)
                end
            end
        end
    end

    best_jump = JumpParameters(best_ϵ, best_λ; p_neg=p_neg, N_tail=N_tail)
    return JumpHiddenMarkovModel(
        model.partition, model.transition, model.emissions,
        model.stationary, best_jump, model.ν, model.rf, model.dt
    )
end
