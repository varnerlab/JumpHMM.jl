"""
    simulate(model, n_steps; n_paths=1000, start=:stationary, seed=nothing) → SimulationResult

Generate synthetic paths from a fitted Jump-HMM.
"""
function simulate(model::JumpHiddenMarkovModel, n_steps::Int;
                  n_paths::Int=1000,
                  start::Union{Int,Symbol}=:stationary,
                  seed::Union{Int,Nothing}=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    N = model.partition.N
    ϵ = model.jump.ϵ
    λ = model.jump.λ
    p_neg = model.jump.p_neg
    N_tail = model.jump.N_tail
    jump_dist = Poisson(λ)

    # pre-build Categorical distributions for each row of the transition matrix
    row_cats = [Categorical(model.transition[s, :]) for s in 1:N]

    # define bottom/top tail state ranges
    bottom_states = collect(1:min(N_tail, N))
    top_states = collect(max(1, N - N_tail + 1):N)

    # stationary distribution for starting state sampling
    π_cat = Categorical(model.stationary)

    paths = Vector{SimulationPath}(undef, n_paths)
    for p in 1:n_paths
        states_out = Vector{Int}(undef, n_steps)
        obs_out = Vector{Float64}(undef, n_steps)
        jumps_out = Vector{Bool}(undef, n_steps)

        # pick starting state
        s = (start === :stationary) ? rand(π_cat) : start

        t = 1
        while t ≤ n_steps
            if ϵ > 0.0 && rand() < ϵ
                # jump event
                K = rand(jump_dist)
                if K == 0
                    # Poisson returned 0: normal transition instead
                    s = rand(row_cats[s])
                    states_out[t] = s
                    obs_out[t] = sample_emission(model.emissions[s])
                    jumps_out[t] = false
                    t += 1
                else
                    for _ in 1:K
                        t > n_steps && break
                        if rand() < p_neg
                            s = rand(bottom_states)
                        else
                            s = rand(top_states)
                        end
                        states_out[t] = s
                        obs_out[t] = sample_emission(model.emissions[s])
                        jumps_out[t] = true
                        t += 1
                    end
                end
            else
                # normal Markov transition
                s = rand(row_cats[s])
                states_out[t] = s
                obs_out[t] = sample_emission(model.emissions[s])
                jumps_out[t] = false
                t += 1
            end
        end

        paths[p] = SimulationPath(states_out, obs_out, jumps_out)
    end

    return SimulationResult(paths)
end
