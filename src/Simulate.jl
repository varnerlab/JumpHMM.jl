"""
    simulate(model::MyJumpHiddenMarkovModel, number_of_steps::Int64; number_of_paths::Int64 = 1000,
        start::Union{Int64,Nothing} = nothing, seed::Union{Int64,Nothing} = nothing) -> NamedTuple

Simulate synthetic growth rate paths from a jump hidden Markov model. Day 1 of each path starts in a state and emits a growth rate.
Every later day is an ordinary day or a jump day. On an ordinary day, the model draws the next state from the transition matrix. On a jump day, the model
jumps into one of the tail states. A jump episode starts with probability `ϵ` on a day and lasts a Poisson(`λ`) number of days.
An episode length of zero means that day is an ordinary day. An episode that would run past the last day stops at the last day.

### Arguments
- `model::MyJumpHiddenMarkovModel`: The model to simulate from
- `number_of_steps::Int64`: The number of days in each path. Must be at least `1`

### Returns
- `NamedTuple`: Three matrices with one row for each day and one column for each path.
    - `growth_rates::Array{Float64,2}`: The simulated growth rates
    - `states::Array{Int64,2}`: The state on each day
    - `episodes::Array{Bool,2}`: `true` if the day is part of a jump episode

### Optional Arguments
- `number_of_paths::Int64`: The number of paths to simulate. Default: `1000`
- `start::Union{Int64,Nothing}`: The state of day 1. If `nothing`, the start state is drawn from the stationary distribution. Default: `nothing`
- `seed::Union{Int64,Nothing}`: The seed of the global random number generator. If `nothing`, the generator is not reseeded. Default: `nothing`

### Example
```julia
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 100, ν = 5.0, ϵ = 1e-3, λ = 90.0));
paths = JumpHMM.simulate(model, 252; number_of_paths = 100, seed = 5660);
```
"""
function simulate(model::MyJumpHiddenMarkovModel, number_of_steps::Int64; number_of_paths::Int64 = 1000,
    start::Union{Int64,Nothing} = nothing, seed::Union{Int64,Nothing} = nothing)::NamedTuple

    # check the number of days -
    if (number_of_steps < 1)
        throw(ArgumentError("number_of_steps must be at least 1, got $(number_of_steps)"));
    end

    # seed the global random number generator, if we were given a seed -
    if (isnothing(seed) == false)
        Random.seed!(seed);
    end

    # initialize -
    number_of_states = model.number_of_states;
    ϵ = model.ϵ;
    p_neg = model.p_neg;
    N_tail = model.N_tail;
    jump_length = Poisson(model.λ);
    growth_rates = Array{Float64,2}(undef, number_of_steps, number_of_paths);
    states = Array{Int64,2}(undef, number_of_steps, number_of_paths);
    episodes = Array{Bool,2}(undef, number_of_steps, number_of_paths);

    # row_samplers[s] draws the next state from row s of the transition matrix, and start_sampler draws a start state -
    row_samplers = [Categorical(model.transition[s, :]) for s ∈ 1:number_of_states];
    start_sampler = Categorical(model.stationary);

    # the states a jump episode can reach: the lowest N_tail states and the highest N_tail states -
    bottom_states = collect(1:min(N_tail, number_of_states));
    top_states = collect(max(1, number_of_states - N_tail + 1):number_of_states);

    # main simulation loop -
    for p ∈ 1:number_of_paths

        # day 1: use the given start state, or draw one from the stationary distribution, then emit -
        s = 0;
        if (isnothing(start) == true)
            s = rand(start_sampler);
        else
            s = start;
        end
        states[1, p] = s;
        growth_rates[1, p] = sample_emission(model, s);
        episodes[1, p] = false;

        # days 2, 3, ... until the path is full -
        t = 2;
        while (t <= number_of_steps)

            # does a jump episode start today? A model with ϵ = 0 never checks, so it draws no random number here -
            episode_starts = false;
            if (ϵ > 0.0)
                if (rand() < ϵ)
                    episode_starts = true;
                end
            end

            if (episode_starts == true)

                # draw how many days the episode lasts -
                K = rand(jump_length);
                if (K == 0)

                    # the episode has zero days, so today is an ordinary day: move using the transition matrix -
                    s = rand(row_samplers[s]);
                    states[t, p] = s;
                    growth_rates[t, p] = sample_emission(model, s);
                    episodes[t, p] = false;
                    t += 1;
                else

                    # each day of the episode jumps into the bottom tail (a crash) or the top tail (a rally) -
                    for _ ∈ 1:K

                        # stop if the path is full, even if the episode is not over -
                        if (t > number_of_steps)
                            break;
                        end

                        if (rand() < p_neg)
                            s = rand(bottom_states);
                        else
                            s = rand(top_states);
                        end
                        states[t, p] = s;
                        growth_rates[t, p] = sample_emission(model, s);
                        episodes[t, p] = true;
                        t += 1;
                    end
                end
            else

                # ordinary day: move using the transition matrix -
                s = rand(row_samplers[s]);
                states[t, p] = s;
                growth_rates[t, p] = sample_emission(model, s);
                episodes[t, p] = false;
                t += 1;
            end
        end
    end

    # return -
    return (growth_rates = growth_rates, states = states, episodes = episodes);
end
