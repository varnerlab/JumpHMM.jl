"""
    decode(model::MyJumpHiddenMarkovModel, growth_rates::Array{Float64,1}) -> Array{Int64,1}

Find the most likely sequence of states for a sequence of growth rates (the Viterbi algorithm). The calculation is done with logarithms.
Each state emits a growth rate from a location-scale Student-t distribution, built from the emission location, the emission scale, and the model's `ν`.
The jump settings `ϵ` and `λ` are not used when decoding.

### Arguments
- `model::MyJumpHiddenMarkovModel`: The model that describes the growth rates
- `growth_rates::Array{Float64,1}`: The growth rates, one for each day. Must not be empty

### Returns
- `Array{Int64,1}`: The most likely state on each day, a number from `1` to `number_of_states`

### Example
```julia
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 100, ν = 5.0, ϵ = 1e-3, λ = 90.0));
states = JumpHMM.decode(model, G);
```
"""
function decode(model::MyJumpHiddenMarkovModel, growth_rates::Array{Float64,1})::Array{Int64,1}

    # check the growth rates -
    if (isempty(growth_rates) == true)
        throw(ArgumentError("growth_rates must not be empty"));
    end

    # initialize -
    number_of_states = model.number_of_states;
    number_of_steps = length(growth_rates);
    ν = model.ν;
    log_transition = log.(model.transition);
    emissions = [model.emission_μ[k] + model.emission_σ[k] * TDist(ν) for k ∈ 1:number_of_states];
    delta = Array{Float64,2}(undef, number_of_steps, number_of_states);
    psi = Array{Int64,2}(undef, number_of_steps, number_of_states);

    # day 1: the log of the stationary probability plus the log of the emission density -
    for k ∈ 1:number_of_states
        log_start = -Inf;
        if (model.stationary[k] > 0.0)
            log_start = log(model.stationary[k]);
        end
        delta[1, k] = log_start + logpdf(emissions[k], growth_rates[1]);
        psi[1, k] = 0;
    end

    # every later day: for each state, keep the best previous state -
    for t ∈ 2:number_of_steps
        for k ∈ 1:number_of_states
            best_value = -Inf;
            best_state = 1;
            for j ∈ 1:number_of_states
                value = delta[t-1, j] + log_transition[j, k];
                if (value > best_value)
                    best_value = value;
                    best_state = j;
                end
            end
            delta[t, k] = best_value + logpdf(emissions[k], growth_rates[t]);
            psi[t, k] = best_state;
        end
    end

    # backtrack from the best state on the last day -
    states = Array{Int64,1}(undef, number_of_steps);
    states[number_of_steps] = argmax(delta[number_of_steps, :]);
    for t ∈ (number_of_steps-1):-1:1
        states[t] = psi[t+1, states[t+1]];
    end

    # return -
    return states;
end

"""
    forward_filter(model::MyJumpHiddenMarkovModel, growth_rates::Array{Float64,1}) -> Array{Float64,2}

Compute the probability of each state on each day, given the growth rates up to that day (the forward algorithm). The calculation is done with logarithms.
Each state emits a growth rate from a location-scale Student-t distribution, built from the emission location, the emission scale, and the model's `ν`.
The jump settings `ϵ` and `λ` are not used when filtering. If every state has zero likelihood on a day, that day gets the same probability for every state.

### Arguments
- `model::MyJumpHiddenMarkovModel`: The model that describes the growth rates
- `growth_rates::Array{Float64,1}`: The growth rates, one for each day. Must not be empty

### Returns
- `Array{Float64,2}`: The state probabilities, one row for each day and one column for each state. Each row sums to `1`

### Example
```julia
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 100, ν = 5.0, ϵ = 1e-3, λ = 90.0));
filtered = JumpHMM.forward_filter(model, G);
```
"""
function forward_filter(model::MyJumpHiddenMarkovModel, growth_rates::Array{Float64,1})::Array{Float64,2}

    # check the growth rates -
    if (isempty(growth_rates) == true)
        throw(ArgumentError("growth_rates must not be empty"));
    end

    # initialize -
    number_of_states = model.number_of_states;
    number_of_steps = length(growth_rates);
    ν = model.ν;
    emissions = [model.emission_μ[k] + model.emission_σ[k] * TDist(ν) for k ∈ 1:number_of_states];
    log_transition = log.(model.transition);
    alpha = Array{Float64,2}(undef, number_of_steps, number_of_states);
    log_alpha = Array{Float64,1}(undef, number_of_states);
    log_alpha_new = Array{Float64,1}(undef, number_of_states);

    # day 1: the log of the stationary probability plus the log of the emission density -
    for k ∈ 1:number_of_states
        log_start = -Inf;
        if (model.stationary[k] > 0.0)
            log_start = log(model.stationary[k]);
        end
        log_alpha[k] = log_start + logpdf(emissions[k], growth_rates[1]);
    end

    # turn the log values into probabilities: subtract the largest value, exponentiate, and divide by the sum -
    m = maximum(log_alpha);
    if (isfinite(m) == true)
        alpha[1, :] .= exp.(log_alpha .- m) ./ sum(exp.(log_alpha .- m));
    else

        # every state has zero likelihood, so give every state the same probability -
        alpha[1, :] .= 1.0 / number_of_states;
        log_alpha .= -log(number_of_states);
    end

    # every later day -
    for t ∈ 2:number_of_steps
        for k ∈ 1:number_of_states

            # the log of the sum over previous states j of alpha[j] * transition[j, k], using the largest term to keep the sum stable -
            max_value = -Inf;
            for j ∈ 1:number_of_states
                value = log_alpha[j] + log_transition[j, k];
                if (value > max_value)
                    max_value = value;
                end
            end
            total = 0.0;
            for j ∈ 1:number_of_states
                total += exp(log_alpha[j] + log_transition[j, k] - max_value);
            end
            log_alpha_new[k] = max_value + log(total) + logpdf(emissions[k], growth_rates[t]);
        end

        # keep the log values for the next day, and store the probabilities for this day -
        m = maximum(log_alpha_new);
        if (isfinite(m) == true)
            log_alpha .= log_alpha_new;
            alpha[t, :] .= exp.(log_alpha_new .- m) ./ sum(exp.(log_alpha_new .- m));
        else

            # every state has zero likelihood, so give every state the same probability -
            alpha[t, :] .= 1.0 / number_of_states;
            log_alpha .= -log(number_of_states);
        end
    end

    # return -
    return alpha;
end

"""
    log_likelihood(model::MyJumpHiddenMarkovModel, growth_rates::Array{Float64,1}) -> Float64

Compute the log-likelihood of a sequence of growth rates under the model (the forward algorithm). The calculation is done with logarithms.
Each state emits a growth rate from a location-scale Student-t distribution, built from the emission location, the emission scale, and the model's `ν`.
The jump settings `ϵ` and `λ` are not used. If every state has zero likelihood on a day, the log-likelihood is `-Inf`.

### Arguments
- `model::MyJumpHiddenMarkovModel`: The model that describes the growth rates
- `growth_rates::Array{Float64,1}`: The growth rates, one for each day. Must not be empty

### Returns
- `Float64`: The log-likelihood of the growth rates

### Example
```julia
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 100, ν = 5.0, ϵ = 1e-3, λ = 90.0));
ll = JumpHMM.log_likelihood(model, G);
```
"""
function log_likelihood(model::MyJumpHiddenMarkovModel, growth_rates::Array{Float64,1})::Float64

    # check the growth rates -
    if (isempty(growth_rates) == true)
        throw(ArgumentError("growth_rates must not be empty"));
    end

    # initialize -
    number_of_states = model.number_of_states;
    number_of_steps = length(growth_rates);
    ν = model.ν;
    emissions = [model.emission_μ[k] + model.emission_σ[k] * TDist(ν) for k ∈ 1:number_of_states];
    log_transition = log.(model.transition);
    log_alpha = Array{Float64,1}(undef, number_of_states);
    log_alpha_new = Array{Float64,1}(undef, number_of_states);
    ll = 0.0;

    # day 1: the log of the stationary probability plus the log of the emission density -
    for k ∈ 1:number_of_states
        log_start = -Inf;
        if (model.stationary[k] > 0.0)
            log_start = log(model.stationary[k]);
        end
        log_alpha[k] = log_start + logpdf(emissions[k], growth_rates[1]);
    end

    # add the log of the day's total to the log-likelihood, and divide it out of the state values -
    m = maximum(log_alpha);
    if (isfinite(m) == true)
        log_Z = m + log(sum(exp.(log_alpha .- m)));
        ll += log_Z;
        log_alpha .-= log_Z;
    else

        # every state has zero likelihood, so the growth rates are impossible under the model -
        return -Inf;
    end

    # every later day -
    for t ∈ 2:number_of_steps
        for k ∈ 1:number_of_states

            # the log of the sum over previous states j of alpha[j] * transition[j, k], using the largest term to keep the sum stable -
            max_value = -Inf;
            for j ∈ 1:number_of_states
                value = log_alpha[j] + log_transition[j, k];
                if (value > max_value)
                    max_value = value;
                end
            end
            total = 0.0;
            for j ∈ 1:number_of_states
                total += exp(log_alpha[j] + log_transition[j, k] - max_value);
            end
            log_alpha_new[k] = max_value + log(total) + logpdf(emissions[k], growth_rates[t]);
        end

        # add the log of the day's total to the log-likelihood, and divide it out of the state values -
        m = maximum(log_alpha_new);
        if (isfinite(m) == true)
            log_Z = m + log(sum(exp.(log_alpha_new .- m)));
            ll += log_Z;
            log_alpha .= log_alpha_new .- log_Z;
        else

            # every state has zero likelihood, so the growth rates are impossible under the model -
            return -Inf;
        end
    end

    # return -
    return ll;
end
