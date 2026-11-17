# --- Estimation steps for a single-asset model ------------------------------------------------------------------------- #

"""
    laplace_boundaries(growth_rates::Array{Float64,1}, number_of_states::Int64) -> NamedTuple

Fit a Laplace distribution to the growth rates, and use its quantiles to cut the growth rates into
`number_of_states` states that are equally likely under the fitted distribution.

### Arguments
- `growth_rates::Array{Float64,1}`: The growth rates. They must not all be equal
- `number_of_states::Int64`: The number of states. It must be at least `1`

### Returns
- `NamedTuple`: The location `μ`, the scale `b`, and the `number_of_states + 1` state `boundaries`. The first boundary is `-Inf` and the last boundary is `Inf`

### Example
```julia
growth_rates = JumpHMM.excess_growth_rates(prices);
partition = JumpHMM.laplace_boundaries(growth_rates, 100);
```
"""
function laplace_boundaries(growth_rates::Array{Float64,1}, number_of_states::Int64)::NamedTuple

    # check: the growth rates must vary, otherwise there is no Laplace scale to fit -
    number_of_growth_rates = length(growth_rates);
    all_equal = true;
    for i ∈ 1:number_of_growth_rates
        if growth_rates[i] != growth_rates[1]
            all_equal = false;
        end
    end
    if number_of_growth_rates < 2 || all_equal == true
        throw(ArgumentError("growth_rates must have at least two different values"));
    end

    # fit the Laplace distribution by maximum likelihood -
    d = fit_mle(Laplace, growth_rates);
    μ, b = params(d);

    # the probabilities that cut the distribution into equally likely states -
    cutoffs = collect(range(0.0, 1.0, length = number_of_states + 1));

    # the boundaries are the quantiles at these probabilities -
    laplace = Laplace(μ, b);
    boundaries = Array{Float64,1}(undef, number_of_states + 1);
    for i ∈ 1:(number_of_states + 1)
        boundaries[i] = quantile(laplace, cutoffs[i]);
    end

    # the outer states reach to infinity -
    boundaries[1] = -Inf;
    boundaries[end] = Inf;

    # return -
    return (μ = μ, b = b, boundaries = boundaries);
end

"""
    assign_states(boundaries::Array{Float64,1}, growth_rates::Array{Float64,1}) -> Array{Int64,1}

Give each growth rate the number of the state that contains it. State `k` holds the growth rates from
`boundaries[k]` up to, but not including, `boundaries[k+1]`. The last state also holds its upper boundary.

### Arguments
- `boundaries::Array{Float64,1}`: The state boundaries, with `number_of_states + 1` entries
- `growth_rates::Array{Float64,1}`: The growth rates

### Returns
- `Array{Int64,1}`: The state of each growth rate

### Example
```julia
states = JumpHMM.assign_states(boundaries, growth_rates);
```
"""
function assign_states(boundaries::Array{Float64,1}, growth_rates::Array{Float64,1})::Array{Int64,1}

    # initialize -
    number_of_states = length(boundaries) - 1;
    number_of_growth_rates = length(growth_rates);
    states = Array{Int64,1}(undef, number_of_growth_rates);

    # main loop -
    for i ∈ 1:number_of_growth_rates

        # the last boundary that is at or below this growth rate -
        k = searchsortedlast(boundaries, growth_rates[i]);

        # keep the state between 1 and number_of_states -
        states[i] = clamp(k, 1, number_of_states);
    end

    # return -
    return states;
end

"""
    estimate_transition(states::Array{Int64,1}, number_of_states::Int64) -> Array{Float64,2}

Estimate the transition matrix by counting how often each state is followed by each other state.
A state that is never followed by anything gets a row of equal probabilities.

### Arguments
- `states::Array{Int64,1}`: The state at each time step
- `number_of_states::Int64`: The number of states

### Returns
- `Array{Float64,2}`: The transition matrix. Row `i` holds the probabilities of moving from state `i` to each state

### Example
```julia
T = JumpHMM.estimate_transition(states, 100);
```
"""
function estimate_transition(states::Array{Int64,1}, number_of_states::Int64)::Array{Float64,2}

    # initialize -
    T = zeros(Float64, number_of_states, number_of_states);

    # count each move from one state to the next -
    for i ∈ 2:length(states)
        T[states[i-1], states[i]] += 1.0;
    end

    # turn the counts of each row into probabilities -
    for row ∈ 1:number_of_states
        total = sum(T[row, :]);
        if total > 0.0
            T[row, :] ./= total;
        else
            T[row, :] .= 1.0 / number_of_states;
        end
    end

    # return -
    return T;
end

"""
    stationary_distribution(T::Array{Float64,2}; power::Int64 = 0) -> Array{Float64,1}

Compute the stationary distribution `π` of a transition matrix, the distribution that satisfies `πT = π`.
We solve the linear equations of `πT = π` together with the condition that the entries of `π` add to one.

### Arguments
- `T::Array{Float64,2}`: The transition matrix
- `power::Int64`: Not used. It is kept so that code written for version 0.4.1 still runs

### Returns
- `Array{Float64,1}`: The stationary distribution

### Example
```julia
π̄ = JumpHMM.stationary_distribution(T);
```
"""
function stationary_distribution(T::Array{Float64,2}; power::Int64 = 0)::Array{Float64,1}

    # initialize -
    number_of_states = size(T, 1);

    # the equations πT = π, with the last equation replaced by "the entries of π add to one" -
    A = transpose(T) - I;
    A[number_of_states, :] .= 1.0;
    b = zeros(number_of_states);
    b[number_of_states] = 1.0;

    # solve for π -
    π̄ = A \ b;

    # remove tiny negative values that come from rounding, then make the entries add to one -
    π̄ .= max.(π̄, 0.0);
    π̄ ./= sum(π̄);

    # return -
    return π̄;
end

"""
    estimate_emissions(states::Array{Int64,1}, growth_rates::Array{Float64,1}, number_of_states::Int64;
        ν::Float64 = 5.0, min_obs::Int64 = 2) -> NamedTuple

Estimate the Student-t emission distribution of each state from the growth rates in that state.
A state with fewer than `min_obs` growth rates, or with a standard deviation below `1e-12`, uses the
standard deviation of all growth rates. A state with fewer than `min_obs` growth rates also uses the mean of all growth rates.

### Arguments
- `states::Array{Int64,1}`: The state at each time step
- `growth_rates::Array{Float64,1}`: The growth rates
- `number_of_states::Int64`: The number of states
- `ν::Float64`: The degrees of freedom of the emission distributions. The default is `5.0`. It is not used in the estimates
- `min_obs::Int64`: The fewest growth rates a state needs to use its own estimates. The default is `2`

### Returns
- `NamedTuple`: The location `μ`, the scale `σ`, the number of growth rates `n`, and the flag `fallback` of each state

### Example
```julia
emissions = JumpHMM.estimate_emissions(states, growth_rates, 100);
```
"""
function estimate_emissions(states::Array{Int64,1}, growth_rates::Array{Float64,1}, number_of_states::Int64;
    ν::Float64 = 5.0, min_obs::Int64 = 2)::NamedTuple

    # initialize -
    μ_global = mean(growth_rates);
    σ_global = std(growth_rates);
    μ = Array{Float64,1}(undef, number_of_states);
    σ = Array{Float64,1}(undef, number_of_states);
    n = Array{Int64,1}(undef, number_of_states);
    fallback = Array{Bool,1}(undef, number_of_states);

    # main loop -
    for k ∈ 1:number_of_states

        # the growth rates that are in state k -
        indexes = findall(==(k), states);
        number_of_observations = length(indexes);
        n[k] = number_of_observations;

        if number_of_observations ≥ min_obs

            # enough growth rates: use the estimates of this state -
            μ_k = mean(growth_rates[indexes]);
            σ_k = std(growth_rates[indexes]);
            if σ_k < 1e-12

                # all growth rates of this state are equal: borrow the global scale -
                μ[k] = μ_k;
                σ[k] = σ_global;
                fallback[k] = true;
            else
                μ[k] = μ_k;
                σ[k] = σ_k;
                fallback[k] = false;
            end
        else

            # too few growth rates: use the global estimates -
            μ[k] = μ_global;
            σ[k] = σ_global;
            fallback[k] = true;
        end
    end

    # return -
    return (μ = μ, σ = σ, n = n, fallback = fallback);
end

"""
    sample_emission(model::MyJumpHiddenMarkovModel, k::Int64) -> Float64

Draw one growth rate from the emission distribution of state `k`: `μ_k + σ_k * t`, where `t` is drawn from a Student-t distribution with `ν` degrees of freedom.

### Arguments
- `model::MyJumpHiddenMarkovModel`: The model
- `k::Int64`: The state to draw from

### Returns
- `Float64`: The growth rate

### Example
```julia
g = JumpHMM.sample_emission(model, 50);
```
"""
function sample_emission(model::MyJumpHiddenMarkovModel, k::Int64)::Float64
    return model.emission_μ[k] + model.emission_σ[k] * rand(TDist(model.ν));
end
