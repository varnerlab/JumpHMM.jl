"""
    build(model::Type{MyJumpHiddenMarkovModel}, data::NamedTuple) -> MyJumpHiddenMarkovModel

Estimate a hidden Markov model of the growth rates of one asset. We cut the growth rates into states,
count the moves between states, estimate the emission distribution of each state, and solve for the stationary distribution.

### Arguments
- `model::Type{MyJumpHiddenMarkovModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `growth_rates::Array{Float64,1}`: The growth rates, at least two finite values that are not all equal
    - `number_of_states::Int64`: The number of states, at least `1`
    - `ν::Float64`: The degrees of freedom of the Student-t emission distributions

### Returns
- `MyJumpHiddenMarkovModel`: The estimated model

### Optional Arguments
The optional keys of `data` and their defaults are:
- `ϵ::Float64`: The probability that a jump episode starts at a time step, between `0` and `1`. The default is `0.0`, which means no jumps
- `λ::Float64`: The rate of the Poisson distribution of the length of a jump episode, above `0`. The default is `1.0`
- `p_neg::Float64`: The probability that a jump episode moves the asset into the negative tail, between `0` and `1`. The default is `0.52`
- `N_tail::Int64`: The number of states in each tail that a jump episode can reach, at least `1`. The default is `5`
- `Δt::Float64`: The time step of the growth rates, in years. The default is `1/252`
- `min_obs::Int64`: The fewest growth rates a state needs to use its own emission estimates. The default is `2`

### Example
```julia
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (
    growth_rates = growth_rates,
    number_of_states = 100,
    ν = 5.0
));
```
"""
function build(model::Type{MyJumpHiddenMarkovModel}, data::NamedTuple)::MyJumpHiddenMarkovModel

    # check: the required keys are present -
    if haskey(data, :growth_rates) == false
        throw(ArgumentError("growth_rates is required"));
    end
    if haskey(data, :number_of_states) == false
        throw(ArgumentError("number_of_states is required"));
    end
    if haskey(data, :ν) == false
        throw(ArgumentError("ν is required"));
    end
    growth_rates = data.growth_rates;
    number_of_states = data.number_of_states;
    ν = data.ν;

    # set the optional keys to their defaults, then replace the ones that were given -
    ϵ = 0.0;
    λ = 1.0;
    p_neg = 0.52;
    N_tail = 5;
    Δt = 1/252;
    min_obs = 2;
    if haskey(data, :ϵ) == true
        ϵ = data.ϵ;
    end
    if haskey(data, :λ) == true
        λ = data.λ;
    end
    if haskey(data, :p_neg) == true
        p_neg = data.p_neg;
    end
    if haskey(data, :N_tail) == true
        N_tail = data.N_tail;
    end
    if haskey(data, :Δt) == true
        Δt = data.Δt;
    end
    if haskey(data, :min_obs) == true
        min_obs = data.min_obs;
    end

    # check: number_of_states is an integer, and the values are in range -
    if isa(number_of_states, Int64) == false
        throw(ArgumentError("number_of_states must be an Int64, got $(number_of_states)"));
    end
    if number_of_states < 1
        throw(ArgumentError("number_of_states must be at least 1, got $(number_of_states)"));
    end
    if ϵ < 0.0 || ϵ > 1.0
        throw(ArgumentError("ϵ must be between 0 and 1, got $(ϵ)"));
    end
    if λ <= 0.0
        throw(ArgumentError("λ must be greater than 0, got $(λ)"));
    end
    if p_neg < 0.0 || p_neg > 1.0
        throw(ArgumentError("p_neg must be between 0 and 1, got $(p_neg)"));
    end
    if N_tail < 1
        throw(ArgumentError("N_tail must be at least 1, got $(N_tail)"));
    end
    number_of_finite = 0;
    for i ∈ eachindex(growth_rates)
        if isfinite(growth_rates[i]) == true
            number_of_finite += 1;
        end
    end
    if number_of_finite < 2 || number_of_finite != length(growth_rates)
        throw(ArgumentError("growth_rates must hold at least two values, and every value must be finite"));
    end

    # cut the growth rates into states -
    partition = laplace_boundaries(growth_rates, number_of_states);
    states = assign_states(partition.boundaries, growth_rates);

    # count the moves between states -
    T = estimate_transition(states, number_of_states);

    # estimate the emission distribution of each state -
    emissions = estimate_emissions(states, growth_rates, number_of_states; ν = ν, min_obs = min_obs);

    # solve for the stationary distribution -
    π̄ = stationary_distribution(T);

    # build an empty model, then fill in the fields -
    m = MyJumpHiddenMarkovModel();
    m.number_of_states = number_of_states;
    m.laplace_μ = partition.μ;
    m.laplace_b = partition.b;
    m.boundaries = partition.boundaries;
    m.transition = T;
    m.stationary = π̄;
    m.emission_μ = emissions.μ;
    m.emission_σ = emissions.σ;
    m.emission_n = emissions.n;
    m.emission_fallback = emissions.fallback;
    m.ν = ν;
    m.ϵ = ϵ;
    m.λ = λ;
    m.p_neg = p_neg;
    m.N_tail = N_tail;
    m.Δt = Δt;

    # return -
    return m;
end
