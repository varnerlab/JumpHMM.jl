"""
    score_candidate(paths::NamedTuple, acf_observed::Array{Float64,1}, κ_observed::Float64,
        lags::Array{Int64,1}, w_κ::Float64) -> Tuple{Float64,Int64}

Score the simulated paths of one candidate (ϵ, λ) pair against the data. Only paths with at least one jump episode day are scored.
The score of a path is the sum of the squared differences between the autocorrelation of the absolute growth rates in the data
and in the path, plus `w_κ` times the squared difference between the excess kurtosis in the data and in the path.

### Arguments
- `paths::NamedTuple`: The output of `simulate`, with the matrices `growth_rates` and `episodes`
- `acf_observed::Array{Float64,1}`: The autocorrelation of the absolute growth rates in the data, at each lag in `lags`
- `κ_observed::Float64`: The kurtosis of the growth rates in the data
- `lags::Array{Int64,1}`: The lags of the autocorrelation
- `w_κ::Float64`: The weight of the kurtosis error in the score

### Returns
- `Tuple{Float64,Int64}`: The sum of the scores of the scored paths, and the number of paths that were scored
"""
function score_candidate(paths::NamedTuple, acf_observed::Array{Float64,1}, κ_observed::Float64,
    lags::Array{Int64,1}, w_κ::Float64)::Tuple{Float64,Int64}

    # initialize -
    number_of_paths = size(paths.growth_rates, 2);
    total_score = 0.0;
    number_scored = 0;

    # main loop - score each path that has at least one jump episode day -
    for p ∈ 1:number_of_paths
        if any(paths.episodes[:, p])

            # the growth rates of this path -
            g = paths.growth_rates[:, p];

            # the autocorrelation of the absolute growth rates, and the kurtosis, of this path -
            acf_simulated = autocor(abs.(g), lags);
            κ_simulated = kurtosis(g);

            # the squared error at each lag -
            squared_errors = Array{Float64,1}(undef, length(lags));
            for i ∈ eachindex(lags)
                squared_errors[i] = (acf_observed[i] - acf_simulated[i])^2;
            end

            # the score of this path -
            acf_error = sum(squared_errors);
            κ_error = w_κ * (κ_observed - κ_simulated)^2;
            total_score += acf_error + κ_error;
            number_scored += 1;
        end
    end

    # return -
    return (total_score, number_scored);
end

"""
    tune(model::MyJumpHiddenMarkovModel, growth_rates::Array{Float64,1}; ϵ_range = range(1e-4, 2.5e-2, length = 20),
        λ_range = range(10.0, 160.0, length = 16), number_of_paths::Int64 = 200, number_of_steps::Int64 = 0,
        w_κ::Float64 = 0.20, p_neg::Float64 = 0.52, N_tail::Int64 = 5, acf_lags::Int64 = 25,
        seed::Union{Int64,Nothing} = nothing) -> MyJumpHiddenMarkovModel

Search a grid of jump settings (`ϵ`, `λ`) for the pair whose simulated paths best match the data. For each pair, we simulate paths and
compare them with the data on two things: the autocorrelation of the absolute growth rates, and the excess kurtosis of the growth rates.
The pair with the lowest mean score wins. The input model is not changed. If no pair produces a path with a jump episode, we warn
and return the first pair of the grid.

### Arguments
- `model::MyJumpHiddenMarkovModel`: The model to tune. We copy it, and we do not change it
- `growth_rates::Array{Float64,1}`: The growth rates of the data, for example from `excess_growth_rates`

### Returns
- `MyJumpHiddenMarkovModel`: A copy of `model` with the best `ϵ` and `λ`, and with `p_neg` and `N_tail` set to the arguments

### Optional Arguments
- `ϵ_range`: The values of `ϵ`, the chance per day that a jump episode starts. Default: `range(1e-4, 2.5e-2, length = 20)`
- `λ_range`: The values of `λ`, the rate of the Poisson distribution of the length of a jump episode, in trading days. Default: `range(10.0, 160.0, length = 16)`
- `number_of_paths::Int64`: The number of paths to simulate for each pair. Default: `200`
- `number_of_steps::Int64`: The number of days in each path. If `0`, we use the number of days in `growth_rates`. Default: `0`
- `w_κ::Float64`: The weight of the kurtosis error in the score. Default: `0.20`
- `p_neg::Float64`: The probability that a day of a jump episode is in the negative tail. Default: `0.52`
- `N_tail::Int64`: The number of states in each tail that a jump episode can reach. Default: `5`
- `acf_lags::Int64`: The number of lags of the autocorrelation. We shorten it if the paths or the data are too short. Default: `25`
- `seed::Union{Int64,Nothing}`: The seed of the global random number generator. If `nothing`, the generator is not reseeded. Default: `nothing`

### Example
```julia
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 100, ν = 5.0));
tuned = JumpHMM.tune(model, G; ϵ_range = range(1e-4, 5e-3, length = 3), λ_range = range(10.0, 160.0, length = 3),
    number_of_paths = 20, seed = 5660);
```
"""
function tune(model::MyJumpHiddenMarkovModel, growth_rates::Array{Float64,1}; ϵ_range = range(1e-4, 2.5e-2, length = 20),
    λ_range = range(10.0, 160.0, length = 16), number_of_paths::Int64 = 200, number_of_steps::Int64 = 0,
    w_κ::Float64 = 0.20, p_neg::Float64 = 0.52, N_tail::Int64 = 5, acf_lags::Int64 = 25,
    seed::Union{Int64,Nothing} = nothing)::MyJumpHiddenMarkovModel

    # seed the global random number generator, if we were given a seed -
    if (isnothing(seed) == false)
        Random.seed!(seed);
    end

    # simulate as many days as the data has, unless we were told otherwise -
    if (number_of_steps == 0)
        number_of_steps = length(growth_rates);
    end

    # the autocorrelation needs fewer lags than there are days -
    acf_lags = min(acf_lags, min(length(growth_rates), number_of_steps) - 1);
    lags = collect(1:acf_lags);

    # what the data looks like: autocorrelation of the absolute growth rates, and kurtosis -
    acf_observed = autocor(abs.(growth_rates), lags);
    κ_observed = kurtosis(growth_rates);

    # initialize: until a pair scores, the best pair is the first pair of the grid -
    best_score = Inf;
    best_ϵ = first(ϵ_range);
    best_λ = first(λ_range);

    # main loop - try every pair, ϵ on the outside and λ on the inside -
    for ϵ_candidate ∈ ϵ_range
        for λ_candidate ∈ λ_range

            # a copy of the model with this pair -
            candidate = deepcopy(model);
            candidate.ϵ = Float64(ϵ_candidate);
            candidate.λ = Float64(λ_candidate);
            candidate.p_neg = p_neg;
            candidate.N_tail = N_tail;

            # simulate paths from the candidate, and score the paths that have a jump episode -
            paths = simulate(candidate, number_of_steps; number_of_paths = number_of_paths);
            (total_score, number_scored) = score_candidate(paths, acf_observed, κ_observed, lags, w_κ);

            # keep this pair if its mean score is the lowest so far -
            if (number_scored > 0)
                mean_score = total_score / number_scored;
                if (mean_score < best_score)
                    best_score = mean_score;
                    best_ϵ = Float64(ϵ_candidate);
                    best_λ = Float64(λ_candidate);
                end
            end
        end
    end

    # warn if no pair produced a path with a jump episode -
    if (isfinite(best_score) == false)
        @warn "tune: no candidate (ϵ, λ) produced any simulated paths with jumps. " *
              "Returning model with (ϵ=$(best_ϵ), λ=$(best_λ)) but this was not " *
              "data-driven. Consider increasing number_of_paths, number_of_steps, or ϵ_range."
    end

    # return a copy of the model with the best pair -
    tuned = deepcopy(model);
    tuned.ϵ = best_ϵ;
    tuned.λ = best_λ;
    tuned.p_neg = p_neg;
    tuned.N_tail = N_tail;
    return tuned;
end
