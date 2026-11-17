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
- `λ::Float64`: The mean of the Poisson episode length, in trading days, above `0`. The default is `1.0`
- `p_neg::Float64`: The probability that a day of a jump episode is in the negative tail, between `0` and `1`. The default is `0.52`
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

"""
    build(model::Type{MyGaussianCopulaModel}, data::NamedTuple) -> MyGaussianCopulaModel

Fit a Gaussian copula to the growth rates of several assets. We replace each column of growth rates by its ranks divided by the
number of rows plus one, and take the correlation matrix of these uniform numbers.

### Arguments
- `model::Type{MyGaussianCopulaModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `growth_rates::Array{Float64,2}`: The growth rates, one row per day and one column per asset. We need at least two columns, at least two rows, and every value must be finite

### Returns
- `MyGaussianCopulaModel`: The fitted copula

### Example
```julia
copula = JumpHMM.build(JumpHMM.MyGaussianCopulaModel, (growth_rates = R,));
```
"""
function build(model::Type{MyGaussianCopulaModel}, data::NamedTuple)::MyGaussianCopulaModel

    # check: the required key is present, and the matrix is usable -
    if haskey(data, :growth_rates) == false
        throw(ArgumentError("growth_rates is required"));
    end
    growth_rates = data.growth_rates;
    _check_copula_growth_rates(growth_rates);

    # estimate the correlation matrix of the uniform numbers -
    U = _uniform_ranks(growth_rates);
    Σ = _valid_correlation(cor(U));

    # build an empty model, then fill in the fields -
    m = MyGaussianCopulaModel();
    m.Σ = Σ;

    # return -
    return m;
end

"""
    build(model::Type{MyStudentTCopulaModel}, data::NamedTuple) -> MyStudentTCopulaModel

Fit a Student-t copula to the growth rates of several assets. We estimate the correlation matrix as for the Gaussian copula.
Then we pick the degrees of freedom from `3, 4, 5, 6, 7, 8, 10, 15, 20, 30` as the value with the largest log-likelihood.

### Arguments
- `model::Type{MyStudentTCopulaModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `growth_rates::Array{Float64,2}`: The growth rates, one row per day and one column per asset. We need at least two columns, at least two rows, and every value must be finite

### Returns
- `MyStudentTCopulaModel`: The fitted copula

### Example
```julia
copula = JumpHMM.build(JumpHMM.MyStudentTCopulaModel, (growth_rates = R,));
```
"""
function build(model::Type{MyStudentTCopulaModel}, data::NamedTuple)::MyStudentTCopulaModel

    # check: the required key is present, and the matrix is usable -
    if haskey(data, :growth_rates) == false
        throw(ArgumentError("growth_rates is required"));
    end
    growth_rates = data.growth_rates;
    _check_copula_growth_rates(growth_rates);

    # estimate the correlation matrix of the uniform numbers -
    U = _uniform_ranks(growth_rates);
    Σ = _valid_correlation(cor(U));

    # pick the degrees of freedom with the largest log-likelihood -
    best_ll = -Inf;
    best_ν = 5.0;
    for ν_candidate ∈ [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 30.0]
        ll = _studentt_copula_loglik(U, Σ, ν_candidate);
        if ll > best_ll
            best_ll = ll;
            best_ν = ν_candidate;
        end
    end

    # build an empty model, then fill in the fields -
    m = MyStudentTCopulaModel();
    m.Σ = Σ;
    m.ν = best_ν;

    # return -
    return m;
end

"""
    _check_copula_growth_rates(growth_rates::Array{Float64,2})

Throw an `ArgumentError` if the growth rates cannot be used to fit a copula: fewer than two columns, fewer than two rows,
or a value that is not finite.

### Arguments
- `growth_rates::Array{Float64,2}`: The growth rates, one row per day and one column per asset
"""
function _check_copula_growth_rates(growth_rates::Array{Float64,2})

    # check: at least two assets and two days -
    if size(growth_rates, 2) < 2
        throw(ArgumentError("growth_rates must have at least two columns (assets)"));
    end
    if size(growth_rates, 1) < 2
        throw(ArgumentError("growth_rates must have at least two rows (days)"));
    end

    # check: every value is finite -
    for i ∈ eachindex(growth_rates)
        if isfinite(growth_rates[i]) == false
            throw(ArgumentError("every value in growth_rates must be finite"));
        end
    end
end

"""
    build(model::Type{MyGaussianBivariateCopulaModel}, data::NamedTuple) -> MyGaussianBivariateCopulaModel

Build a Gaussian copula for two assets from its parameter.

### Arguments
- `model::Type{MyGaussianBivariateCopulaModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `ρ::Float64`: The correlation, strictly between `-1` and `1`

### Returns
- `MyGaussianBivariateCopulaModel`: The copula

### Example
```julia
copula = JumpHMM.build(JumpHMM.MyGaussianBivariateCopulaModel, (ρ = 0.5,));
```
"""
function build(model::Type{MyGaussianBivariateCopulaModel}, data::NamedTuple)::MyGaussianBivariateCopulaModel

    # check: the required keys are present -
    if haskey(data, :ρ) == false
        throw(ArgumentError("ρ is required"));
    end
    ρ = data.ρ;

    # check: the parameters are in range -
    if (ρ > -1.0 && ρ < 1.0) == false
        throw(ArgumentError("ρ must be strictly between -1 and 1"));
    end

    # build an empty model, then fill in the fields -
    m = MyGaussianBivariateCopulaModel();
    m.ρ = ρ;

    # return -
    return m;
end

"""
    build(model::Type{MyStudentTBivariateCopulaModel}, data::NamedTuple) -> MyStudentTBivariateCopulaModel

Build a Student-t copula for two assets from its parameter.

### Arguments
- `model::Type{MyStudentTBivariateCopulaModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `ρ::Float64`: The correlation, strictly between `-1` and `1`
    - `ν::Float64`: The degrees of freedom, above `0`

### Returns
- `MyStudentTBivariateCopulaModel`: The copula

### Example
```julia
copula = JumpHMM.build(JumpHMM.MyStudentTBivariateCopulaModel, (ρ = 0.5, ν = 5.0));
```
"""
function build(model::Type{MyStudentTBivariateCopulaModel}, data::NamedTuple)::MyStudentTBivariateCopulaModel

    # check: the required keys are present -
    if haskey(data, :ρ) == false
        throw(ArgumentError("ρ is required"));
    end
    if haskey(data, :ν) == false
        throw(ArgumentError("ν is required"));
    end
    ρ = data.ρ;
    ν = data.ν;

    # check: the parameters are in range -
    if (ρ > -1.0 && ρ < 1.0) == false
        throw(ArgumentError("ρ must be strictly between -1 and 1"));
    end
    if (ν > 0.0) == false
        throw(ArgumentError("ν must be above 0"));
    end

    # build an empty model, then fill in the fields -
    m = MyStudentTBivariateCopulaModel();
    m.ρ = ρ;
    m.ν = ν;

    # return -
    return m;
end

"""
    build(model::Type{MyClaytonBivariateCopulaModel}, data::NamedTuple) -> MyClaytonBivariateCopulaModel

Build a Clayton copula for two assets from its parameter.

### Arguments
- `model::Type{MyClaytonBivariateCopulaModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `θ::Float64`: The dependence parameter, above `0`

### Returns
- `MyClaytonBivariateCopulaModel`: The copula

### Example
```julia
copula = JumpHMM.build(JumpHMM.MyClaytonBivariateCopulaModel, (θ = 2.0,));
```
"""
function build(model::Type{MyClaytonBivariateCopulaModel}, data::NamedTuple)::MyClaytonBivariateCopulaModel

    # check: the required keys are present -
    if haskey(data, :θ) == false
        throw(ArgumentError("θ is required"));
    end
    θ = data.θ;

    # check: the parameters are in range -
    if (θ > 0.0) == false
        throw(ArgumentError("θ must be above 0"));
    end

    # build an empty model, then fill in the fields -
    m = MyClaytonBivariateCopulaModel();
    m.θ = θ;

    # return -
    return m;
end

"""
    build(model::Type{MyGumbelBivariateCopulaModel}, data::NamedTuple) -> MyGumbelBivariateCopulaModel

Build a Gumbel copula for two assets from its parameter.

### Arguments
- `model::Type{MyGumbelBivariateCopulaModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `θ::Float64`: The dependence parameter, at least `1`

### Returns
- `MyGumbelBivariateCopulaModel`: The copula

### Example
```julia
copula = JumpHMM.build(JumpHMM.MyGumbelBivariateCopulaModel, (θ = 2.0,));
```
"""
function build(model::Type{MyGumbelBivariateCopulaModel}, data::NamedTuple)::MyGumbelBivariateCopulaModel

    # check: the required keys are present -
    if haskey(data, :θ) == false
        throw(ArgumentError("θ is required"));
    end
    θ = data.θ;

    # check: the parameters are in range -
    if (θ ≥ 1.0) == false
        throw(ArgumentError("θ must be at least 1"));
    end

    # build an empty model, then fill in the fields -
    m = MyGumbelBivariateCopulaModel();
    m.θ = θ;

    # return -
    return m;
end

"""
    build(model::Type{MyFrankBivariateCopulaModel}, data::NamedTuple) -> MyFrankBivariateCopulaModel

Build a Frank copula for two assets from its parameter.

### Arguments
- `model::Type{MyFrankBivariateCopulaModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `θ::Float64`: The dependence parameter, finite

### Returns
- `MyFrankBivariateCopulaModel`: The copula

### Example
```julia
copula = JumpHMM.build(JumpHMM.MyFrankBivariateCopulaModel, (θ = 3.0,));
```
"""
function build(model::Type{MyFrankBivariateCopulaModel}, data::NamedTuple)::MyFrankBivariateCopulaModel

    # check: the required keys are present -
    if haskey(data, :θ) == false
        throw(ArgumentError("θ is required"));
    end
    θ = data.θ;

    # check: the parameters are in range -
    if isfinite(θ) == false
        throw(ArgumentError("θ must be finite"));
    end

    # build an empty model, then fill in the fields -
    m = MyFrankBivariateCopulaModel();
    m.θ = θ;

    # return -
    return m;
end

"""
    build(model::Type{MyVineEdgeModel}, data::NamedTuple) -> MyVineEdgeModel

Build one edge of a vine copula from the two assets, the conditioned assets, and the bivariate copula.

### Arguments
- `model::Type{MyVineEdgeModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `first::Int64`: The index of the first asset of the edge
    - `second::Int64`: The index of the second asset of the edge
    - `conditioned::Array{Int64,1}`: The indices of the assets that the edge is conditioned on
    - `copula::AbstractBivariateCopulaModel`: The bivariate copula of the edge

### Returns
- `MyVineEdgeModel`: The edge

### Example
```julia
edge = JumpHMM.build(JumpHMM.MyVineEdgeModel, (first = 1, second = 2, conditioned = Int64[], copula = copula));
```
"""
function build(model::Type{MyVineEdgeModel}, data::NamedTuple)::MyVineEdgeModel

    # check: the required keys are present -
    if haskey(data, :first) == false
        throw(ArgumentError("first is required"));
    end
    if haskey(data, :second) == false
        throw(ArgumentError("second is required"));
    end
    if haskey(data, :conditioned) == false
        throw(ArgumentError("conditioned is required"));
    end
    if haskey(data, :copula) == false
        throw(ArgumentError("copula is required"));
    end

    # build an empty model, then fill in the fields -
    m = MyVineEdgeModel();
    m.first = data.first;
    m.second = data.second;
    m.conditioned = data.conditioned;
    m.copula = data.copula;

    # return -
    return m;
end

"""
    build(model::Type{MyVineCopulaModel}, data::NamedTuple) -> MyVineCopulaModel

Fit a C-vine copula to the growth rates of several assets. We replace each column of growth rates by its ranks divided by the
number of rows plus one. We order the assets by the sum of the absolute Kendall's tau with every other asset, largest first.
Then we build the vine tree by tree. In tree `k`, the `k`-th asset in this order is paired with each later asset, and we pick the
bivariate copula of each edge by AIC among the Gaussian, Student-t, Clayton, Gumbel, and Frank families.

### Arguments
- `model::Type{MyVineCopulaModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `growth_rates::Array{Float64,2}`: The growth rates, one row per day and one column per asset. We need at least two columns, at least two rows, and every value must be finite

### Returns
- `MyVineCopulaModel`: The fitted vine copula

### Example
```julia
vine = JumpHMM.build(JumpHMM.MyVineCopulaModel, (growth_rates = R,));
```
"""
function build(model::Type{MyVineCopulaModel}, data::NamedTuple)::MyVineCopulaModel

    # check: the required key is present, and the matrix is usable -
    if haskey(data, :growth_rates) == false
        throw(ArgumentError("growth_rates is required"));
    end
    growth_rates = data.growth_rates;
    _check_copula_growth_rates(growth_rates);

    # initialize -
    U = _uniform_ranks(growth_rates);
    n, d = size(U);

    # order the assets by the sum of the absolute Kendall's tau with every other asset -
    tau_sums = zeros(d);
    for i ∈ 1:d
        for j ∈ (i+1):d
            τ = abs(corkendall(U[:, i], U[:, j]));
            tau_sums[i] += τ;
            tau_sums[j] += τ;
        end
    end
    order = sortperm(tau_sums, rev = true);
    V = U[:, order];

    # the pseudo-observations of each asset, which become conditional uniform numbers from tree to tree -
    pseudo = [V[:, j] for j ∈ 1:d];
    edges = Array{MyVineEdgeModel,1}[];

    # main loop - one tree at a time -
    for k ∈ 1:(d-1)
        tree_edges = MyVineEdgeModel[];
        new_pseudo = Array{Array{Float64,1},1}(undef, d);

        # the k-th asset is the center of tree k -
        center_data = pseudo[k];
        for j ∈ (k+1):d
            paired_data = pseudo[j];

            # pick the copula of the edge, then build the edge -
            cop = _select_bicop(center_data, paired_data);
            edge = build(MyVineEdgeModel, (first = order[k], second = order[j], conditioned = order[1:(k-1)], copula = cop));
            push!(tree_edges, edge);

            # the conditional uniform numbers of the paired asset, given the center -
            h_values = zeros(n);
            for i ∈ 1:n
                h_values[i] = _bicop_hfunc(cop, paired_data[i], center_data[i]);
            end
            new_pseudo[j] = h_values;
        end

        # the conditional uniform numbers of the center, given the first paired asset -
        if k < d - 1
            cop_first = tree_edges[1].copula;
            h_center = zeros(n);
            for i ∈ 1:n
                h_center[i] = _bicop_hfunc(cop_first, center_data[i], pseudo[k+1][i]);
            end
            # this value is never read, because the next tree starts at asset k + 1 -
            new_pseudo[k] = h_center;
        end

        # update the pseudo-observations for the next tree -
        for j ∈ (k+1):d
            pseudo[j] = new_pseudo[j];
        end
        push!(edges, tree_edges);
    end

    # build an empty model, then fill in the fields -
    m = MyVineCopulaModel();
    m.number_of_assets = d;
    m.order = order;
    m.edges = edges;

    # return -
    return m;
end

"""
    build(model::Type{MySingleIndexDependenceModel}, data::NamedTuple) -> MySingleIndexDependenceModel

Fit a single-index model to the growth rates of several assets. For each asset, we regress its growth rates on the growth rates of the market,
`g_i = α_i + β_i g_m + ε_i`, using ordinary least squares. We keep the residuals of each asset, and we build a hidden Markov model of the market.

### Arguments
- `model::Type{MySingleIndexDependenceModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `growth_rates::Array{Float64,2}`: The growth rates of the assets, one row per day and one column per asset
    - `market_growth_rates::Array{Float64,1}`: The growth rates of the market. The length must equal the number of rows of `growth_rates`, and the values must not all be equal
    - `number_of_states::Int64`: The number of states of the hidden Markov model of the market
    - `ν::Float64`: The degrees of freedom of the Student-t emission distributions of the market model

### Returns
- `MySingleIndexDependenceModel`: The fitted model

### Optional Arguments
The optional keys of `data` and their defaults are:
- `market_prices::Union{Array{Float64,1},Nothing}`: The prices of the market. If `nothing`, we rebuild prices from `market_growth_rates`, starting at `100.0`, and fit the market model to the growth rates of the rebuilt prices. The default is `nothing`
- `rf::Float64`: The risk-free rate, per year. The default is `0.0`
- `Δt::Float64`: The time step of the growth rates, in years. The default is `1/252`
- `min_obs::Int64`: The fewest growth rates a state of the market model needs to use its own emission estimates. The default is `2`
- `residual_method::Symbol`: How `sample_dependence` draws the residuals, either `:bootstrap` or `:gaussian`. The default is `:bootstrap`

### Example
```julia
model = JumpHMM.build(JumpHMM.MySingleIndexDependenceModel, (
    growth_rates = G[:, 2:end],
    market_growth_rates = G[:, 1],
    number_of_states = 20,
    ν = 5.0
));
```
"""
function build(model::Type{MySingleIndexDependenceModel}, data::NamedTuple)::MySingleIndexDependenceModel

    # check: the required keys are present -
    if haskey(data, :growth_rates) == false
        throw(ArgumentError("growth_rates is required"));
    end
    if haskey(data, :market_growth_rates) == false
        throw(ArgumentError("market_growth_rates is required"));
    end
    if haskey(data, :number_of_states) == false
        throw(ArgumentError("number_of_states is required"));
    end
    if haskey(data, :ν) == false
        throw(ArgumentError("ν is required"));
    end
    growth_rates = data.growth_rates;
    market_growth_rates = data.market_growth_rates;
    number_of_states = data.number_of_states;
    ν = data.ν;

    # set the optional keys to their defaults, then replace the ones that were given -
    market_prices = nothing;
    rf = 0.0;
    Δt = 1/252;
    min_obs = 2;
    residual_method = :bootstrap;
    if haskey(data, :market_prices) == true
        market_prices = data.market_prices;
    end
    if haskey(data, :rf) == true
        rf = data.rf;
    end
    if haskey(data, :Δt) == true
        Δt = data.Δt;
    end
    if haskey(data, :min_obs) == true
        min_obs = data.min_obs;
    end
    if haskey(data, :residual_method) == true
        residual_method = data.residual_method;
    end

    # check: the residual method is known, and the market matches the assets -
    if residual_method != :bootstrap && residual_method != :gaussian
        throw(ArgumentError("residual_method must be :bootstrap or :gaussian, got :$(residual_method)"));
    end
    (number_of_days, number_of_assets) = size(growth_rates);
    if length(market_growth_rates) != number_of_days
        throw(ArgumentError("market_growth_rates must have one value for each row of growth_rates"));
    end

    # fit the hidden Markov model of the market -
    if isnothing(market_prices) == true
        # no prices were given, so we rebuild prices from the market growth rates, starting at 100.0, and take their growth rates.
        # This round trip is redundant, but it keeps the numbers equal to the earlier version -
        market_prices = prices_from_growth_rates(market_growth_rates, 100.0; rf = rf, Δt = Δt);
    end
    market_model = build(MyJumpHiddenMarkovModel, (
        growth_rates = excess_growth_rates(market_prices; rf = rf, Δt = Δt),
        number_of_states = number_of_states,
        ν = ν,
        Δt = Δt,
        min_obs = min_obs
    ));

    # check: the market is not constant -
    market_mean = mean(market_growth_rates);
    market_variance = var(market_growth_rates);
    if market_variance ≈ 0.0
        throw(ArgumentError("the market growth rates have zero variance (a constant market)"));
    end

    # regress each asset on the market, and keep the residuals -
    α = Array{Float64,1}(undef, number_of_assets);
    β = Array{Float64,1}(undef, number_of_assets);
    σ_ε = Array{Float64,1}(undef, number_of_assets);
    residuals = Array{Float64,2}(undef, number_of_days, number_of_assets);
    for i ∈ 1:number_of_assets
        asset_growth_rates = growth_rates[:, i];
        β[i] = cov(asset_growth_rates, market_growth_rates) / market_variance;
        α[i] = mean(asset_growth_rates) - β[i] * market_mean;
        residuals[:, i] = asset_growth_rates .- α[i] .- β[i] .* market_growth_rates;
        σ_ε[i] = std(residuals[:, i]);
    end

    # build an empty model, then fill in the fields -
    m = MySingleIndexDependenceModel();
    m.α = α;
    m.β = β;
    m.σ_ε = σ_ε;
    m.residuals = residuals;
    m.residual_method = residual_method;
    m.market_model = market_model;

    # return -
    return m;
end

"""
    build(model::Type{MyHybridSingleIndexDependenceModel}, data::NamedTuple) -> MyHybridSingleIndexDependenceModel

Fit a hybrid single-index model from a matrix of prices. We build a hidden Markov model of the market and one of each other asset,
regress the growth rates of each asset on the growth rates of the market to get `α`, `β`, and `r²`, and fit a copula to the residuals.

### Arguments
- `model::Type{MyHybridSingleIndexDependenceModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `tickers::Array{String,1}`: The label of each column of `prices`. The list must include `market`
    - `prices::Array{Float64,2}`: The prices, one row per day and one column per ticker
    - `market::String`: The ticker of the market
    - `number_of_states::Int64`: The number of states of each hidden Markov model
    - `ν::Float64`: The degrees of freedom of the Student-t emission distributions

### Returns
- `MyHybridSingleIndexDependenceModel`: The fitted model. The tickers of the model are the tickers without the market, in their original order

### Optional Arguments
The optional keys of `data` and their defaults are:
- `copula::Type`: The type of the copula fit to the residuals. The default is `MyStudentTCopulaModel`
- `rf::Float64`: The risk-free rate, per year. The default is `0.0`
- `Δt::Float64`: The time step of the growth rates, in years. The default is `1/252`
- `min_obs::Int64`: The fewest growth rates a state needs to use its own emission estimates. The default is `2`
- `r2_preserve_threshold::Float64`: The R² at or above which the model keeps the R² of the asset. The default is `0.80`
- `idiosyncratic_floor::Float64`: The smallest fraction of the variance of an asset that is idiosyncratic. The default is `0.10`

### Example
```julia
model = JumpHMM.build(JumpHMM.MyHybridSingleIndexDependenceModel, (
    tickers = tickers,
    prices = P,
    market = "SPY",
    number_of_states = 20,
    ν = 5.0
));
```
"""
function build(model::Type{MyHybridSingleIndexDependenceModel}, data::NamedTuple)::MyHybridSingleIndexDependenceModel

    # check: the required keys are present -
    if haskey(data, :tickers) == false
        throw(ArgumentError("tickers is required"));
    end
    if haskey(data, :prices) == false
        throw(ArgumentError("prices is required"));
    end
    if haskey(data, :market) == false
        throw(ArgumentError("market is required"));
    end
    if haskey(data, :number_of_states) == false
        throw(ArgumentError("number_of_states is required"));
    end
    if haskey(data, :ν) == false
        throw(ArgumentError("ν is required"));
    end
    tickers = data.tickers;
    prices = data.prices;
    market = data.market;
    number_of_states = data.number_of_states;
    ν = data.ν;

    # set the optional keys to their defaults, then replace the ones that were given -
    copula_type = MyStudentTCopulaModel;
    rf = 0.0;
    Δt = 1/252;
    min_obs = 2;
    r2_preserve_threshold = 0.80;
    idiosyncratic_floor = 0.10;
    if haskey(data, :copula) == true
        copula_type = data.copula;
    end
    if haskey(data, :rf) == true
        rf = data.rf;
    end
    if haskey(data, :Δt) == true
        Δt = data.Δt;
    end
    if haskey(data, :min_obs) == true
        min_obs = data.min_obs;
    end
    if haskey(data, :r2_preserve_threshold) == true
        r2_preserve_threshold = data.r2_preserve_threshold;
    end
    if haskey(data, :idiosyncratic_floor) == true
        idiosyncratic_floor = data.idiosyncratic_floor;
    end

    # check: the tickers match the columns, and the market is one of them -
    number_of_columns = size(prices, 2);
    if length(tickers) != number_of_columns
        throw(ArgumentError("tickers must have one label for each column of prices"));
    end
    market_index = nothing;
    for j ∈ eachindex(tickers)
        if tickers[j] == market
            market_index = j;
            break;
        end
    end
    if isnothing(market_index) == true
        throw(ArgumentError("the market ticker $(market) is not in tickers"));
    end

    # compute the growth rates of every column, and of the market -
    growth_rates = excess_growth_rates(prices; rf = rf, Δt = Δt);
    market_growth_rates = growth_rates[:, market_index];
    σ_market = std(market_growth_rates);
    if σ_market ≈ 0.0
        throw(ArgumentError("the market growth rates have zero variance (a constant market)"));
    end

    # build the hidden Markov model of the market -
    market_model = build(MyJumpHiddenMarkovModel, (
        growth_rates = excess_growth_rates(prices[:, market_index]; rf = rf, Δt = Δt),
        number_of_states = number_of_states,
        ν = ν,
        Δt = Δt,
        min_obs = min_obs
    ));

    # the assets are every column except the market -
    asset_columns = Array{Int64,1}();
    for j ∈ 1:number_of_columns
        if j != market_index
            push!(asset_columns, j);
        end
    end
    asset_tickers = tickers[asset_columns];
    number_of_assets = length(asset_tickers);

    # initialize -
    marginals = Dict{String,MyJumpHiddenMarkovModel}();
    α = Array{Float64,1}(undef, number_of_assets);
    β = Array{Float64,1}(undef, number_of_assets);
    r² = Array{Float64,1}(undef, number_of_assets);
    residuals = Array{Float64,2}(undef, size(growth_rates, 1), number_of_assets);
    market_mean = mean(market_growth_rates);
    market_variance = var(market_growth_rates);

    # main loop: for each asset, build its hidden Markov model and regress it on the market -
    for k ∈ 1:number_of_assets
        j = asset_columns[k];

        # the hidden Markov model of the asset -
        marginals[asset_tickers[k]] = build(MyJumpHiddenMarkovModel, (
            growth_rates = excess_growth_rates(prices[:, j]; rf = rf, Δt = Δt),
            number_of_states = number_of_states,
            ν = ν,
            Δt = Δt,
            min_obs = min_obs
        ));

        # the regression of the asset on the market -
        asset_growth_rates = growth_rates[:, j];
        asset_mean = mean(asset_growth_rates);
        β[k] = cov(asset_growth_rates, market_growth_rates) / market_variance;
        α[k] = asset_mean - β[k] * market_mean;
        residual = asset_growth_rates .- α[k] .- β[k] .* market_growth_rates;
        residuals[:, k] = residual;

        # the R² of the regression: 0.0 if the asset growth rates are constant -
        SS_residual = dot(residual, residual);
        SS_total = dot(asset_growth_rates .- asset_mean, asset_growth_rates .- asset_mean);
        r²[k] = 0.0;
        if SS_total > 0.0
            r²[k] = 1.0 - SS_residual / SS_total;
        end
    end

    # fit the copula to the residuals -
    copula = build(copula_type, (growth_rates = residuals,));

    # build an empty model, then fill in the fields -
    m = MyHybridSingleIndexDependenceModel();
    m.α = α;
    m.β = β;
    m.r² = r²;
    m.σ_market = σ_market;
    m.marginals = marginals;
    m.copula = copula;
    m.market_model = market_model;
    m.tickers = asset_tickers;
    m.r2_preserve_threshold = r2_preserve_threshold;
    m.idiosyncratic_floor = idiosyncratic_floor;

    # return -
    return m;
end

"""
    build(model::Type{MySectorFactorModel}, data::NamedTuple) -> MySectorFactorModel

Fit a sector factor model from a matrix of prices: `g_i(t) = α_i + β_i(t) g_m(t) + δ_i f_s(t) + ψ_i (ε_i(t) - ε_center[i])`.
We regress the growth rates of each asset on the growth rates of the market to get `α`, `β`, `r²`, and the residuals.
The factor of a sector is the mean residual of its assets (`sector_factors`), and `δ` is the sector loading of each asset (`sector_loadings`).
The own model of each asset is fit to its residual after the sector term (`own_residuals`).

We build a hidden Markov model of the growth rates of the market, of each sector factor, and of each leftover residual. We fit the sector and
own models to level series that start at `100.0` and compound the factor or residual, with `rf = 0`. The long-run mean
`ε_center[i]` of own model `i` is the mean of one `center_steps`-day draw, seeded with `center_seed + i`. The scale `ψ` of each own draw
follows from `own_variances` and the drift of `β` (`own_scale`).

### Arguments
- `model::Type{MySectorFactorModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `tickers::Array{String,1}`: The label of each column of `prices`. The list must include `market`
    - `prices::Array{Float64,2}`: The prices, one row per day and one column per ticker
    - `market::String`: The ticker of the market
    - `sector::Array{String,1}`: The sector label of each asset, in the order of the tickers without the market. Every sector needs at least two assets
    - `number_of_states::Int64`: The number of states of each hidden Markov model
    - `ν::Float64`: The degrees of freedom of the Student-t emission distributions

### Returns
- `MySectorFactorModel`: The fitted model. The tickers of the model are the tickers without the market, in their original order

### Optional Arguments
The optional keys of `data` and their defaults are:
- `rf::Float64`: The risk-free rate, per year, used for the growth rates of the prices. The default is `0.0`
- `Δt::Float64`: The time step of the growth rates, in years. The default is `1/252`
- `min_obs::Int64`: The fewest growth rates a state needs to use its own emission estimates. The default is `2`
- `intercept_pull::Float64`: How much of its distance from the mean intercept each `α` keeps: `α = ᾱ + intercept_pull (α̂ - ᾱ)`. A value of `1.0` keeps the fitted intercepts, and `0.0` gives every asset the mean intercept. The default is `1.0`
- `β_halflife::Float64`: The half-life of the drift of `β`, in trading days. The default is `126.0`
- `β_scale::Union{Array{Float64,1},Nothing}`: The drift scale of `β` for each asset. If `nothing`, every scale is `0.0`, which keeps `β` fixed. The default is `nothing`
- `center_steps::Int64`: The length of the draw that sets the long-run mean of each own model. The default is `1_000_000`
- `center_seed::Int64`: The seed offset of the centering draws. Asset `i` uses `center_seed + i`. The default is `4 * 10^12`
- `market_model::Union{MyJumpHiddenMarkovModel,Nothing}`: A market model to use instead of the one we would fit. The default is `nothing`
- `sector_models::Union{Array{MyJumpHiddenMarkovModel,1},Nothing}`: The sector models to use instead of the ones we would fit, one per sector, in sorted label order. The default is `nothing`
- `own_models::Union{Array{MyJumpHiddenMarkovModel,1},Nothing}`: The own models to use instead of the ones we would fit, one per asset, in asset order. We compute `ε_center` from the own models the model stores. The default is `nothing`

### Example
```julia
model = JumpHMM.build(JumpHMM.MySectorFactorModel, (
    tickers = tickers,
    prices = P,
    market = "SPY",
    sector = sector,
    number_of_states = 20,
    ν = 5.0
));
```
"""
function build(model::Type{MySectorFactorModel}, data::NamedTuple)::MySectorFactorModel

    # check: the required keys are present -
    if haskey(data, :tickers) == false
        throw(ArgumentError("tickers is required"));
    end
    if haskey(data, :prices) == false
        throw(ArgumentError("prices is required"));
    end
    if haskey(data, :market) == false
        throw(ArgumentError("market is required"));
    end
    if haskey(data, :sector) == false
        throw(ArgumentError("sector is required"));
    end
    if haskey(data, :number_of_states) == false
        throw(ArgumentError("number_of_states is required"));
    end
    if haskey(data, :ν) == false
        throw(ArgumentError("ν is required"));
    end
    tickers = data.tickers;
    prices = data.prices;
    market = data.market;
    sector = data.sector;
    number_of_states = data.number_of_states;
    ν = data.ν;

    # set the optional keys to their defaults, then replace the ones that were given -
    rf = 0.0;
    Δt = 1/252;
    min_obs = 2;
    intercept_pull = 1.0;
    β_halflife = 126.0;
    β_scale = nothing;
    center_steps = 1_000_000;
    center_seed = 4 * 10^12;
    market_model = nothing;
    sector_models = nothing;
    own_models = nothing;
    if haskey(data, :rf) == true
        rf = data.rf;
    end
    if haskey(data, :Δt) == true
        Δt = data.Δt;
    end
    if haskey(data, :min_obs) == true
        min_obs = data.min_obs;
    end
    if haskey(data, :intercept_pull) == true
        intercept_pull = data.intercept_pull;
    end
    if haskey(data, :β_halflife) == true
        β_halflife = data.β_halflife;
    end
    if haskey(data, :β_scale) == true
        β_scale = data.β_scale;
    end
    if haskey(data, :center_steps) == true
        center_steps = data.center_steps;
    end
    if haskey(data, :center_seed) == true
        center_seed = data.center_seed;
    end
    if haskey(data, :market_model) == true
        market_model = data.market_model;
    end
    if haskey(data, :sector_models) == true
        sector_models = data.sector_models;
    end
    if haskey(data, :own_models) == true
        own_models = data.own_models;
    end

    # check: the tickers match the columns, and the market is one of them -
    number_of_columns = size(prices, 2);
    if length(tickers) != number_of_columns
        throw(ArgumentError("tickers has $(length(tickers)) labels but prices has $(number_of_columns) columns"));
    end
    market_index = 0;
    for j ∈ 1:number_of_columns
        if tickers[j] == market
            market_index = j;
            break;
        end
    end
    if market_index == 0
        throw(ArgumentError("the market ticker $(market) is not in tickers"));
    end

    # the assets are every column except the market -
    asset_columns = Array{Int64,1}();
    for j ∈ 1:number_of_columns
        if j != market_index
            push!(asset_columns, j);
        end
    end
    asset_tickers = tickers[asset_columns];
    number_of_assets = length(asset_tickers);

    # check: one sector label and one drift scale per asset. No drift scales means zeros, a fixed β -
    if length(sector) != number_of_assets
        throw(ArgumentError("sector has $(length(sector)) labels but there are $(number_of_assets) assets"));
    end
    if isnothing(β_scale) == true
        β_scale = zeros(number_of_assets);
    end
    if length(β_scale) != number_of_assets
        throw(ArgumentError("β_scale has $(length(β_scale)) entries but there are $(number_of_assets) assets"));
    end

    # compute the growth rates of every column, then split off the market and the assets -
    all_growth_rates = excess_growth_rates(prices; rf = rf, Δt = Δt);
    market_growth_rates = all_growth_rates[:, market_index];
    growth_rates = all_growth_rates[:, asset_columns];

    # regress each asset on the market with ordinary least squares, and keep the residuals -
    X = hcat(ones(length(market_growth_rates)), market_growth_rates);
    α̂ = zeros(number_of_assets);
    β̂ = zeros(number_of_assets);
    r² = zeros(number_of_assets);
    E = similar(growth_rates);
    for i ∈ 1:number_of_assets
        coefficients = X \ growth_rates[:, i];
        α̂[i] = coefficients[1];
        β̂[i] = coefficients[2];
        E[:, i] = growth_rates[:, i] .- X * coefficients;
        r²[i] = 1 - var(E[:, i]) / var(growth_rates[:, i]);
    end

    # the sector factors and loadings. These throw for a sector with one asset, or a covariance that is not positive -
    sf = sector_factors(E, sector);
    δ = sector_loadings(E, sector);

    # the position of the sector of each asset in the sorted labels -
    sector_index = zeros(Int64, number_of_assets);
    for i ∈ 1:number_of_assets
        for s ∈ eachindex(sf.sectors)
            if sf.sectors[s] == sector[i]
                sector_index[i] = s;
            end
        end
    end

    # the residual each asset has left after its sector term, and the variances that size its own draw -
    U = own_residuals(E, sf.F, δ, sector_index);
    variances = own_variances(E, sf.F, U, δ, sector_index; tickers = asset_tickers);
    σ²_m = var(market_growth_rates);

    # the intercepts, pulled toward their mean by intercept_pull -
    ᾱ = mean(α̂);
    α = ᾱ .+ intercept_pull .* (α̂ .- ᾱ);

    # check: given models come one per asset and one per sector -
    if isnothing(own_models) == false
        if length(own_models) != number_of_assets
            throw(ArgumentError("own_models has $(length(own_models)) models but there are $(number_of_assets) assets"));
        end
    end
    if isnothing(sector_models) == false
        if length(sector_models) != length(sf.sectors)
            throw(ArgumentError("sector_models has $(length(sector_models)) models but there are $(length(sf.sectors)) sectors"));
        end
    end

    # the market model: use the given one, or fit one to the market prices -
    if isnothing(market_model) == true
        market_model = build(MyJumpHiddenMarkovModel, (
            growth_rates = excess_growth_rates(prices[:, market_index]; rf = rf, Δt = Δt),
            number_of_states = number_of_states,
            ν = ν,
            Δt = Δt,
            min_obs = min_obs
        ));
    end

    # compound each sector factor into a level series that starts at 100.0. We fit each sector model to the growth rates of
    # its level series. This round trip is redundant, but it keeps the numbers equal to the earlier version -
    sector_levels = 100.0 .* exp.(vcat(zeros(1, size(sf.F, 2)), cumsum(sf.F .* Δt; dims = 1)));

    # the sector models: copy the given list, or fit one model to each sector level series, with rf = 0 -
    if isnothing(sector_models) == false
        sector_models = copy(sector_models);
    else
        sector_models = Array{MyJumpHiddenMarkovModel,1}();
        for s ∈ eachindex(sf.sectors)
            sector_model = build(MyJumpHiddenMarkovModel, (
                growth_rates = excess_growth_rates(sector_levels[:, s]; rf = 0.0, Δt = Δt),
                number_of_states = number_of_states,
                ν = ν,
                Δt = Δt,
                min_obs = min_obs
            ));
            push!(sector_models, sector_model);
        end
    end

    # compound each leftover residual into a level series that starts at 100.0, the same way -
    own_levels = 100.0 .* exp.(vcat(zeros(1, size(U, 2)), cumsum(U .* Δt; dims = 1)));

    # the own models: copy the given list, or fit one model to each own level series, with rf = 0 -
    if isnothing(own_models) == false
        own_models = copy(own_models);
    else
        own_models = Array{MyJumpHiddenMarkovModel,1}();
        for i ∈ 1:number_of_assets
            own_model = build(MyJumpHiddenMarkovModel, (
                growth_rates = excess_growth_rates(own_levels[:, i]; rf = 0.0, Δt = Δt),
                number_of_states = number_of_states,
                ν = ν,
                Δt = Δt,
                min_obs = min_obs
            ));
            push!(own_models, own_model);
        end
    end

    # the long-run mean of each own model: the mean of one long draw, seeded with center_seed + i.
    # We take the mean of the whole draw at once, so the sum adds in the same order as the earlier version -
    ε_center = zeros(number_of_assets);
    for i ∈ 1:number_of_assets
        center_path = simulate(own_models[i], center_steps; number_of_paths = 1, seed = center_seed + i);
        ε_center[i] = mean(center_path.growth_rates[:, 1]);
    end

    # the scale of each own draw under this drift of β -
    ψ = zeros(number_of_assets);
    for i ∈ 1:number_of_assets
        σ²_drift = drift_variance(β_halflife, β_scale[i]) * σ²_m;
        ψ[i] = own_scale(variances.σ²_own[i], variances.σ²_u[i], σ²_drift; ticker = asset_tickers[i]);
    end

    # build an empty model, then fill in the fields -
    m = MySectorFactorModel();
    m.tickers = asset_tickers;
    m.sector = sector;
    m.sectors = sf.sectors;
    m.sector_index = sector_index;
    m.α = α;
    m.β = β̂;
    m.δ = δ;
    m.r² = r²;
    m.ψ = ψ;
    m.σ²_own = variances.σ²_own;
    m.σ²_u = variances.σ²_u;
    m.σ²_m = σ²_m;
    m.ε_center = ε_center;
    m.β_halflife = β_halflife;
    m.β_scale = β_scale;
    m.market_model = market_model;
    m.sector_models = sector_models;
    m.own_models = own_models;
    m.sector_levels = sector_levels;
    m.own_levels = own_levels;
    m.center_steps = center_steps;
    m.center_seed = center_seed;

    # return -
    return m;
end

"""
    build(model::Type{MyPortfolioModel}, data::NamedTuple) -> MyPortfolioModel

Fit a model of several assets: one hidden Markov model for each asset, and a dependence model that links the assets.
The dependence model is a Gaussian, Student-t, or vine copula fit to the growth rates of all the assets, a single-index model,
a hybrid single-index model, or a sector factor model. The single-index, hybrid, and sector factor models use a market,
and a portfolio built from one of them leaves the market ticker out of `tickers`.

### Arguments
- `model::Type{MyPortfolioModel}`: The type to build
- `data::NamedTuple`: The inputs. The required keys are:
    - `tickers::Array{String,1}`: The label of each column of `prices`
    - `prices::Array{Float64,2}`: The prices, one row per day and one column per ticker
    - `dependence::Type`: The type of the dependence model: `MyGaussianCopulaModel`, `MyStudentTCopulaModel`, `MyVineCopulaModel`, `MySingleIndexDependenceModel`, `MyHybridSingleIndexDependenceModel`, or `MySectorFactorModel`
    - `number_of_states::Int64`: The number of states of each hidden Markov model
    - `ν::Float64`: The degrees of freedom of the Student-t emission distributions

### Returns
- `MyPortfolioModel`: The fitted model

### Optional Arguments
The optional keys of `data` and their defaults are:
- `market::String`: The ticker of the market. It is required for the single-index, hybrid, and sector factor models, and it must be in `tickers`. The default is `""`
- `rf::Float64`: The risk-free rate, per year, used for the growth rates of the prices. The default is `0.0`
- `Δt::Float64`: The time step of the growth rates, in years. The default is `1/252`
- `min_obs::Int64`: The fewest growth rates a state needs to use its own emission estimates. The default is `2`
- `residual_method::Symbol`: How a single-index model draws the residuals, either `:bootstrap` or `:gaussian`. The default is `:bootstrap`
- `sector::Array{String,1}`: The sector label of each ticker except the market. It is required for the sector factor model. The default is `String[]`
- `sector_options::NamedTuple`: More optional keys for the build of the sector factor model, for example `(center_steps = 20_000,)`. The default is `NamedTuple()`

### Example
```julia
portfolio = JumpHMM.build(JumpHMM.MyPortfolioModel, (
    tickers = tickers,
    prices = P,
    dependence = JumpHMM.MyStudentTCopulaModel,
    number_of_states = 20,
    ν = 5.0
));
```
"""
function build(model::Type{MyPortfolioModel}, data::NamedTuple)::MyPortfolioModel

    # check: the required keys are present -
    if haskey(data, :tickers) == false
        throw(ArgumentError("tickers is required"));
    end
    if haskey(data, :prices) == false
        throw(ArgumentError("prices is required"));
    end
    if haskey(data, :dependence) == false
        throw(ArgumentError("dependence is required"));
    end
    if haskey(data, :number_of_states) == false
        throw(ArgumentError("number_of_states is required"));
    end
    if haskey(data, :ν) == false
        throw(ArgumentError("ν is required"));
    end
    tickers = data.tickers;
    prices = data.prices;
    dependence = data.dependence;
    number_of_states = data.number_of_states;
    ν = data.ν;

    # set the optional keys to their defaults, then replace the ones that were given -
    market = "";
    rf = 0.0;
    Δt = 1/252;
    min_obs = 2;
    residual_method = :bootstrap;
    sector = String[];
    sector_options = NamedTuple();
    if haskey(data, :market) == true
        market = data.market;
    end
    if haskey(data, :rf) == true
        rf = data.rf;
    end
    if haskey(data, :Δt) == true
        Δt = data.Δt;
    end
    if haskey(data, :min_obs) == true
        min_obs = data.min_obs;
    end
    if haskey(data, :residual_method) == true
        residual_method = data.residual_method;
    end
    if haskey(data, :sector) == true
        sector = data.sector;
    end
    if haskey(data, :sector_options) == true
        sector_options = data.sector_options;
    end

    # check: one ticker for each column of prices -
    number_of_columns = size(prices, 2);
    if length(tickers) != number_of_columns
        throw(ArgumentError("tickers has $(length(tickers)) labels but prices has $(number_of_columns) columns"));
    end

    # check: the dependence type is one we know, and the market and sector labels it needs are given -
    is_copula = (dependence <: MyGaussianCopulaModel) || (dependence <: MyStudentTCopulaModel) || (dependence <: MyVineCopulaModel);
    is_single_index = dependence <: MySingleIndexDependenceModel;
    is_hybrid = dependence <: MyHybridSingleIndexDependenceModel;
    is_sector = dependence <: MySectorFactorModel;
    if (is_copula || is_single_index || is_hybrid || is_sector) == false
        throw(ArgumentError("unsupported dependence model: $(dependence)"));
    end
    if (is_single_index || is_hybrid || is_sector) == true
        if isempty(market) == true
            throw(ArgumentError("market is required for $(dependence)"));
        end
    end
    if is_sector == true
        if isempty(sector) == true
            throw(ArgumentError("sector (one label for each ticker except the market) is required for $(dependence)"));
        end
    end

    # the column of each ticker in the price matrix -
    tickers_map = Dict{String,Int64}();
    for j ∈ 1:number_of_columns
        tickers_map[tickers[j]] = j;
    end

    # build an empty model, and fill in the fields that every dependence type shares -
    m = MyPortfolioModel();
    m.tickers_map = tickers_map;
    m.market_ticker = market;
    m.rf = rf;
    m.Δt = Δt;

    # copula: a hidden Markov model for each asset, then a copula fit to the growth rates of all the assets -
    if is_copula == true
        marginals = Dict{String,MyJumpHiddenMarkovModel}();
        for j ∈ 1:number_of_columns
            marginals[tickers[j]] = build(MyJumpHiddenMarkovModel, (
                growth_rates = excess_growth_rates(prices[:, j]; rf = rf, Δt = Δt),
                number_of_states = number_of_states,
                ν = ν,
                Δt = Δt,
                min_obs = min_obs
            ));
        end
        growth_rates = excess_growth_rates(prices; rf = rf, Δt = Δt);
        m.tickers = copy(tickers);
        m.marginals = marginals;
        m.dependence = build(dependence, (growth_rates = growth_rates,));

        # a copula portfolio has no market, so the stored market ticker is empty -
        m.market_ticker = "";
    end

    # single index: the single-index model, then a hidden Markov model for each asset except the market -
    if is_single_index == true
        market_index = nothing;
        for j ∈ eachindex(tickers)
            if tickers[j] == market
                market_index = j;
                break;
            end
        end
        if isnothing(market_index) == true
            throw(ArgumentError("the market ticker $(market) is not in tickers"));
        end
        asset_columns = Array{Int64,1}();
        for j ∈ 1:number_of_columns
            if j != market_index
                push!(asset_columns, j);
            end
        end
        growth_rates = excess_growth_rates(prices; rf = rf, Δt = Δt);
        dependence_model = build(dependence, (
            growth_rates = growth_rates[:, asset_columns],
            market_growth_rates = growth_rates[:, market_index],
            market_prices = prices[:, market_index],
            number_of_states = number_of_states,
            ν = ν,
            rf = rf,
            Δt = Δt,
            min_obs = min_obs,
            residual_method = residual_method
        ));
        marginals = Dict{String,MyJumpHiddenMarkovModel}();
        for j ∈ asset_columns
            marginals[tickers[j]] = build(MyJumpHiddenMarkovModel, (
                growth_rates = excess_growth_rates(prices[:, j]; rf = rf, Δt = Δt),
                number_of_states = number_of_states,
                ν = ν,
                Δt = Δt,
                min_obs = min_obs
            ));
        end
        m.tickers = tickers[asset_columns];
        m.marginals = marginals;
        m.dependence = dependence_model;
    end

    # hybrid single index: the model holds the hidden Markov models of the assets, and the portfolio shares them -
    if is_hybrid == true
        dependence_model = build(dependence, (
            tickers = tickers,
            prices = prices,
            market = market,
            number_of_states = number_of_states,
            ν = ν,
            rf = rf,
            Δt = Δt,
            min_obs = min_obs
        ));
        m.tickers = dependence_model.tickers;
        m.marginals = dependence_model.marginals;
        m.dependence = dependence_model;
    end

    # sector factor: the model holds the own models, so the marginals are the own models in the order of the tickers -
    if is_sector == true
        dependence_model = build(dependence, merge((
            tickers = tickers,
            prices = prices,
            market = market,
            sector = sector,
            number_of_states = number_of_states,
            ν = ν,
            rf = rf,
            Δt = Δt,
            min_obs = min_obs
        ), sector_options));
        marginals = Dict{String,MyJumpHiddenMarkovModel}();
        for i ∈ eachindex(dependence_model.tickers)
            marginals[dependence_model.tickers[i]] = dependence_model.own_models[i];
        end
        m.tickers = dependence_model.tickers;
        m.marginals = marginals;
        m.dependence = dependence_model;
    end

    # return -
    return m;
end
