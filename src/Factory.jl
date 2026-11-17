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
    - `θ::Float64`: The dependence parameter, not `0`

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
    if (isfinite(θ) && θ != 0.0) == false
        throw(ArgumentError("θ must be finite and not 0"));
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
