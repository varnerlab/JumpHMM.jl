"""
    tune(portfolio::MyPortfolioModel, prices::Array{Float64,2}; tickers_map::Union{Dict{String,Int64},Nothing} = nothing,
        market_col::Union{Int64,Nothing} = nothing, ϵ_range = range(1e-4, 2.5e-2, length = 20),
        λ_range = range(10.0, 160.0, length = 16), number_of_paths::Int64 = 200, number_of_steps::Int64 = 0,
        w_κ::Float64 = 0.20, p_neg::Float64 = 0.52, N_tail::Int64 = 5, acf_lags::Int64 = 25,
        seed::Union{Int64,Nothing} = nothing) -> MyPortfolioModel

Tune the jump settings `ϵ` and `λ` of each hidden Markov model of a portfolio with `tune`, and return a new portfolio. We turn each price column
into growth rates with `portfolio.rf` and `portfolio.Δt`. The portfolio you give is not changed. The new portfolio shares every part that tuning
does not change with the portfolio you give.

What we tune depends on the dependence model of the portfolio. A copula portfolio tunes the hidden Markov model of each asset. A single-index
portfolio also tunes the market model, and a hybrid portfolio tunes the market model and the models of the assets. A sector factor portfolio tunes
the market model on the market prices, each sector model on its level series, and each own model on its level series. We turn the level series into
growth rates with `rf = 0.0`, because we built them that way. After tuning, we draw the long-run mean of each own model again, with the same
`center_steps` and seeds as the portfolio you give. The scales `ψ` of the own draws stay as they are.

### Arguments
- `portfolio::MyPortfolioModel`: The portfolio to tune
- `prices::Array{Float64,2}`: The prices, one row per day. The columns are the ones the portfolio was built from

### Returns
- `MyPortfolioModel`: The tuned portfolio

### Optional Arguments
- `tickers_map::Union{Dict{String,Int64},Nothing}`: The column of each ticker in `prices`. If `nothing`, we use `portfolio.tickers_map`. Default: `nothing`
- `market_col::Union{Int64,Nothing}`: The column of the market in `prices`. If `nothing`, we use the column of `portfolio.market_ticker`. Default: `nothing`
- `ϵ_range`: The candidate values of `ϵ`. Default: `range(1e-4, 2.5e-2, length = 20)`
- `λ_range`: The candidate values of `λ`. Default: `range(10.0, 160.0, length = 16)`
- `number_of_paths::Int64`: The number of paths to simulate for each candidate pair. Default: `200`
- `number_of_steps::Int64`: The number of days of each simulated path. If `0`, we use the number of days of the data. Default: `0`
- `w_κ::Float64`: The weight of the kurtosis error in the score. Default: `0.20`
- `p_neg::Float64`: The probability that a jump episode moves into the negative tail. Default: `0.52`
- `N_tail::Int64`: The number of states in each tail that a jump episode can reach. Default: `5`
- `acf_lags::Int64`: The number of lags of the autocorrelation. We shorten it if the data is too short. Default: `25`
- `seed::Union{Int64,Nothing}`: The seed of the global random number generator, set before each model is tuned. If `nothing`, the generator is not reseeded. Default: `nothing`

### Example
```julia
tuned = JumpHMM.tune(portfolio, P; ϵ_range = [1e-4, 3e-3], λ_range = [10.0, 100.0], number_of_paths = 5, seed = 15);
```
"""
function tune(portfolio::MyPortfolioModel, prices::Array{Float64,2}; tickers_map::Union{Dict{String,Int64},Nothing} = nothing,
    market_col::Union{Int64,Nothing} = nothing, ϵ_range = range(1e-4, 2.5e-2, length = 20),
    λ_range = range(10.0, 160.0, length = 16), number_of_paths::Int64 = 200, number_of_steps::Int64 = 0,
    w_κ::Float64 = 0.20, p_neg::Float64 = 0.52, N_tail::Int64 = 5, acf_lags::Int64 = 25,
    seed::Union{Int64,Nothing} = nothing)::MyPortfolioModel

    # initialize - use the stored columns unless we were given others -
    col_map = portfolio.tickers_map;
    if (isnothing(tickers_map) == false)
        col_map = tickers_map;
    end
    rf = portfolio.rf;
    Δt = portfolio.Δt;
    dep = portfolio.dependence;

    # the column of the market: the given one, or the stored one if the portfolio has a market -
    mcol = market_col;
    if (isnothing(mcol) == true && isempty(portfolio.market_ticker) == false)
        if (haskey(col_map, portfolio.market_ticker) == true)
            mcol = col_map[portfolio.market_ticker];
        end
    end

    # tune the hidden Markov model of each ticker on its price column. The own models of a sector factor model are tuned below -
    tuned_marginals = Dict{String,MyJumpHiddenMarkovModel}();
    if (isa(dep, MySectorFactorModel) == false)
        for ticker ∈ portfolio.tickers
            growth_rates = excess_growth_rates(prices[:, col_map[ticker]]; rf = rf, Δt = Δt);
            tuned_marginals[ticker] = tune(portfolio.marginals[ticker], growth_rates; ϵ_range = ϵ_range, λ_range = λ_range,
                number_of_paths = number_of_paths, number_of_steps = number_of_steps, w_κ = w_κ, p_neg = p_neg,
                N_tail = N_tail, acf_lags = acf_lags, seed = seed);
        end
    end

    # tune the market model of a dependence model that has one, if we know the market column; otherwise keep it -
    market_tuned = nothing;
    if (isa(dep, MySingleIndexDependenceModel) || isa(dep, MyHybridSingleIndexDependenceModel) || isa(dep, MySectorFactorModel))
        market_tuned = dep.market_model;
        if (isnothing(mcol) == false)
            market_growth_rates = excess_growth_rates(prices[:, mcol]; rf = rf, Δt = Δt);
            market_tuned = tune(dep.market_model, market_growth_rates; ϵ_range = ϵ_range, λ_range = λ_range,
                number_of_paths = number_of_paths, number_of_steps = number_of_steps, w_κ = w_κ, p_neg = p_neg,
                N_tail = N_tail, acf_lags = acf_lags, seed = seed);
        end
    end

    # single index: a new model with the tuned market model, and everything else shared -
    if (isa(dep, MySingleIndexDependenceModel) == true)
        tuned_dep = MySingleIndexDependenceModel();
        tuned_dep.α = dep.α;
        tuned_dep.β = dep.β;
        tuned_dep.σ_ε = dep.σ_ε;
        tuned_dep.residuals = dep.residuals;
        tuned_dep.residual_method = dep.residual_method;
        tuned_dep.market_model = market_tuned;
        dep = tuned_dep;
    end

    # hybrid: a new model with the tuned market model and the tuned hidden Markov models of the assets -
    if (isa(dep, MyHybridSingleIndexDependenceModel) == true)
        tuned_dep = MyHybridSingleIndexDependenceModel();
        tuned_dep.α = dep.α;
        tuned_dep.β = dep.β;
        tuned_dep.r² = dep.r²;
        tuned_dep.σ_market = dep.σ_market;
        tuned_dep.marginals = tuned_marginals;
        tuned_dep.copula = dep.copula;
        tuned_dep.market_model = market_tuned;
        tuned_dep.tickers = dep.tickers;
        tuned_dep.r2_preserve_threshold = dep.r2_preserve_threshold;
        tuned_dep.idiosyncratic_floor = dep.idiosyncratic_floor;
        dep = tuned_dep;
    end

    # sector factor: tune each sector model and each own model on its level series, with rf = 0.0, because we built the levels with rf = 0.0 -
    if (isa(dep, MySectorFactorModel) == true)
        sector_models = Array{MyJumpHiddenMarkovModel,1}();
        for s ∈ eachindex(dep.sector_models)
            sector_growth_rates = excess_growth_rates(dep.sector_levels[:, s]; rf = 0.0, Δt = Δt);
            push!(sector_models, tune(dep.sector_models[s], sector_growth_rates; ϵ_range = ϵ_range, λ_range = λ_range,
                number_of_paths = number_of_paths, number_of_steps = number_of_steps, w_κ = w_κ, p_neg = p_neg,
                N_tail = N_tail, acf_lags = acf_lags, seed = seed));
        end
        own_models = Array{MyJumpHiddenMarkovModel,1}();
        for i ∈ eachindex(dep.tickers)
            own_growth_rates = excess_growth_rates(dep.own_levels[:, i]; rf = 0.0, Δt = Δt);
            push!(own_models, tune(dep.own_models[i], own_growth_rates; ϵ_range = ϵ_range, λ_range = λ_range,
                number_of_paths = number_of_paths, number_of_steps = number_of_steps, w_κ = w_κ, p_neg = p_neg,
                N_tail = N_tail, acf_lags = acf_lags, seed = seed));
            tuned_marginals[dep.tickers[i]] = own_models[i];
        end

        # tuning changes the long-run mean of each own model, so draw it again. We take the mean of the whole draw at once, as build does -
        ε_center = zeros(length(dep.tickers));
        for i ∈ eachindex(dep.tickers)
            center_path = simulate(own_models[i], dep.center_steps; number_of_paths = 1, seed = dep.center_seed + i);
            ε_center[i] = mean(center_path.growth_rates[:, 1]);
        end

        # a new model with the tuned hidden Markov models and centers, and everything else shared. We do not recompute ψ -
        tuned_dep = MySectorFactorModel();
        tuned_dep.tickers = dep.tickers;
        tuned_dep.sector = dep.sector;
        tuned_dep.sectors = dep.sectors;
        tuned_dep.sector_index = dep.sector_index;
        tuned_dep.α = dep.α;
        tuned_dep.β = dep.β;
        tuned_dep.δ = dep.δ;
        tuned_dep.r² = dep.r²;
        tuned_dep.ψ = dep.ψ;
        tuned_dep.σ²_own = dep.σ²_own;
        tuned_dep.σ²_u = dep.σ²_u;
        tuned_dep.σ²_m = dep.σ²_m;
        tuned_dep.ε_center = ε_center;
        tuned_dep.β_halflife = dep.β_halflife;
        tuned_dep.β_scale = dep.β_scale;
        tuned_dep.market_model = market_tuned;
        tuned_dep.sector_models = sector_models;
        tuned_dep.own_models = own_models;
        tuned_dep.sector_levels = dep.sector_levels;
        tuned_dep.own_levels = dep.own_levels;
        tuned_dep.center_steps = dep.center_steps;
        tuned_dep.center_seed = dep.center_seed;
        dep = tuned_dep;
    end

    # build the tuned portfolio -
    tuned = MyPortfolioModel();
    tuned.tickers = portfolio.tickers;
    tuned.marginals = tuned_marginals;
    tuned.dependence = dep;
    tuned.tickers_map = portfolio.tickers_map;
    tuned.market_ticker = portfolio.market_ticker;
    tuned.rf = portfolio.rf;
    tuned.Δt = portfolio.Δt;

    # return -
    return tuned;
end

"""
    simulate(portfolio::MyPortfolioModel, number_of_steps::Int64; number_of_paths::Int64 = 1000,
        seed::Union{Int64,Nothing} = nothing) -> Dict{String,NamedTuple}

Simulate synthetic paths of every asset of a portfolio, with the dependence between the assets that the dependence model gives.
The way we draw the paths depends on the dependence model:

- Copula: For each path, we draw uniform numbers from the copula and simulate one path of each asset from its hidden Markov model. Then we sort
  the growth rates of each asset and put them in the order of the ranks of its uniform numbers. The states and episodes stay as they were simulated
- Single-index and hybrid: We draw the growth rates with `sample_dependence`. The states are `0` and the episodes are `false`, because these
  paths have no states. A hybrid portfolio has `number_of_steps - 1` days, because `sample_dependence` drops the first day
- Sector factor: We draw a seed for each path from `seed`, and call `sample_dependence` with it. The states of an asset are `0`, and a day
  is an episode if the market, the sector, or the own draw of the asset was in a jump episode. The market gets its own states and episodes

### Arguments
- `portfolio::MyPortfolioModel`: The portfolio to simulate from
- `number_of_steps::Int64`: The number of days in each path. Must be at least `1`

### Returns
- `Dict{String,NamedTuple}`: One entry for each ticker, keyed by ticker. A sector factor portfolio also has an entry for the market. Each entry has three matrices with one row for each day and one column for each path.
    - `growth_rates::Array{Float64,2}`: The simulated growth rates
    - `states::Array{Int64,2}`: The state on each day
    - `episodes::Array{Bool,2}`: `true` if the day is part of a jump episode

### Optional Arguments
- `number_of_paths::Int64`: The number of paths to simulate. Default: `1000`
- `seed::Union{Int64,Nothing}`: The seed of the global random number generator. If `nothing`, the generator is not reseeded. A sector factor portfolio draws the seed of each path from it. Default: `nothing`

### Example
```julia
paths = JumpHMM.simulate(portfolio, 252; number_of_paths = 100, seed = 5660);
```
"""
function simulate(portfolio::MyPortfolioModel, number_of_steps::Int64; number_of_paths::Int64 = 1000,
    seed::Union{Int64,Nothing} = nothing)::Dict{String,NamedTuple}

    # check the number of days -
    if (number_of_steps < 1)
        throw(ArgumentError("number_of_steps must be at least 1, got $(number_of_steps)"));
    end

    # check the dependence model, before we touch the random number generator -
    dep = portfolio.dependence;
    is_known = (isa(dep, MyGaussianCopulaModel) || isa(dep, MyStudentTCopulaModel) || isa(dep, MyVineCopulaModel) ||
        isa(dep, MySingleIndexDependenceModel) || isa(dep, MyHybridSingleIndexDependenceModel) || isa(dep, MySectorFactorModel));
    if (is_known == false)
        throw(ArgumentError("unsupported dependence model: $(typeof(dep))"));
    end

    # seed the global random number generator, if we were given a seed -
    if (isnothing(seed) == false)
        Random.seed!(seed);
    end

    # initialize -
    tickers = portfolio.tickers;
    number_of_assets = length(tickers);
    results = Dict{String,NamedTuple}();

    # copula: simulate each asset on its own, then put its growth rates in the order the copula gives -
    if (isa(dep, MyGaussianCopulaModel) || isa(dep, MyStudentTCopulaModel) || isa(dep, MyVineCopulaModel))
        for ticker ∈ tickers
            results[ticker] = (growth_rates = Array{Float64,2}(undef, number_of_steps, number_of_paths),
                states = Array{Int64,2}(undef, number_of_steps, number_of_paths),
                episodes = Array{Bool,2}(undef, number_of_steps, number_of_paths));
        end
        for p ∈ 1:number_of_paths

            # draw the uniform numbers of every asset for this path -
            U = sample_dependence(dep, number_of_steps);
            for j ∈ 1:number_of_assets
                ticker = tickers[j];

                # one path of this asset: sort its growth rates, then place them by the ranks of the uniform numbers -
                path = simulate(portfolio.marginals[ticker], number_of_steps; number_of_paths = 1);
                sorted_growth_rates = sort(path.growth_rates[:, 1]);
                ranks = ordinalrank(U[:, j]);
                results[ticker].growth_rates[:, p] = sorted_growth_rates[ranks];
                results[ticker].states[:, p] = path.states[:, 1];
                results[ticker].episodes[:, p] = path.episodes[:, 1];
            end
        end

    # single index: the growth rates come from the dependence model, and the paths have no states or episodes -
    elseif (isa(dep, MySingleIndexDependenceModel) == true)
        for ticker ∈ tickers
            results[ticker] = (growth_rates = Array{Float64,2}(undef, number_of_steps, number_of_paths),
                states = zeros(Int64, number_of_steps, number_of_paths),
                episodes = fill(false, number_of_steps, number_of_paths));
        end
        for p ∈ 1:number_of_paths
            sample = sample_dependence(dep, number_of_steps);
            for j ∈ 1:number_of_assets
                results[tickers[j]].growth_rates[:, p] = sample[:, j];
            end
        end

    # hybrid: the same, but the sample has one day fewer -
    elseif (isa(dep, MyHybridSingleIndexDependenceModel) == true)
        number_of_days = number_of_steps - 1;
        for ticker ∈ dep.tickers
            results[ticker] = (growth_rates = Array{Float64,2}(undef, number_of_days, number_of_paths),
                states = zeros(Int64, number_of_days, number_of_paths),
                episodes = fill(false, number_of_days, number_of_paths));
        end
        for p ∈ 1:number_of_paths
            sample = sample_dependence(dep, number_of_steps);
            for j ∈ eachindex(dep.tickers)
                results[dep.tickers[j]].growth_rates[:, p] = sample[:, j];
            end
        end

    # sector factor: a different seed for each path, drawn from the seed we were given, or from the system entropy source -
    elseif (isa(dep, MySectorFactorModel) == true)
        rng = Random.RandomDevice();
        if (isnothing(seed) == false)
            rng = Xoshiro(seed);
        end
        path_seeds = StatsBase.sample(rng, 0:(10^8 - 1), number_of_paths; replace = false);
        for ticker ∈ vcat(tickers, portfolio.market_ticker)
            results[ticker] = (growth_rates = Array{Float64,2}(undef, number_of_steps, number_of_paths),
                states = zeros(Int64, number_of_steps, number_of_paths),
                episodes = fill(false, number_of_steps, number_of_paths));
        end
        for p ∈ 1:number_of_paths
            sample = sample_dependence(dep, number_of_steps; seed = path_seeds[p]);

            # each asset has states 0 and its own episodes; the market has the states and episodes of its model -
            for j ∈ eachindex(dep.tickers)
                results[dep.tickers[j]].growth_rates[:, p] = sample.G[:, j];
                results[dep.tickers[j]].episodes[:, p] = sample.name_episodes[:, j];
            end
            results[portfolio.market_ticker].growth_rates[:, p] = sample.g_m;
            results[portfolio.market_ticker].states[:, p] = sample.m_states;
            results[portfolio.market_ticker].episodes[:, p] = sample.episodes;
        end
    else
        throw(ArgumentError("unsupported dependence model: $(typeof(dep))"));
    end

    # return -
    return results;
end

"""
    validate(portfolio::MyPortfolioModel, prices::Array{Float64,2}; tickers_map::Union{Dict{String,Int64},Nothing} = nothing,
        number_of_paths::Int64 = 1000, α::Float64 = 0.05, acf_lags::Int64 = 25,
        seed::Union{Int64,Nothing} = nothing) -> Dict{String,MyValidationReportModel}

Validate the hidden Markov model of each asset of a portfolio against the growth rates of that asset, with `validate`. We turn each price column
into growth rates with `portfolio.rf` and `portfolio.Δt`. A sector factor portfolio validates each own model against its own level series
instead, with `rf = 0.0`, because the own model describes the residual of the asset after its sector term.

### Arguments
- `portfolio::MyPortfolioModel`: The portfolio to validate
- `prices::Array{Float64,2}`: The prices, one row per day. The columns are the ones the portfolio was built from

### Returns
- `Dict{String,MyValidationReportModel}`: The report of each ticker, keyed by ticker

### Optional Arguments
- `tickers_map::Union{Dict{String,Int64},Nothing}`: The column of each ticker in `prices`. If `nothing`, we use `portfolio.tickers_map`. Default: `nothing`
- `number_of_paths::Int64`: The number of paths to simulate for each asset. Default: `1000`
- `α::Float64`: The significance level of the tests. A path passes a test if its p-value is above `α`. Default: `0.05`
- `acf_lags::Int64`: The number of lags of the autocorrelation. We shorten it if the data is too short. Default: `25`
- `seed::Union{Int64,Nothing}`: The seed of the global random number generator, set before each asset is validated. If `nothing`, the generator is not reseeded. Default: `nothing`

### Example
```julia
reports = JumpHMM.validate(portfolio, P; number_of_paths = 100, seed = 18);
```
"""
function validate(portfolio::MyPortfolioModel, prices::Array{Float64,2}; tickers_map::Union{Dict{String,Int64},Nothing} = nothing,
    number_of_paths::Int64 = 1000, α::Float64 = 0.05, acf_lags::Int64 = 25,
    seed::Union{Int64,Nothing} = nothing)::Dict{String,MyValidationReportModel}

    # initialize - use the stored columns unless we were given others -
    col_map = portfolio.tickers_map;
    if (isnothing(tickers_map) == false)
        col_map = tickers_map;
    end
    dep = portfolio.dependence;
    reports = Dict{String,MyValidationReportModel}();

    # sector factor: check each own model on its level series -
    if (isa(dep, MySectorFactorModel) == true)
        for i ∈ eachindex(dep.tickers)
            growth_rates = excess_growth_rates(dep.own_levels[:, i]; rf = 0.0, Δt = portfolio.Δt);
            reports[dep.tickers[i]] = validate(dep.own_models[i], growth_rates; number_of_paths = number_of_paths, α = α,
                acf_lags = acf_lags, seed = seed);
        end
    else
        # every other dependence model: check the hidden Markov model of each ticker on its price column -
        for ticker ∈ portfolio.tickers
            growth_rates = excess_growth_rates(prices[:, col_map[ticker]]; rf = portfolio.rf, Δt = portfolio.Δt);
            reports[ticker] = validate(portfolio.marginals[ticker], growth_rates; number_of_paths = number_of_paths, α = α,
                acf_lags = acf_lags, seed = seed);
        end
    end

    # return -
    return reports;
end
