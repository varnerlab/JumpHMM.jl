"""
    sample_dependence(model::MySingleIndexDependenceModel, number_of_steps::Int64) -> Array{Float64,2}

Simulate synthetic growth rates for the assets of a single-index model. We simulate one path of market growth rates from the market model,
and then compute the growth rates of each asset as `g_i = α_i + β_i g_m + ε_i`. With the `:bootstrap` residual method, each day's `ε_i` is
the residual of one training day, and all assets share that day. With the `:gaussian` method, each `ε_i` is a normal draw with standard deviation `σ_ε[i]`.
The draws come from the global random number generator. To repeat a sample, call `Random.seed!` first.

### Arguments
- `model::MySingleIndexDependenceModel`: The single-index model
- `number_of_steps::Int64`: The number of days to simulate

### Returns
- `Array{Float64,2}`: The simulated growth rates, with `number_of_steps` rows (days) and one column per asset

### Example
```julia
G = JumpHMM.sample_dependence(model, 252);
```
"""
function sample_dependence(model::MySingleIndexDependenceModel, number_of_steps::Int64)::Array{Float64,2}

    # initialize: simulate one market path -
    market_path = simulate(model.market_model, number_of_steps; number_of_paths = 1);
    market_growth_rates = market_path.growth_rates[:, 1];
    number_of_assets = length(model.α);
    number_of_training_days = size(model.residuals, 1);
    growth_rates = Array{Float64,2}(undef, number_of_steps, number_of_assets);

    # bootstrap: every asset uses the residuals of the same training day -
    if model.residual_method == :bootstrap
        days = rand(1:number_of_training_days, number_of_steps);
        for i ∈ 1:number_of_assets
            for t ∈ 1:number_of_steps
                growth_rates[t, i] = model.α[i] + model.β[i] * market_growth_rates[t] + model.residuals[days[t], i];
            end
        end
    else
        # gaussian: every asset draws a normal residual -
        for i ∈ 1:number_of_assets
            for t ∈ 1:number_of_steps
                growth_rates[t, i] = model.α[i] + model.β[i] * market_growth_rates[t] + model.σ_ε[i] * randn();
            end
        end
    end

    # return -
    return growth_rates;
end

"""
    sample_dependence(model::MyHybridSingleIndexDependenceModel, number_of_steps::Int64) -> Array{Float64,2}

Simulate synthetic growth rates for the assets of a hybrid single-index model. We simulate a market path and one path from the hidden Markov model
of each asset, and drop the first day of every path. We scale the path of each asset into its idiosyncratic part, and reorder that part so that
its ranks follow the ranks of a draw from the copula. Then we compute `g_i = α_i + β_i g_m + ε_i`.

If `r²` of an asset is at least `r2_preserve_threshold`, we set the variance of the scaled path from the variance of the market in the data, so that the expected R² of the simulated asset matches `r²`.
Otherwise, we scale the path to keep the variance of the hidden Markov model of the asset, and we lower `β` if the market part would take
more than `1 - idiosyncratic_floor` of that variance.

The draws come from the global random number generator. To repeat a sample, call `Random.seed!` first.

### Arguments
- `model::MyHybridSingleIndexDependenceModel`: The hybrid single-index model
- `number_of_steps::Int64`: The number of days to simulate before the first day is dropped

### Returns
- `Array{Float64,2}`: The simulated growth rates, with `number_of_steps - 1` rows (days) and one column per asset. The first simulated day is dropped

### Example
```julia
G = JumpHMM.sample_dependence(model, 252);
```
"""
function sample_dependence(model::MyHybridSingleIndexDependenceModel, number_of_steps::Int64)::Array{Float64,2}

    # initialize -
    number_of_assets = length(model.tickers);
    σ²_market = model.σ_market^2;
    f = model.idiosyncratic_floor;
    threshold = model.r2_preserve_threshold;

    # simulate the market path, and drop its first day -
    market_path = simulate(model.market_model, number_of_steps; number_of_paths = 1);
    market_growth_rates = market_path.growth_rates[2:end, 1];
    number_of_days = length(market_growth_rates);

    # draw the copula uniform numbers, and drop the first day -
    U = sample_dependence(model.copula, number_of_steps);
    U_trimmed = U[2:end, :];

    # main loop: build the growth rates of each asset -
    growth_rates = Array{Float64,2}(undef, number_of_days, number_of_assets);
    for k ∈ 1:number_of_assets
        α_k = model.α[k];
        β_k = model.β[k];
        r²_k = model.r²[k];

        # draw a path from the hidden Markov model of the asset, and drop its first day -
        asset_path = simulate(model.marginals[model.tickers[k]], number_of_steps; number_of_paths = 1);
        asset_growth_rates = asset_path.growth_rates[2:end, 1];
        σ²_asset = var(asset_growth_rates);

        # scale the path into the idiosyncratic part -
        β_effective = β_k;
        ε_scaled = zeros(number_of_days);
        if r²_k >= threshold

            # keep the R² of the asset: the target variance of the idiosyncratic part comes from r² -
            σ²_ε_target = 0.0;
            if r²_k < 1.0 - 1e-12
                σ²_ε_target = β_k^2 * σ²_market * (1.0 - r²_k) / r²_k;
            end
            if σ²_asset > 0.0 && σ²_ε_target > 0.0
                ε_scaled = sqrt(σ²_ε_target / σ²_asset) .* asset_growth_rates;
            end
        else

            # keep the variance of the asset, and lower β if the market part is too large -
            ρ = β_k^2 * σ²_market / max(σ²_asset, 1e-30);
            s² = 1.0 - ρ;
            if ρ > 1.0 - f
                β_effective = sign(β_k) * sqrt((1.0 - f) * σ²_asset / σ²_market);
                s² = f;
            end
            ε_scaled = sqrt(s²) .* asset_growth_rates;
        end

        # reorder the idiosyncratic part so that its ranks follow the copula draw -
        sorted_ε = sort(ε_scaled);
        copula_ranks = ordinalrank(U_trimmed[:, k]);
        ε_reordered = sorted_ε[copula_ranks];

        # compose the growth rates of the asset -
        growth_rates[:, k] = α_k .+ β_effective .* market_growth_rates .+ ε_reordered;
    end

    # return -
    return growth_rates;
end
