"""
    wasserstein_distance(x::Array{Float64,1}, y::Array{Float64,1}) -> Float64

Compute the Wasserstein-1 distance between two samples. If the samples have the same length, we sort both and take the mean absolute
difference. Otherwise, we compare the two samples at the same number of evenly spaced quantiles.

### Arguments
- `x::Array{Float64,1}`: The first sample
- `y::Array{Float64,1}`: The second sample

### Returns
- `Float64`: The Wasserstein-1 distance between `x` and `y`

### Example
```julia
d = JumpHMM.wasserstein_distance([0.1, -0.2, 0.3], [0.0, 0.2, 0.4]);
```
"""
function wasserstein_distance(x::Array{Float64,1}, y::Array{Float64,1})::Float64

    # sort both samples -
    sx = sort(x);
    sy = sort(y);

    # same length: compare the sorted values one by one -
    if (length(sx) == length(sy))
        return mean(abs.(sx .- sy));
    end

    # different lengths: compare the two samples at the same number of evenly spaced quantiles -
    number_of_quantiles = max(length(sx), length(sy));
    probabilities = range(0, 1, length = number_of_quantiles);
    qx = Array{Float64,1}(undef, number_of_quantiles);
    qy = Array{Float64,1}(undef, number_of_quantiles);
    for (i, p) ∈ enumerate(probabilities)
        qx[i] = quantile(sx, p);
        qy[i] = quantile(sy, p);
    end
    return mean(abs.(qx .- qy));
end

"""
    hellinger_distance(x::Array{Float64,1}, y::Array{Float64,1}) -> Float64

Compute the Hellinger distance between two samples. We put both samples into the same histogram bins, scale each histogram to sum to one,
and compare the square roots of the bin probabilities. If both samples are the same single value, the distance is `0.0`.

### Arguments
- `x::Array{Float64,1}`: The first sample
- `y::Array{Float64,1}`: The second sample

### Returns
- `Float64`: The Hellinger distance between `x` and `y`, between `0.0` and `1.0`

### Example
```julia
d = JumpHMM.hellinger_distance([0.1, -0.2, 0.3], [0.0, 0.2, 0.4]);
```
"""
function hellinger_distance(x::Array{Float64,1}, y::Array{Float64,1})::Float64

    # the range that covers both samples -
    lowest = min(minimum(x), minimum(y));
    highest = max(maximum(x), maximum(y));

    # both samples are the same single value, so the distributions are identical -
    if (lowest == highest)
        return 0.0;
    end

    # common bin edges for both samples -
    number_of_bins = max(30, round(Int, sqrt(max(length(x), length(y)))));
    edges = range(lowest, highest, length = number_of_bins + 1);

    # histogram each sample, and scale each histogram to sum to one -
    hx = StatsBase.fit(Histogram, x, edges);
    hy = StatsBase.fit(Histogram, y, edges);
    px = hx.weights ./ sum(hx.weights);
    py = hy.weights ./ sum(hy.weights);

    # the Hellinger distance -
    return sqrt(0.5 * sum((sqrt.(px) .- sqrt.(py)).^2));
end

"""
    validate(model::MyJumpHiddenMarkovModel, growth_rates::Array{Float64,1}; number_of_paths::Int64 = 1000, α::Float64 = 0.05,
        acf_lags::Int64 = 25, seed::Union{Int64,Nothing} = nothing) -> MyValidationReportModel

Simulate paths from the model and compare each path with the data. For each path, we compute the Kolmogorov-Smirnov and Anderson-Darling
p-values, the Wasserstein and Hellinger distances, the mean absolute error in the autocorrelation of the absolute growth rates, and the
excess kurtosis. Each path has as many days as `growth_rates`.

### Arguments
- `model::MyJumpHiddenMarkovModel`: The model to validate
- `growth_rates::Array{Float64,1}`: The growth rates of the data, for example from `excess_growth_rates`

### Returns
- `MyValidationReportModel`: The results for each path, and their summaries over the paths

### Optional Arguments
- `number_of_paths::Int64`: The number of paths to simulate. Default: `1000`
- `α::Float64`: The significance level of the tests. A path passes a test if its p-value is above `α`. Default: `0.05`
- `acf_lags::Int64`: The number of lags of the autocorrelation. We shorten it if the data is too short. Default: `25`
- `seed::Union{Int64,Nothing}`: The seed of the global random number generator. If `nothing`, the generator is not reseeded. Default: `nothing`

### Example
```julia
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 100, ν = 5.0, ϵ = 1e-3, λ = 90.0));
report = JumpHMM.validate(model, G; number_of_paths = 20, seed = 9);
```
"""
function validate(model::MyJumpHiddenMarkovModel, growth_rates::Array{Float64,1}; number_of_paths::Int64 = 1000,
    α::Float64 = 0.05, acf_lags::Int64 = 25, seed::Union{Int64,Nothing} = nothing)::MyValidationReportModel

    # initialize - the data has as many days as the growth rates, and the autocorrelation needs fewer lags than days -
    number_of_steps = length(growth_rates);
    acf_lags = min(acf_lags, number_of_steps - 1);
    lags = collect(1:acf_lags);
    acf_emp = autocor(abs.(growth_rates), lags);
    kurtosis_emp = kurtosis(growth_rates);

    # simulate the paths; path i is column i -
    paths = simulate(model, number_of_steps; number_of_paths = number_of_paths, seed = seed);

    # initialize - one value per path for each measure -
    ks_pvalue = Array{Float64,1}(undef, number_of_paths);
    ad_pvalue = Array{Float64,1}(undef, number_of_paths);
    wasserstein = Array{Float64,1}(undef, number_of_paths);
    hellinger = Array{Float64,1}(undef, number_of_paths);
    acf_mae = Array{Float64,1}(undef, number_of_paths);
    kurtosis_sim = Array{Float64,1}(undef, number_of_paths);

    # main loop - compare each path with the data -
    for i ∈ 1:number_of_paths
        simulated_growth_rates = paths.growth_rates[:, i];

        # the two-sample tests -
        ks_pvalue[i] = pvalue(ApproximateTwoSampleKSTest(growth_rates, simulated_growth_rates));
        ad_pvalue[i] = pvalue(KSampleADTest(growth_rates, simulated_growth_rates));

        # the distances -
        wasserstein[i] = wasserstein_distance(growth_rates, simulated_growth_rates);
        hellinger[i] = hellinger_distance(growth_rates, simulated_growth_rates);

        # the error in the autocorrelation of the absolute growth rates, and the kurtosis -
        acf_sim = autocor(abs.(simulated_growth_rates), lags);
        acf_mae[i] = mean(abs.(acf_emp .- acf_sim));
        kurtosis_sim[i] = kurtosis(simulated_growth_rates);
    end

    # summarize - add the values one at a time in path order, then divide by the number of paths.
    # We do not call mean on the vectors, because it can add in a different order, and the last digit can change -
    ks_passed = 0;
    ad_passed = 0;
    total_acf_mae = 0.0;
    total_wasserstein = 0.0;
    total_hellinger = 0.0;
    total_kurtosis = 0.0;
    for i ∈ 1:number_of_paths
        if (ks_pvalue[i] > α)
            ks_passed += 1;
        end
        if (ad_pvalue[i] > α)
            ad_passed += 1;
        end
        total_acf_mae += acf_mae[i];
        total_wasserstein += wasserstein[i];
        total_hellinger += hellinger[i];
        total_kurtosis += kurtosis_sim[i];
    end

    # fill the report -
    report = MyValidationReportModel();
    report.ks_pvalue = ks_pvalue;
    report.ad_pvalue = ad_pvalue;
    report.wasserstein = wasserstein;
    report.hellinger = hellinger;
    report.acf_mae = acf_mae;
    report.kurtosis_sim = kurtosis_sim;
    report.kurtosis_emp = kurtosis_emp;
    report.ks_pass_rate = ks_passed / number_of_paths;
    report.ad_pass_rate = ad_passed / number_of_paths;
    report.mean_acf_mae = total_acf_mae / number_of_paths;
    report.mean_wasserstein = total_wasserstein / number_of_paths;
    report.mean_hellinger = total_hellinger / number_of_paths;
    report.mean_kurtosis = total_kurtosis / number_of_paths;
    report.α = α;

    # return -
    return report;
end
