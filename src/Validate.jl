"""
    validate(model, prices; kwargs...) → ValidationReport

Statistical validation: simulate paths and compare against empirical data.
"""
function validate(model::JumpHiddenMarkovModel,
                  prices::AbstractVector{<:Real};
                  n_paths::Int=1000,
                  α::Float64=0.05,
                  acf_lags::Int=25,
                  rf::Float64=model.rf,
                  seed::Union{Int,Nothing}=nothing)

    G_emp = excess_growth_rates(prices; rf=rf, dt=model.dt)
    n_steps = length(G_emp)
    lags = collect(1:acf_lags)
    acf_emp = autocor(abs.(G_emp), lags)
    κ_emp = kurtosis(G_emp)

    result = simulate(model, n_steps; n_paths=n_paths, seed=seed)

    path_results = Vector{PathTestResult}(undef, n_paths)
    for i in 1:n_paths
        G_sim = result.paths[i].observations

        # KS test
        ks_p = pvalue(ApproximateTwoSampleKSTest(G_emp, G_sim))

        # Anderson-Darling test
        ad_p = pvalue(KSampleADTest(G_emp, G_sim))

        # Wasserstein-1 distance
        w1 = _wasserstein1(G_emp, G_sim)

        # Hellinger distance
        h = _hellinger(G_emp, G_sim)

        # ACF-MAE
        acf_sim = autocor(abs.(G_sim), lags)
        acf_mae = mean(abs.(acf_emp .- acf_sim))

        # Kurtosis
        κ_sim = kurtosis(G_sim)

        path_results[i] = PathTestResult(ks_p, ad_p, w1, h, acf_mae, κ_sim, κ_emp)
    end

    ks_pass = mean(r.ks_pvalue > α for r in path_results)
    ad_pass = mean(r.ad_pvalue > α for r in path_results)
    mean_acf = mean(r.acf_mae for r in path_results)
    mean_w1 = mean(r.wasserstein for r in path_results)
    mean_h = mean(r.hellinger for r in path_results)
    mean_κ = mean(r.kurtosis_sim for r in path_results)

    return ValidationReport(path_results, ks_pass, ad_pass,
                            mean_acf, mean_w1, mean_h, mean_κ, α)
end

# --- Private helpers -------------------------------------------------------

function _wasserstein1(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    sx = sort(x)
    sy = sort(y)
    if length(sx) == length(sy)
        return mean(abs.(sx .- sy))
    else
        # interpolate to common quantile grid
        n = max(length(sx), length(sy))
        qx = [quantile(sx, p) for p in range(0, 1, length=n)]
        qy = [quantile(sy, p) for p in range(0, 1, length=n)]
        return mean(abs.(qx .- qy))
    end
end

function _hellinger(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    # common bin edges
    lo = min(minimum(x), minimum(y))
    hi = max(maximum(x), maximum(y))
    n_bins = max(30, round(Int, sqrt(max(length(x), length(y)))))
    edges = range(lo, hi, length=n_bins + 1)

    hx = StatsBase.fit(Histogram, x, edges)
    hy = StatsBase.fit(Histogram, y, edges)

    # normalize to probability distributions
    px = hx.weights ./ sum(hx.weights)
    py = hy.weights ./ sum(hy.weights)

    # Hellinger distance
    return sqrt(0.5 * sum((sqrt.(px) .- sqrt.(py)).^2))
end
