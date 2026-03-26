"""
    fit(::Type{GaussianCopula}, returns) → GaussianCopula

Fit a Gaussian copula from an (n_obs × n_assets) matrix of excess growth rates.
"""
function fit(::Type{GaussianCopula}, returns::AbstractMatrix{Float64})
    U = _pit(returns)
    Σ = cor(U)
    return GaussianCopula(Σ)
end

"""
    fit(::Type{StudentTCopula}, returns) → StudentTCopula

Fit a Student-t copula. Estimates correlation from PIT values and ν via profile MLE.
"""
function fit(::Type{StudentTCopula}, returns::AbstractMatrix{Float64})
    U = _pit(returns)
    Σ = cor(U)
    d = size(returns, 2)

    # profile MLE over candidate ν values
    best_ll = -Inf
    best_ν = 5.0
    for ν_cand in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 30.0]
        ll = _copula_loglik(U, Σ, ν_cand)
        if ll > best_ll
            best_ll = ll
            best_ν = ν_cand
        end
    end

    return StudentTCopula(Σ, best_ν)
end

"""
    sample_dependence(gc::GaussianCopula, n) → Matrix{Float64}

Draw n samples from the Gaussian copula. Returns (n × d) uniform matrix.
"""
function sample_dependence(gc::GaussianCopula, n::Int)
    d = size(gc.Σ, 1)
    Z = rand(MvNormal(zeros(d), gc.Σ), n)'  # n × d
    U = cdf.(Normal(), Z)
    return U
end

"""
    sample_dependence(tc::StudentTCopula, n) → Matrix{Float64}

Draw n samples from the Student-t copula. Returns (n × d) uniform matrix.
"""
function sample_dependence(tc::StudentTCopula, n::Int)
    d = size(tc.Σ, 1)
    # MvTDist from Distributions.jl
    dist = MvTDist(tc.ν, zeros(d), PDMats.PDMat(Symmetric(tc.Σ)))
    Z = rand(dist, n)'  # n × d
    U = cdf.(TDist(tc.ν), Z)
    return U
end

# --- Private helpers -------------------------------------------------------

function _pit(data::AbstractMatrix{Float64})
    n, d = size(data)
    U = Matrix{Float64}(undef, n, d)
    for j in 1:d
        ranks = ordinalrank(data[:, j])
        U[:, j] = ranks ./ (n + 1)  # avoid exact 0 and 1
    end
    return U
end

function _copula_loglik(U::Matrix{Float64}, Σ::Matrix{Float64}, ν::Float64)
    n, d = size(U)
    # transform uniform marginals to t-quantiles
    X = quantile.(TDist(ν), U)

    ll = 0.0
    try
        Σ_inv = inv(Σ)
        log_det_Σ = logdet(Σ)
        for i in 1:n
            x = X[i, :]
            quad = dot(x, Σ_inv * x)
            # multivariate t density / product of marginal t densities
            log_joint = loggamma((ν + d) / 2) - loggamma(ν / 2) -
                        (d / 2) * log(ν * π) - 0.5 * log_det_Σ -
                        ((ν + d) / 2) * log(1 + quad / ν)
            log_marginals = 0.0
            for j in 1:d
                log_marginals += logpdf(TDist(ν), x[j])
            end
            ll += log_joint - log_marginals
        end
    catch
        return -Inf
    end

    return ll
end
