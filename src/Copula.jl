"""
    sample_dependence(model::MyGaussianCopulaModel, number_of_steps::Int64) -> Array{Float64,2}

Draw dependent uniform numbers from a Gaussian copula. We draw correlated standard normal numbers with the correlation matrix
of the model, and then map each normal number to a uniform number with the standard normal cumulative distribution function.
The draws come from the global random number generator. To repeat a sample, call `Random.seed!` first.

### Arguments
- `model::MyGaussianCopulaModel`: The Gaussian copula
- `number_of_steps::Int64`: The number of rows to draw

### Returns
- `Array{Float64,2}`: The uniform numbers, with `number_of_steps` rows (days) and one column per asset. Every value is strictly between `0` and `1`

### Example
```julia
U = JumpHMM.sample_dependence(copula, 250);
```
"""
function sample_dependence(model::MyGaussianCopulaModel, number_of_steps::Int64)::Array{Float64,2}

    # initialize -
    d = size(model.Σ, 1);

    # draw correlated standard normal numbers, one row per step -
    Z = rand(MvNormal(zeros(d), model.Σ), number_of_steps)';

    # map each normal number to a uniform number -
    U = cdf.(Normal(), Z);

    # return -
    return U;
end

"""
    sample_dependence(model::MyStudentTCopulaModel, number_of_steps::Int64) -> Array{Float64,2}

Draw dependent uniform numbers from a Student-t copula. We draw correlated Student-t numbers with the correlation matrix and the
degrees of freedom of the model, and then map each number to a uniform number with the Student-t cumulative distribution function.
The draws come from the global random number generator. To repeat a sample, call `Random.seed!` first.

### Arguments
- `model::MyStudentTCopulaModel`: The Student-t copula
- `number_of_steps::Int64`: The number of rows to draw

### Returns
- `Array{Float64,2}`: The uniform numbers, with `number_of_steps` rows (days) and one column per asset. Every value is strictly between `0` and `1`

### Example
```julia
U = JumpHMM.sample_dependence(copula, 250);
```
"""
function sample_dependence(model::MyStudentTCopulaModel, number_of_steps::Int64)::Array{Float64,2}

    # initialize -
    d = size(model.Σ, 1);
    dist = MvTDist(model.ν, zeros(d), PDMats.PDMat(Symmetric(model.Σ)));

    # draw correlated Student-t numbers, one row per step -
    Z = rand(dist, number_of_steps)';

    # map each Student-t number to a uniform number -
    U = cdf.(TDist(model.ν), Z);

    # return -
    return U;
end

"""
    _uniform_ranks(data::Array{Float64,2}) -> Array{Float64,2}

Replace each column of `data` by its ranks divided by the number of rows plus one. The result is strictly between `0` and `1`
and keeps only the order of the values in each column.

### Arguments
- `data::Array{Float64,2}`: The growth rates, one row per day and one column per asset

### Returns
- `Array{Float64,2}`: The uniform numbers, with the same size as `data`
"""
function _uniform_ranks(data::Array{Float64,2})::Array{Float64,2}

    # initialize -
    n, d = size(data);
    U = Matrix{Float64}(undef, n, d);

    # rank each column, and divide by n + 1 to stay away from 0 and 1 -
    for j ∈ 1:d
        ranks = ordinalrank(data[:, j]);
        U[:, j] = ranks ./ (n + 1);
    end

    # return -
    return U;
end

"""
    _valid_correlation(Σ::Array{Float64,2}; tol::Float64 = 1e-8) -> Array{Float64,2}

Return a correlation matrix that can be factored. If `Σ` is already positive definite, we return it as a symmetric matrix.
Otherwise we raise every eigenvalue below `tol` up to `tol` and rescale the matrix to a unit diagonal.

### Arguments
- `Σ::Array{Float64,2}`: The correlation matrix to check

### Returns
- `Array{Float64,2}`: A symmetric, positive definite correlation matrix

### Optional Arguments
- `tol::Float64`: The smallest eigenvalue to keep. The default is `1e-8`
"""
function _valid_correlation(Σ::Array{Float64,2}; tol::Float64 = 1e-8)::Array{Float64,2}

    # a positive definite matrix is already valid -
    Σs = Symmetric(Σ);
    if isposdef(Σs) == true
        return Matrix(Σs);
    end

    # raise the small eigenvalues to tol -
    F = eigen(Σs);
    λ = max.(F.values, tol);
    Σ_fixed = F.vectors * Diagonal(λ) * F.vectors';

    # rescale to a unit diagonal -
    D = Diagonal(1.0 ./ sqrt.(diag(Σ_fixed)));
    return Matrix(Symmetric(D * Σ_fixed * D));
end

"""
    _studentt_copula_loglik(U::Array{Float64,2}, Σ::Array{Float64,2}, ν::Float64) -> Float64

Compute the log-likelihood of the uniform numbers `U` under a Student-t copula with correlation matrix `Σ` and `ν` degrees of
freedom. Each row contributes the log of the joint Student-t density minus the log of the product of the marginal densities.
If `Σ` cannot be inverted, or the log-density cannot be evaluated for any other reason, the function returns `-Inf`.

### Arguments
- `U::Array{Float64,2}`: The uniform numbers, one row per day and one column per asset
- `Σ::Array{Float64,2}`: The correlation matrix
- `ν::Float64`: The degrees of freedom

### Returns
- `Float64`: The log-likelihood
"""
function _studentt_copula_loglik(U::Array{Float64,2}, Σ::Array{Float64,2}, ν::Float64)::Float64

    # initialize -
    n, d = size(U);
    X = quantile.(TDist(ν), U);
    ll = 0.0;

    # add the log-density of each row -
    try
        Σ_inv = inv(Σ);
        log_det_Σ = logdet(Σ);
        for i ∈ 1:n
            x = X[i, :];
            quad = dot(x, Σ_inv * x);

            # log of the joint multivariate Student-t density -
            log_joint = loggamma((ν + d) / 2) - loggamma(ν / 2) -
                        (d / 2) * log(ν * π) - 0.5 * log_det_Σ -
                        ((ν + d) / 2) * log(1 + quad / ν);

            # log of the product of the marginal Student-t densities -
            log_marginals = 0.0;
            for j ∈ 1:d
                log_marginals += logpdf(TDist(ν), x[j]);
            end
            ll += log_joint - log_marginals;
        end
    catch
        return -Inf;
    end

    # return -
    return ll;
end
