# the five bivariate copula families that we try on every edge of the vine -
const _BICOP_FAMILIES = [MyGaussianBivariateCopulaModel, MyStudentTBivariateCopulaModel, MyClaytonBivariateCopulaModel,
    MyGumbelBivariateCopulaModel, MyFrankBivariateCopulaModel];

"""
    _clamp01(x::Float64) -> Float64

Clamp `x` to the interval from `1e-10` to `1 - 1e-10`, so that logarithms and quantiles of the result stay finite.

### Arguments
- `x::Float64`: The number to clamp

### Returns
- `Float64`: The clamped number

### Example
```julia
x = JumpHMM._clamp01(1.0); # 1 - 1e-10
```
"""
function _clamp01(x::Float64)::Float64
    return clamp(x, 1e-10, 1.0 - 1e-10);
end

"""
    _bicop_hfunc(c::MyGaussianBivariateCopulaModel, u::Float64, v::Float64) -> Float64

Evaluate the h-function of a Gaussian copula. The h-function is the conditional distribution function of the first asset given the second asset, the derivative of the copula with respect to `v`. We map `u` and `v` to standard normal quantiles, and evaluate the closed form with the normal cumulative distribution function. The result is clamped away from `0` and `1`.

### Arguments
- `c::MyGaussianBivariateCopulaModel`: The Gaussian copula
- `u::Float64`: The uniform number of the first asset
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The conditional probability, strictly between `0` and `1`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyGaussianBivariateCopulaModel, (ρ = 0.5,));
w = JumpHMM._bicop_hfunc(c, 0.3, 0.7);
```
"""
function _bicop_hfunc(c::MyGaussianBivariateCopulaModel, u::Float64, v::Float64)::Float64

    # initialize -
    x = quantile(Normal(), _clamp01(u));
    y = quantile(Normal(), _clamp01(v));
    ρ = c.ρ;

    # return -
    return _clamp01(cdf(Normal(), (x - ρ * y) / sqrt(1.0 - ρ^2)));
end

"""
    _bicop_hfunc(c::MyStudentTBivariateCopulaModel, u::Float64, v::Float64) -> Float64

Evaluate the h-function of a Student-t copula. The h-function is the conditional distribution function of the first asset given the second asset, the derivative of the copula with respect to `v`. We map `u` and `v` to Student-t quantiles, and evaluate the closed form with the Student-t cumulative distribution function with `ν + 1` degrees of freedom. The result is clamped away from `0` and `1`.

### Arguments
- `c::MyStudentTBivariateCopulaModel`: The Student-t copula
- `u::Float64`: The uniform number of the first asset
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The conditional probability, strictly between `0` and `1`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyStudentTBivariateCopulaModel, (ρ = 0.5, ν = 5.0));
w = JumpHMM._bicop_hfunc(c, 0.3, 0.7);
```
"""
function _bicop_hfunc(c::MyStudentTBivariateCopulaModel, u::Float64, v::Float64)::Float64

    # initialize -
    ν = c.ν;
    ρ = c.ρ;
    x = quantile(TDist(ν), _clamp01(u));
    y = quantile(TDist(ν), _clamp01(v));

    # compute the numerator and the scale -
    num = x - ρ * y;
    denom = sqrt((ν + y^2) * (1.0 - ρ^2) / (ν + 1.0));

    # return -
    return _clamp01(cdf(TDist(ν + 1.0), num / denom));
end

"""
    _bicop_hfunc(c::MyClaytonBivariateCopulaModel, u::Float64, v::Float64) -> Float64

Evaluate the h-function of a Clayton copula. The h-function is the conditional distribution function of the first asset given the second asset, the derivative of the copula with respect to `v`. We evaluate the closed form of the derivative of the Clayton copula with respect to `v`. The result is clamped away from `0` and `1`.

### Arguments
- `c::MyClaytonBivariateCopulaModel`: The Clayton copula
- `u::Float64`: The uniform number of the first asset
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The conditional probability, strictly between `0` and `1`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyClaytonBivariateCopulaModel, (θ = 2.0,));
w = JumpHMM._bicop_hfunc(c, 0.3, 0.7);
```
"""
function _bicop_hfunc(c::MyClaytonBivariateCopulaModel, u::Float64, v::Float64)::Float64

    # initialize -
    θ = c.θ;
    u, v = _clamp01(u), _clamp01(v);

    # keep the sum positive -
    S = u^(-θ) + v^(-θ) - 1.0;
    S = max(S, 1e-20);

    # return -
    return _clamp01(v^(-θ - 1.0) * S^(-1.0/θ - 1.0));
end

"""
    _bicop_hfunc(c::MyGumbelBivariateCopulaModel, u::Float64, v::Float64) -> Float64

Evaluate the h-function of a Gumbel copula. The h-function is the conditional distribution function of the first asset given the second asset, the derivative of the copula with respect to `v`. We evaluate the closed form of the derivative of the Gumbel copula with respect to `v`. The result is clamped away from `0` and `1`.

### Arguments
- `c::MyGumbelBivariateCopulaModel`: The Gumbel copula
- `u::Float64`: The uniform number of the first asset
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The conditional probability, strictly between `0` and `1`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyGumbelBivariateCopulaModel, (θ = 1.5,));
w = JumpHMM._bicop_hfunc(c, 0.3, 0.7);
```
"""
function _bicop_hfunc(c::MyGumbelBivariateCopulaModel, u::Float64, v::Float64)::Float64

    # initialize -
    θ = c.θ;
    u, v = _clamp01(u), _clamp01(v);
    lu = -log(u);
    lv = -log(v);

    # compute the copula and its derivative with respect to v -
    A = (lu^θ + lv^θ)^(1.0/θ);
    C = exp(-A);

    # return -
    return _clamp01(C * (A^(1.0 - θ)) * (lv^(θ - 1.0)) / v);
end

"""
    _bicop_hfunc(c::MyFrankBivariateCopulaModel, u::Float64, v::Float64) -> Float64

Evaluate the h-function of a Frank copula. The h-function is the conditional distribution function of the first asset given the second asset, the derivative of the copula with respect to `v`. We evaluate the closed form of the derivative of the Frank copula with respect to `v`. The result is clamped away from `0` and `1`. For `θ = 0.0`, the copula is the independence copula, and the h-function returns `u`.

### Arguments
- `c::MyFrankBivariateCopulaModel`: The Frank copula
- `u::Float64`: The uniform number of the first asset
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The conditional probability, strictly between `0` and `1`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyFrankBivariateCopulaModel, (θ = 4.0,));
w = JumpHMM._bicop_hfunc(c, 0.3, 0.7);
```
"""
function _bicop_hfunc(c::MyFrankBivariateCopulaModel, u::Float64, v::Float64)::Float64

    # initialize -
    θ = c.θ;
    u, v = _clamp01(u), _clamp01(v);

    # θ = 0 is the independence copula, where the closed form divides zero by zero. The h-function of independence is u -
    if θ == 0.0
        return u;
    end

    # the exponential terms of the closed form -
    eu = exp(-θ * u);
    ev = exp(-θ * v);
    eθ = exp(-θ);

    # compute the numerator and the denominator -
    num = ev * (eu - 1.0);
    denom = (eθ - 1.0) + (ev - 1.0) * (eu - 1.0);

    # return -
    return _clamp01(num / denom);
end

"""
    _hinv_bisect(c::AbstractBivariateCopulaModel, w::Float64, v::Float64; tol::Float64 = 1e-8, maxiter::Int64 = 100) -> Float64

Invert the h-function of the copula `c` by bisection. We search the interval from `1e-10` to `1 - 1e-10` for the number `u` with `_bicop_hfunc(c, u, v) = w`, and stop when the interval is shorter than `tol` or after `maxiter` halvings.

### Arguments
- `c::AbstractBivariateCopulaModel`: The copula
- `w::Float64`: The conditional probability to invert
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The uniform number of the first asset

### Optional Arguments
- `tol::Float64`: The length of the interval at which we stop. Default: `1e-8`
- `maxiter::Int64`: The largest number of halvings. Default: `100`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyFrankBivariateCopulaModel, (θ = 4.0,));
u = JumpHMM._hinv_bisect(c, 0.4, 0.7);
```
"""
function _hinv_bisect(c::AbstractBivariateCopulaModel, w::Float64, v::Float64; tol::Float64 = 1e-8, maxiter::Int64 = 100)::Float64

    # initialize -
    lo = 1e-10;
    hi = 1.0 - 1e-10;

    # main loop - halve the interval until it is short enough -
    for _ ∈ 1:maxiter
        mid = 0.5 * (lo + hi);
        h_mid = _bicop_hfunc(c, mid, v);
        if h_mid < w
            lo = mid;
        else
            hi = mid;
        end
        if (hi - lo) < tol
            break;
        end
    end

    # return -
    return 0.5 * (lo + hi);
end

"""
    _bicop_hinv(c::MyGaussianBivariateCopulaModel, w::Float64, v::Float64) -> Float64

Evaluate the inverse of the h-function of a Gaussian copula in closed form. The result is clamped away from `0` and `1`.

### Arguments
- `c::MyGaussianBivariateCopulaModel`: The Gaussian copula
- `w::Float64`: The conditional probability to invert
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The uniform number `u` of the first asset with `_bicop_hfunc(c, u, v) = w`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyGaussianBivariateCopulaModel, (ρ = 0.5,));
u = JumpHMM._bicop_hinv(c, 0.4, 0.7);
```
"""
function _bicop_hinv(c::MyGaussianBivariateCopulaModel, w::Float64, v::Float64)::Float64

    # initialize -
    y = quantile(Normal(), _clamp01(v));
    z = quantile(Normal(), _clamp01(w));
    ρ = c.ρ;

    # invert the closed form -
    x = z * sqrt(1.0 - ρ^2) + ρ * y;

    # return -
    return _clamp01(cdf(Normal(), x));
end

"""
    _bicop_hinv(c::MyStudentTBivariateCopulaModel, w::Float64, v::Float64) -> Float64

Evaluate the inverse of the h-function of a Student-t copula in closed form. The result is clamped away from `0` and `1`.

### Arguments
- `c::MyStudentTBivariateCopulaModel`: The Student-t copula
- `w::Float64`: The conditional probability to invert
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The uniform number `u` of the first asset with `_bicop_hfunc(c, u, v) = w`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyStudentTBivariateCopulaModel, (ρ = 0.5, ν = 5.0));
u = JumpHMM._bicop_hinv(c, 0.4, 0.7);
```
"""
function _bicop_hinv(c::MyStudentTBivariateCopulaModel, w::Float64, v::Float64)::Float64

    # initialize -
    ν = c.ν;
    ρ = c.ρ;
    y = quantile(TDist(ν), _clamp01(v));
    z = quantile(TDist(ν + 1.0), _clamp01(w));

    # invert the closed form -
    denom = sqrt((ν + y^2) * (1.0 - ρ^2) / (ν + 1.0));
    x = z * denom + ρ * y;

    # return -
    return _clamp01(cdf(TDist(ν), x));
end

"""
    _bicop_hinv(c::MyClaytonBivariateCopulaModel, w::Float64, v::Float64) -> Float64

Evaluate the inverse of the h-function of a Clayton copula. The inverse has no closed form that we trust numerically. We find the inverse by bisection with `_hinv_bisect`.

### Arguments
- `c::MyClaytonBivariateCopulaModel`: The Clayton copula
- `w::Float64`: The conditional probability to invert
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The uniform number `u` of the first asset with `_bicop_hfunc(c, u, v) = w`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyClaytonBivariateCopulaModel, (θ = 2.0,));
u = JumpHMM._bicop_hinv(c, 0.4, 0.7);
```
"""
function _bicop_hinv(c::MyClaytonBivariateCopulaModel, w::Float64, v::Float64)::Float64
    return _hinv_bisect(c, w, v);
end

"""
    _bicop_hinv(c::MyGumbelBivariateCopulaModel, w::Float64, v::Float64) -> Float64

Evaluate the inverse of the h-function of a Gumbel copula. The inverse has no closed form that we trust numerically. We find the inverse by bisection with `_hinv_bisect`.

### Arguments
- `c::MyGumbelBivariateCopulaModel`: The Gumbel copula
- `w::Float64`: The conditional probability to invert
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The uniform number `u` of the first asset with `_bicop_hfunc(c, u, v) = w`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyGumbelBivariateCopulaModel, (θ = 1.5,));
u = JumpHMM._bicop_hinv(c, 0.4, 0.7);
```
"""
function _bicop_hinv(c::MyGumbelBivariateCopulaModel, w::Float64, v::Float64)::Float64
    return _hinv_bisect(c, w, v);
end

"""
    _bicop_hinv(c::MyFrankBivariateCopulaModel, w::Float64, v::Float64) -> Float64

Evaluate the inverse of the h-function of a Frank copula. The inverse has no closed form that we trust numerically. We find the inverse by bisection with `_hinv_bisect`. For `θ = 0.0`, the copula is the independence copula, and the inverse returns `w`, clamped away from `0` and `1`.

### Arguments
- `c::MyFrankBivariateCopulaModel`: The Frank copula
- `w::Float64`: The conditional probability to invert
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The uniform number `u` of the first asset with `_bicop_hfunc(c, u, v) = w`

### Example
```julia
c = JumpHMM.build(JumpHMM.MyFrankBivariateCopulaModel, (θ = 4.0,));
u = JumpHMM._bicop_hinv(c, 0.4, 0.7);
```
"""
function _bicop_hinv(c::MyFrankBivariateCopulaModel, w::Float64, v::Float64)::Float64

    # θ = 0 is the independence copula, whose h-function is u, so the inverse is w -
    if c.θ == 0.0
        return _clamp01(w);
    end

    # return -
    return _hinv_bisect(c, w, v);
end

"""
    _bicop_logpdf(c::MyGaussianBivariateCopulaModel, u::Float64, v::Float64) -> Float64

Evaluate the log of the density of a Gaussian copula at the point `(u, v)`. We use the log-density to compute the AIC of the copula.

### Arguments
- `c::MyGaussianBivariateCopulaModel`: The Gaussian copula
- `u::Float64`: The uniform number of the first asset
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The log-density

### Example
```julia
c = JumpHMM.build(JumpHMM.MyGaussianBivariateCopulaModel, (ρ = 0.5,));
log_density = JumpHMM._bicop_logpdf(c, 0.3, 0.7);
```
"""
function _bicop_logpdf(c::MyGaussianBivariateCopulaModel, u::Float64, v::Float64)::Float64

    # initialize -
    x = quantile(Normal(), _clamp01(u));
    y = quantile(Normal(), _clamp01(v));
    ρ = c.ρ;
    r2 = ρ^2;

    # return -
    return -0.5 * log(1.0 - r2) - (r2 * (x^2 + y^2) - 2.0 * ρ * x * y) / (2.0 * (1.0 - r2));
end

"""
    _bicop_logpdf(c::MyStudentTBivariateCopulaModel, u::Float64, v::Float64) -> Float64

Evaluate the log of the density of a Student-t copula at the point `(u, v)`. We use the log-density to compute the AIC of the copula.

### Arguments
- `c::MyStudentTBivariateCopulaModel`: The Student-t copula
- `u::Float64`: The uniform number of the first asset
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The log-density

### Example
```julia
c = JumpHMM.build(JumpHMM.MyStudentTBivariateCopulaModel, (ρ = 0.5, ν = 5.0));
log_density = JumpHMM._bicop_logpdf(c, 0.3, 0.7);
```
"""
function _bicop_logpdf(c::MyStudentTBivariateCopulaModel, u::Float64, v::Float64)::Float64

    # initialize -
    ν = c.ν;
    ρ = c.ρ;
    x = quantile(TDist(ν), _clamp01(u));
    y = quantile(TDist(ν), _clamp01(v));
    r2 = ρ^2;

    # bivariate Student-t copula density -
    ll = loggamma((ν + 2.0) / 2.0) + loggamma(ν / 2.0) -
         2.0 * loggamma((ν + 1.0) / 2.0) -
         0.5 * log(1.0 - r2) -
         ((ν + 2.0) / 2.0) * log(1.0 + (x^2 + y^2 - 2.0 * ρ * x * y) / (ν * (1.0 - r2))) +
         ((ν + 1.0) / 2.0) * (log(1.0 + x^2 / ν) + log(1.0 + y^2 / ν));

    # return -
    return ll;
end

"""
    _bicop_logpdf(c::MyClaytonBivariateCopulaModel, u::Float64, v::Float64) -> Float64

Evaluate the log of the density of a Clayton copula at the point `(u, v)`. We use the log-density to compute the AIC of the copula.

### Arguments
- `c::MyClaytonBivariateCopulaModel`: The Clayton copula
- `u::Float64`: The uniform number of the first asset
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The log-density

### Example
```julia
c = JumpHMM.build(JumpHMM.MyClaytonBivariateCopulaModel, (θ = 2.0,));
log_density = JumpHMM._bicop_logpdf(c, 0.3, 0.7);
```
"""
function _bicop_logpdf(c::MyClaytonBivariateCopulaModel, u::Float64, v::Float64)::Float64

    # initialize -
    θ = c.θ;
    u, v = _clamp01(u), _clamp01(v);

    # return -
    return log(1.0 + θ) + (-θ - 1.0) * (log(u) + log(v)) +
           (-2.0 - 1.0/θ) * log(u^(-θ) + v^(-θ) - 1.0);
end

"""
    _bicop_logpdf(c::MyGumbelBivariateCopulaModel, u::Float64, v::Float64) -> Float64

Evaluate the log of the density of a Gumbel copula at the point `(u, v)`. We use the log-density to compute the AIC of the copula.

### Arguments
- `c::MyGumbelBivariateCopulaModel`: The Gumbel copula
- `u::Float64`: The uniform number of the first asset
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The log-density

### Example
```julia
c = JumpHMM.build(JumpHMM.MyGumbelBivariateCopulaModel, (θ = 1.5,));
log_density = JumpHMM._bicop_logpdf(c, 0.3, 0.7);
```
"""
function _bicop_logpdf(c::MyGumbelBivariateCopulaModel, u::Float64, v::Float64)::Float64

    # initialize -
    θ = c.θ;
    u, v = _clamp01(u), _clamp01(v);
    lu = -log(u);
    lv = -log(v);
    A = (lu^θ + lv^θ)^(1.0/θ);

    # Gumbel density, derived from the distribution function, in five terms -
    C = exp(-A);
    t1 = log(C);
    t2 = (θ - 1.0) * (log(lu) + log(lv));
    t3 = -log(u) - log(v);
    t4 = (1.0/θ - 2.0) * log(lu^θ + lv^θ);
    t5 = log(A + θ - 1.0);

    # return -
    return t1 + t2 + t3 + t4 + t5;
end

"""
    _bicop_logpdf(c::MyFrankBivariateCopulaModel, u::Float64, v::Float64) -> Float64

Evaluate the log of the density of a Frank copula at the point `(u, v)`. We use the log-density to compute the AIC of the copula. We return `-Inf` when the density is too small to evaluate. For `θ = 0.0`, the copula is the independence copula, whose density is `1`, and we return `0.0`.

### Arguments
- `c::MyFrankBivariateCopulaModel`: The Frank copula
- `u::Float64`: The uniform number of the first asset
- `v::Float64`: The uniform number of the second asset

### Returns
- `Float64`: The log-density

### Example
```julia
c = JumpHMM.build(JumpHMM.MyFrankBivariateCopulaModel, (θ = 4.0,));
log_density = JumpHMM._bicop_logpdf(c, 0.3, 0.7);
```
"""
function _bicop_logpdf(c::MyFrankBivariateCopulaModel, u::Float64, v::Float64)::Float64

    # initialize -
    θ = c.θ;
    u, v = _clamp01(u), _clamp01(v);

    # θ = 0 is the independence copula, whose density is 1 everywhere -
    if θ == 0.0
        return 0.0;
    end

    # the numerator and the denominator of the density -
    et = exp(-θ);
    eu = exp(-θ * u);
    ev = exp(-θ * v);
    num = θ * (1.0 - et) * exp(-θ * (u + v));
    denom = ((1.0 - et) - (1.0 - eu) * (1.0 - ev))^2;

    # the density is not usable when the denominator is this small -
    if denom < 1e-30
        return -Inf;
    end

    # return -
    return log(abs(num)) - log(denom);
end

"""
    _bicop_fit(model::Type{MyGaussianBivariateCopulaModel}, u::Array{Float64,1}, v::Array{Float64,1}) -> MyGaussianBivariateCopulaModel

Fit a Gaussian copula to the uniform numbers `u` and `v`. We convert Kendall's tau to the correlation `ρ = sin(π τ / 2)`, and clamp `ρ` to the interval from `-0.999` to `0.999`.

### Arguments
- `model::Type{MyGaussianBivariateCopulaModel}`: The type to fit
- `u::Array{Float64,1}`: The uniform numbers of the first asset
- `v::Array{Float64,1}`: The uniform numbers of the second asset

### Returns
- `MyGaussianBivariateCopulaModel`: The fitted Gaussian copula

### Example
```julia
U = JumpHMM._uniform_ranks(GA[:, 2:3]);
c = JumpHMM._bicop_fit(JumpHMM.MyGaussianBivariateCopulaModel, U[:, 1], U[:, 2]);
```
"""
function _bicop_fit(model::Type{MyGaussianBivariateCopulaModel}, u::Array{Float64,1}, v::Array{Float64,1})::MyGaussianBivariateCopulaModel

    # estimate the correlation from Kendall's tau -
    τ = corkendall(u, v);
    ρ = sin(π / 2.0 * τ);
    ρ = clamp(ρ, -0.999, 0.999);

    # return -
    return build(MyGaussianBivariateCopulaModel, (ρ = ρ,));
end

"""
    _bicop_fit(model::Type{MyStudentTBivariateCopulaModel}, u::Array{Float64,1}, v::Array{Float64,1}) -> MyStudentTBivariateCopulaModel

Fit a Student-t copula to the uniform numbers `u` and `v`. We estimate `ρ` from Kendall's tau as for the Gaussian copula. Then we pick `ν` from `2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30` as the value with the largest log-likelihood.

### Arguments
- `model::Type{MyStudentTBivariateCopulaModel}`: The type to fit
- `u::Array{Float64,1}`: The uniform numbers of the first asset
- `v::Array{Float64,1}`: The uniform numbers of the second asset

### Returns
- `MyStudentTBivariateCopulaModel`: The fitted Student-t copula

### Example
```julia
U = JumpHMM._uniform_ranks(GA[:, 2:3]);
c = JumpHMM._bicop_fit(JumpHMM.MyStudentTBivariateCopulaModel, U[:, 1], U[:, 2]);
```
"""
function _bicop_fit(model::Type{MyStudentTBivariateCopulaModel}, u::Array{Float64,1}, v::Array{Float64,1})::MyStudentTBivariateCopulaModel

    # estimate the correlation from Kendall's tau -
    τ = corkendall(u, v);
    ρ = sin(π / 2.0 * τ);
    ρ = clamp(ρ, -0.999, 0.999);

    # pick the degrees of freedom with the largest log-likelihood -
    best_ν = 5.0;
    best_ll = -Inf;
    for ν_candidate ∈ [2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0]
        c = build(MyStudentTBivariateCopulaModel, (ρ = ρ, ν = ν_candidate));
        ll = sum(_bicop_logpdf(c, u[i], v[i]) for i ∈ eachindex(u));
        if isfinite(ll) == true
            if ll > best_ll
                best_ll = ll;
                best_ν = ν_candidate;
            end
        end
    end

    # return -
    return build(MyStudentTBivariateCopulaModel, (ρ = ρ, ν = best_ν));
end

"""
    _bicop_fit(model::Type{MyClaytonBivariateCopulaModel}, u::Array{Float64,1}, v::Array{Float64,1}) -> MyClaytonBivariateCopulaModel

Fit a Clayton copula to the uniform numbers `u` and `v`. We set `θ = 2τ / (1 - τ)` from Kendall's tau, with a floor of `0.01`. If tau is at most `0.01`, we return `θ = 0.02`, which is close to independence.

### Arguments
- `model::Type{MyClaytonBivariateCopulaModel}`: The type to fit
- `u::Array{Float64,1}`: The uniform numbers of the first asset
- `v::Array{Float64,1}`: The uniform numbers of the second asset

### Returns
- `MyClaytonBivariateCopulaModel`: The fitted Clayton copula

### Example
```julia
U = JumpHMM._uniform_ranks(GA[:, 2:3]);
c = JumpHMM._bicop_fit(JumpHMM.MyClaytonBivariateCopulaModel, U[:, 1], U[:, 2]);
```
"""
function _bicop_fit(model::Type{MyClaytonBivariateCopulaModel}, u::Array{Float64,1}, v::Array{Float64,1})::MyClaytonBivariateCopulaModel

    # near independence, return a small parameter -
    τ = corkendall(u, v);
    if τ ≤ 0.01
        return build(MyClaytonBivariateCopulaModel, (θ = 0.02,));
    end

    # estimate the parameter from Kendall's tau -
    θ = 2.0 * τ / (1.0 - τ);
    θ = max(θ, 0.01);

    # return -
    return build(MyClaytonBivariateCopulaModel, (θ = θ,));
end

"""
    _bicop_fit(model::Type{MyGumbelBivariateCopulaModel}, u::Array{Float64,1}, v::Array{Float64,1}) -> MyGumbelBivariateCopulaModel

Fit a Gumbel copula to the uniform numbers `u` and `v`. We set `θ = 1 / (1 - τ)` from Kendall's tau, with a floor of `1`. If tau is at most `0.01`, we return `θ = 1`, which is independence.

### Arguments
- `model::Type{MyGumbelBivariateCopulaModel}`: The type to fit
- `u::Array{Float64,1}`: The uniform numbers of the first asset
- `v::Array{Float64,1}`: The uniform numbers of the second asset

### Returns
- `MyGumbelBivariateCopulaModel`: The fitted Gumbel copula

### Example
```julia
U = JumpHMM._uniform_ranks(GA[:, 2:3]);
c = JumpHMM._bicop_fit(JumpHMM.MyGumbelBivariateCopulaModel, U[:, 1], U[:, 2]);
```
"""
function _bicop_fit(model::Type{MyGumbelBivariateCopulaModel}, u::Array{Float64,1}, v::Array{Float64,1})::MyGumbelBivariateCopulaModel

    # at independence, return the smallest parameter -
    τ = corkendall(u, v);
    if τ ≤ 0.01
        return build(MyGumbelBivariateCopulaModel, (θ = 1.0,));
    end

    # estimate the parameter from Kendall's tau -
    θ = 1.0 / (1.0 - τ);
    θ = max(θ, 1.0);

    # return -
    return build(MyGumbelBivariateCopulaModel, (θ = θ,));
end

"""
    _frank_tau(θ::Float64) -> Float64

Compute Kendall's tau of a Frank copula with parameter `θ`, `τ = 1 - 4/θ (1 - D₁(θ))`, where `D₁` is the first Debye function. We integrate the Debye function with the midpoint rule on `100` points. We return `0` when `θ` is smaller than `0.01` in absolute value.

### Arguments
- `θ::Float64`: The parameter of the Frank copula

### Returns
- `Float64`: Kendall's tau

### Example
```julia
τ = JumpHMM._frank_tau(4.0);
```
"""
function _frank_tau(θ::Float64)::Float64

    # a parameter this small means independence -
    if abs(θ) < 0.01
        return 0.0;
    end

    # integrate the Debye function with the midpoint rule -
    number_of_points = 100;
    quad_step = abs(θ) / number_of_points;
    D1 = 0.0;
    for k ∈ 1:number_of_points
        t = (k - 0.5) * quad_step;
        D1 += t / (exp(t) - 1.0) * quad_step;
    end
    D1 /= abs(θ);

    # return -
    return sign(θ) * (1.0 - 4.0 / abs(θ) * (1.0 - D1));
end

"""
    _bicop_fit(model::Type{MyFrankBivariateCopulaModel}, u::Array{Float64,1}, v::Array{Float64,1}) -> MyFrankBivariateCopulaModel

Fit a Frank copula to the uniform numbers `u` and `v`. We find `θ` by bisection on `_frank_tau` so that the copula has the Kendall's tau of the data. If tau is smaller than `0.01` in absolute value, we return `θ = ±0.01` with the sign of tau.

### Arguments
- `model::Type{MyFrankBivariateCopulaModel}`: The type to fit
- `u::Array{Float64,1}`: The uniform numbers of the first asset
- `v::Array{Float64,1}`: The uniform numbers of the second asset

### Returns
- `MyFrankBivariateCopulaModel`: The fitted Frank copula

### Example
```julia
U = JumpHMM._uniform_ranks(GA[:, 2:3]);
c = JumpHMM._bicop_fit(JumpHMM.MyFrankBivariateCopulaModel, U[:, 1], U[:, 2]);
```
"""
function _bicop_fit(model::Type{MyFrankBivariateCopulaModel}, u::Array{Float64,1}, v::Array{Float64,1})::MyFrankBivariateCopulaModel

    # near independence, return a small parameter with the sign of tau -
    τ = corkendall(u, v);
    if abs(τ) < 0.01
        return build(MyFrankBivariateCopulaModel, (θ = 0.01 * sign(τ + 0.001),));
    end

    # the search interval has the sign of tau -
    lo = -50.0;
    hi = -0.01;
    if τ > 0
        lo = 0.01;
        hi = 50.0;
    end

    # main loop - halve the interval until it is short enough -
    for _ ∈ 1:100
        mid = 0.5 * (lo + hi);
        τ_mid = _frank_tau(mid);
        if τ_mid < τ
            lo = mid;
        else
            hi = mid;
        end
        if abs(hi - lo) < 0.01
            break;
        end
    end

    # return -
    return build(MyFrankBivariateCopulaModel, (θ = 0.5 * (lo + hi),));
end

"""
    _select_bicop(u::Array{Float64,1}, v::Array{Float64,1}) -> AbstractBivariateCopulaModel

Choose the bivariate copula for one edge. We fit each of the five families to `u` and `v`, skip a family whose log-density is not finite at some point, and keep the family with the smallest AIC. The Student-t copula has two parameters, and every other family has one.

### Arguments
- `u::Array{Float64,1}`: The uniform numbers of the first asset
- `v::Array{Float64,1}`: The uniform numbers of the second asset

### Returns
- `AbstractBivariateCopulaModel`: The fitted copula with the smallest AIC

### Example
```julia
U = JumpHMM._uniform_ranks(GA[:, 2:3]);
c = JumpHMM._select_bicop(U[:, 1], U[:, 2]);
```
"""
function _select_bicop(u::Array{Float64,1}, v::Array{Float64,1})::AbstractBivariateCopulaModel

    # initialize -
    best_aic = Inf;
    best_cop = build(MyGaussianBivariateCopulaModel, (ρ = 0.0,));

    # main loop - fit each family, and keep the one with the smallest AIC -
    for family ∈ _BICOP_FAMILIES
        cop = _bicop_fit(family, u, v);

        # count the parameters -
        number_of_parameters = 1;
        if cop isa MyStudentTBivariateCopulaModel
            number_of_parameters = 2;
        end

        # add up the log-density, and stop if it is not finite -
        ll = 0.0;
        valid = true;
        for i ∈ eachindex(u)
            lp = _bicop_logpdf(cop, u[i], v[i]);
            if isfinite(lp) == false
                valid = false;
                break;
            end
            ll += lp;
        end

        # keep the copula if it has the smallest AIC so far -
        if valid == true
            aic = 2.0 * number_of_parameters - 2.0 * ll;
            if aic < best_aic
                best_aic = aic;
                best_cop = cop;
            end
        end
    end

    # return -
    return best_cop;
end

"""
    sample_dependence(model::MyVineCopulaModel, number_of_steps::Int64) -> Array{Float64,2}

Draw dependent uniform numbers from a C-vine copula. For each step we draw one independent uniform number per asset, and push them through the inverse h-functions of the edges, from the last tree back to the first. Then we put the columns back in the order of the assets. The draws come from the global random number generator. To repeat a sample, call `Random.seed!` first.

### Arguments
- `model::MyVineCopulaModel`: The vine copula
- `number_of_steps::Int64`: The number of rows to draw

### Returns
- `Array{Float64,2}`: The uniform numbers, with `number_of_steps` rows (days) and one column per asset. Every value is strictly between `0` and `1`

### Example
```julia
U = JumpHMM.sample_dependence(vine, 250);
```
"""
function sample_dependence(model::MyVineCopulaModel, number_of_steps::Int64)::Array{Float64,2}

    # initialize -
    d = model.number_of_assets;
    order = model.order;
    edges = model.edges;
    U = Matrix{Float64}(undef, number_of_steps, d);

    # main loop - one row per step -
    for s ∈ 1:number_of_steps

        # draw one independent uniform number per asset -
        w = rand(d);
        v = zeros(d);
        v[1] = w[1];

        # push each draw through the inverse h-functions, from the last tree back to the first -
        for k ∈ 2:d
            q = w[k];
            for tree ∈ (k-1):-1:1
                edge_index = k - tree;
                if edge_index > length(edges[tree])
                    continue;
                end
                cop = edges[tree][edge_index].copula;
                q = _bicop_hinv(cop, q, v[tree]);
            end
            v[k] = q;
        end

        # put the columns back in the order of the assets -
        for k ∈ 1:d
            U[s, order[k]] = v[k];
        end
    end

    # return -
    return U;
end
