# ============================================================================
# Vine Copula Implementation (C-vine with 5 bivariate families)
# ============================================================================

const _BICOP_FAMILIES = [GaussianBiCopula, StudentTBiCopula, ClaytonBiCopula,
                          GumbelBiCopula, FrankBiCopula]

# clamp to avoid log(0), quantile(0), etc.
_clamp01(x) = clamp(x, 1e-10, 1.0 - 1e-10)

# ============================================================================
# h-functions: h(u|v) = ∂C(u,v)/∂v  (conditional CDF)
# ============================================================================

function _bicop_hfunc(c::GaussianBiCopula, u::Float64, v::Float64)
    x = quantile(Normal(), _clamp01(u))
    y = quantile(Normal(), _clamp01(v))
    ρ = c.ρ
    return _clamp01(cdf(Normal(), (x - ρ * y) / sqrt(1.0 - ρ^2)))
end

function _bicop_hfunc(c::StudentTBiCopula, u::Float64, v::Float64)
    ν = c.ν
    ρ = c.ρ
    x = quantile(TDist(ν), _clamp01(u))
    y = quantile(TDist(ν), _clamp01(v))
    num = x - ρ * y
    denom = sqrt((ν + y^2) * (1.0 - ρ^2) / (ν + 1.0))
    return _clamp01(cdf(TDist(ν + 1.0), num / denom))
end

function _bicop_hfunc(c::ClaytonBiCopula, u::Float64, v::Float64)
    θ = c.θ
    u, v = _clamp01(u), _clamp01(v)
    S = u^(-θ) + v^(-θ) - 1.0
    S = max(S, 1e-20)  # guard against negative values
    return _clamp01(v^(-θ - 1.0) * S^(-1.0/θ - 1.0))
end

function _bicop_hfunc(c::GumbelBiCopula, u::Float64, v::Float64)
    θ = c.θ
    u, v = _clamp01(u), _clamp01(v)
    lu = -log(u)
    lv = -log(v)
    A = (lu^θ + lv^θ)^(1.0/θ)
    C = exp(-A)
    # ∂C/∂v = C * A^(1-θ) * (lv)^(θ-1) / v
    return _clamp01(C * (A^(1.0 - θ)) * (lv^(θ - 1.0)) / v)
end

function _bicop_hfunc(c::FrankBiCopula, u::Float64, v::Float64)
    θ = c.θ
    u, v = _clamp01(u), _clamp01(v)
    eu = exp(-θ * u)
    ev = exp(-θ * v)
    eθ = exp(-θ)
    num = ev * (eu - 1.0)
    denom = (eθ - 1.0) + (ev - 1.0) * (eu - 1.0)
    return _clamp01(num / denom)
end

# ============================================================================
# Inverse h-functions: h⁻¹(w|v) such that h(h⁻¹(w|v)|v) = w
# ============================================================================

function _bicop_hinv(c::GaussianBiCopula, w::Float64, v::Float64)
    y = quantile(Normal(), _clamp01(v))
    z = quantile(Normal(), _clamp01(w))
    ρ = c.ρ
    x = z * sqrt(1.0 - ρ^2) + ρ * y
    return _clamp01(cdf(Normal(), x))
end

function _bicop_hinv(c::StudentTBiCopula, w::Float64, v::Float64)
    ν = c.ν
    ρ = c.ρ
    y = quantile(TDist(ν), _clamp01(v))
    z = quantile(TDist(ν + 1.0), _clamp01(w))
    denom = sqrt((ν + y^2) * (1.0 - ρ^2) / (ν + 1.0))
    x = z * denom + ρ * y
    return _clamp01(cdf(TDist(ν), x))
end

function _bicop_hinv(c::ClaytonBiCopula, w::Float64, v::Float64)
    # numerical inversion via bisection
    return _hinv_bisect(c, w, v)
end

function _bicop_hinv(c::GumbelBiCopula, w::Float64, v::Float64)
    return _hinv_bisect(c, w, v)
end

function _bicop_hinv(c::FrankBiCopula, w::Float64, v::Float64)
    # use bisection — analytical formula is numerically fragile
    return _hinv_bisect(c, w, v)
end

function _hinv_bisect(c::AbstractBivariateCopula, w::Float64, v::Float64;
                       tol::Float64=1e-8, maxiter::Int=100)
    lo, hi = 1e-10, 1.0 - 1e-10
    for _ in 1:maxiter
        mid = 0.5 * (lo + hi)
        h_mid = _bicop_hfunc(c, mid, v)
        if h_mid < w
            lo = mid
        else
            hi = mid
        end
        (hi - lo) < tol && break
    end
    return 0.5 * (lo + hi)
end

# ============================================================================
# Log-PDF for AIC computation
# ============================================================================

function _bicop_logpdf(c::GaussianBiCopula, u::Float64, v::Float64)
    x = quantile(Normal(), _clamp01(u))
    y = quantile(Normal(), _clamp01(v))
    ρ = c.ρ
    r2 = ρ^2
    return -0.5 * log(1.0 - r2) - (r2 * (x^2 + y^2) - 2.0 * ρ * x * y) / (2.0 * (1.0 - r2))
end

function _bicop_logpdf(c::StudentTBiCopula, u::Float64, v::Float64)
    ν = c.ν
    ρ = c.ρ
    x = quantile(TDist(ν), _clamp01(u))
    y = quantile(TDist(ν), _clamp01(v))
    r2 = ρ^2
    # bivariate t copula density
    ll = loggamma((ν + 2.0) / 2.0) + loggamma(ν / 2.0) -
         2.0 * loggamma((ν + 1.0) / 2.0) -
         0.5 * log(1.0 - r2) -
         ((ν + 2.0) / 2.0) * log(1.0 + (x^2 + y^2 - 2.0 * ρ * x * y) / (ν * (1.0 - r2))) +
         ((ν + 1.0) / 2.0) * (log(1.0 + x^2 / ν) + log(1.0 + y^2 / ν))
    return ll
end

function _bicop_logpdf(c::ClaytonBiCopula, u::Float64, v::Float64)
    θ = c.θ
    u, v = _clamp01(u), _clamp01(v)
    return log(1.0 + θ) + (-θ - 1.0) * (log(u) + log(v)) +
           (-2.0 - 1.0/θ) * log(u^(-θ) + v^(-θ) - 1.0)
end

function _bicop_logpdf(c::GumbelBiCopula, u::Float64, v::Float64)
    θ = c.θ
    u, v = _clamp01(u), _clamp01(v)
    lu = -log(u)
    lv = -log(v)
    A = (lu^θ + lv^θ)^(1.0/θ)
    # Gumbel PDF (derived from CDF)
    C = exp(-A)
    t1 = log(C) # = -A
    t2 = (θ - 1.0) * (log(lu) + log(lv))
    t3 = -log(u) - log(v)
    t4 = (1.0/θ - 2.0) * log(lu^θ + lv^θ)
    t5 = log(A + θ - 1.0)
    return t1 + t2 + t3 + t4 + t5
end

function _bicop_logpdf(c::FrankBiCopula, u::Float64, v::Float64)
    θ = c.θ
    u, v = _clamp01(u), _clamp01(v)
    et = exp(-θ)
    eu = exp(-θ * u)
    ev = exp(-θ * v)
    num = θ * (1.0 - et) * exp(-θ * (u + v))
    denom = ((1.0 - et) - (1.0 - eu) * (1.0 - ev))^2
    if denom < 1e-30
        return -Inf
    end
    return log(abs(num)) - log(denom)
end

# ============================================================================
# Fitting bivariate copulas from (u, v) uniform data
# ============================================================================

function _bicop_fit(::Type{GaussianBiCopula}, u::Vector{Float64}, v::Vector{Float64})
    τ = corkendall(u, v)
    ρ = sin(π / 2.0 * τ)
    ρ = clamp(ρ, -0.999, 0.999)
    return GaussianBiCopula(ρ)
end

function _bicop_fit(::Type{StudentTBiCopula}, u::Vector{Float64}, v::Vector{Float64})
    τ = corkendall(u, v)
    ρ = sin(π / 2.0 * τ)
    ρ = clamp(ρ, -0.999, 0.999)
    # profile MLE for ν
    best_ν = 5.0
    best_ll = -Inf
    for ν_cand in [2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0]
        c = StudentTBiCopula(ρ, ν_cand)
        ll = sum(_bicop_logpdf(c, u[i], v[i]) for i in eachindex(u))
        if isfinite(ll) && ll > best_ll
            best_ll = ll
            best_ν = ν_cand
        end
    end
    return StudentTBiCopula(ρ, best_ν)
end

function _bicop_fit(::Type{ClaytonBiCopula}, u::Vector{Float64}, v::Vector{Float64})
    τ = corkendall(u, v)
    if τ ≤ 0.01
        return ClaytonBiCopula(0.02)  # near-independence
    end
    θ = 2.0 * τ / (1.0 - τ)
    θ = max(θ, 0.01)
    return ClaytonBiCopula(θ)
end

function _bicop_fit(::Type{GumbelBiCopula}, u::Vector{Float64}, v::Vector{Float64})
    τ = corkendall(u, v)
    if τ ≤ 0.01
        return GumbelBiCopula(1.0)  # independence
    end
    θ = 1.0 / (1.0 - τ)
    θ = max(θ, 1.0)
    return GumbelBiCopula(θ)
end

function _bicop_fit(::Type{FrankBiCopula}, u::Vector{Float64}, v::Vector{Float64})
    τ = corkendall(u, v)
    # numerical inversion: τ = 1 - 4/θ * (1 - D₁(θ))
    # where D₁(θ) = (1/θ) ∫₀ᶿ t/(eᵗ-1) dt (first Debye function)
    # use bisection
    if abs(τ) < 0.01
        return FrankBiCopula(0.01 * sign(τ + 0.001))
    end

    function _frank_tau(θ)
        if abs(θ) < 0.01
            return 0.0
        end
        # numerical Debye function via quadrature
        n_quad = 100
        dt = abs(θ) / n_quad
        D1 = 0.0
        for k in 1:n_quad
            t = (k - 0.5) * dt
            D1 += t / (exp(t) - 1.0) * dt
        end
        D1 /= abs(θ)
        return sign(θ) * (1.0 - 4.0 / abs(θ) * (1.0 - D1))
    end

    # bisection to find θ given τ
    if τ > 0
        lo, hi = 0.01, 50.0
    else
        lo, hi = -50.0, -0.01
    end
    for _ in 1:100
        mid = 0.5 * (lo + hi)
        τ_mid = _frank_tau(mid)
        if τ_mid < τ
            lo = mid
        else
            hi = mid
        end
        abs(hi - lo) < 0.01 && break
    end
    return FrankBiCopula(0.5 * (lo + hi))
end

# ============================================================================
# AIC-based family selection
# ============================================================================

function _select_bicop(u::Vector{Float64}, v::Vector{Float64})
    best_aic = Inf
    best_cop = GaussianBiCopula(0.0)

    for Family in _BICOP_FAMILIES
        cop = _bicop_fit(Family, u, v)
        n_params = (cop isa StudentTBiCopula) ? 2 : 1
        ll = 0.0
        valid = true
        for i in eachindex(u)
            lp = _bicop_logpdf(cop, u[i], v[i])
            if !isfinite(lp)
                valid = false
                break
            end
            ll += lp
        end
        if !valid
            continue
        end
        aic = 2.0 * n_params - 2.0 * ll
        if aic < best_aic
            best_aic = aic
            best_cop = cop
        end
    end

    return best_cop
end

# ============================================================================
# C-Vine fitting
# ============================================================================

"""
    fit(::Type{VineCopula}, returns) → VineCopula

Fit a C-vine copula from an (n_obs × n_assets) matrix of excess growth rates.
Automatically selects bivariate copula families per edge via AIC.
Variable ordering: most connected variable (by sum of |Kendall's tau|) is
placed at the center of tree 1.
"""
function fit(::Type{VineCopula}, returns::AbstractMatrix{Float64})
    U = _pit(returns)
    n, d = size(U)

    # determine variable order by sum of absolute Kendall's tau
    tau_sums = zeros(d)
    for i in 1:d
        for j in (i+1):d
            τ = abs(corkendall(U[:, i], U[:, j]))
            tau_sums[i] += τ
            tau_sums[j] += τ
        end
    end
    order = sortperm(tau_sums, rev=true)

    # reorder columns
    V = U[:, order]

    # build vine tree by tree
    # pseudo-observations at each level
    # v_data[tree][edge_idx] stores the pseudo-obs for that edge
    edges = Vector{Vector{VineEdge}}()

    # current pseudo-observations: initially the raw uniforms
    # For C-vine, tree k has center = order[k]
    # We need h-function outputs for conditioning
    # Store: hfunc_cache[k, j] = h(V_j | V_{order[k]}; conditioned on order[1:k-1])
    # represented as vectors of pseudo-observations

    # working data: pseudo[j] = current pseudo-obs for variable j (1-indexed in reordered space)
    pseudo = [V[:, j] for j in 1:d]

    for k in 1:(d-1)
        tree_edges = Vector{VineEdge}()
        new_pseudo = Vector{Vector{Float64}}(undef, d)

        center = k  # in reordered space, variable k is the center of tree k
        center_data = pseudo[center]

        for j in (k+1):d
            paired_data = pseudo[j]

            # fit bivariate copula
            cop = _select_bicop(center_data, paired_data)

            # conditioned set (in original variable indices)
            cond_set = order[1:(k-1)]

            edge = VineEdge(order[center], order[j], cond_set, cop)
            push!(tree_edges, edge)

            # compute pseudo-observations for next tree level
            # h(V_j | V_center)
            h_vals = [_bicop_hfunc(cop, paired_data[i], center_data[i]) for i in 1:n]
            new_pseudo[j] = h_vals
        end

        # update pseudo-observations for next level
        # center variable's pseudo-obs for next level
        if k < d - 1
            # h(V_center | V_{first paired}) — use the first edge
            cop_first = tree_edges[1].copula
            first_j = k + 1
            h_center = [_bicop_hfunc(cop_first, center_data[i], pseudo[first_j][i])
                        for i in 1:n]
            new_pseudo[center] = h_center
        end

        for j in (k+1):d
            pseudo[j] = new_pseudo[j]
        end

        push!(edges, tree_edges)
    end

    return VineCopula(d, order, edges)
end

# ============================================================================
# C-Vine sampling
# ============================================================================

"""
    sample_dependence(vc::VineCopula, n) → Matrix{Float64}

Draw n samples from the C-vine copula. Returns (n × d) uniform matrix.
"""
function sample_dependence(vc::VineCopula, n::Int)
    d = vc.d
    order = vc.order
    edges = vc.edges

    U = Matrix{Float64}(undef, n, d)

    for s in 1:n
        w = rand(d)  # d independent Uniform(0,1)
        v = zeros(d)  # working array in vine-order space

        v[1] = w[1]

        for k in 2:d
            # transform w[k] through inverse h-functions
            # walking back through trees (k-1), (k-2), ..., 1
            q = w[k]
            for tree in (k-1):-1:1
                # find the edge at tree level `tree` that involves variable k
                edge_idx = k - tree  # in C-vine, edge index within the tree
                if edge_idx > length(edges[tree])
                    continue
                end
                edge = edges[tree][edge_idx]
                cop = edge.copula
                q = _bicop_hinv(cop, q, v[tree])
            end
            v[k] = q
        end

        # map back from vine order to original variable order
        for k in 1:d
            U[s, order[k]] = v[k]
        end
    end

    return U
end
