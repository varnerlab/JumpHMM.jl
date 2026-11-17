# --- Abstract types ---------------------------------------------------------------------------------------------------- #

"""
    abstract type AbstractMarkovModel

Supertype for the hidden Markov models in `JumpHMM`.
"""
abstract type AbstractMarkovModel end

"""
    abstract type AbstractDependenceModel

Supertype for the models that describe how the growth rates of several assets depend on each other.
"""
abstract type AbstractDependenceModel end

"""
    abstract type AbstractBivariateCopulaModel

Supertype for the two-variable copula families used as the building blocks of a vine copula.
"""
abstract type AbstractBivariateCopulaModel end

"""
    abstract type AbstractReportModel

Supertype for the models that hold the results of a validation run.
"""
abstract type AbstractReportModel end

"""
    abstract type AbstractVineEdgeModel

Supertype for the edges of a vine copula. Each edge links two assets through a bivariate copula.
"""
abstract type AbstractVineEdgeModel end

"""
    abstract type AbstractPortfolioModel

Supertype for the models of several assets that pair a hidden Markov model for each asset with a dependence model.
"""
abstract type AbstractPortfolioModel end

# --- Jump hidden Markov model ------------------------------------------------------------------------------------------ #

"""
    mutable struct MyJumpHiddenMarkovModel <: AbstractMarkovModel

The `MyJumpHiddenMarkovModel` mutable struct holds a hidden Markov model of the growth rates of one asset.
The states partition the growth rates, each state emits growth rates from a Student-t distribution,
and a jump episode can move the asset into the tails.

### Required fields
- `number_of_states::Int64`: The number of states of the model
- `laplace_μ::Float64`: The location of the Laplace distribution that sets the state boundaries
- `laplace_b::Float64`: The scale of the Laplace distribution that sets the state boundaries
- `boundaries::Array{Float64,1}`: The `number_of_states + 1` edges of the states. The first edge is `-Inf` and the last edge is `Inf`
- `transition::Array{Float64,2}`: The transition matrix. Row `i` holds the probabilities of moving from state `i` to each state
- `stationary::Array{Float64,1}`: The stationary distribution of the transition matrix
- `emission_μ::Array{Float64,1}`: The location of the Student-t emission distribution of each state
- `emission_σ::Array{Float64,1}`: The scale of the Student-t emission distribution of each state
- `emission_n::Array{Int64,1}`: The number of observations used to estimate the emission of each state
- `emission_fallback::Array{Bool,1}`: A flag that is `true` if a state uses the standard deviation of all growth rates, because the state had fewer than `min_obs` growth rates or a standard deviation below `1e-12`
- `ν::Float64`: The degrees of freedom of the Student-t emission distributions
- `ϵ::Float64`: The probability that a jump episode starts on a given day. A value of `0.0` means the model has no jumps
- `λ::Float64`: The mean of the Poisson episode length, in trading days
- `p_neg::Float64`: The probability that a day of a jump episode is in the negative tail
- `N_tail::Int64`: The number of states in each tail that a jump episode can reach
- `Δt::Float64`: The time step of the growth rates, in years
"""
mutable struct MyJumpHiddenMarkovModel <: AbstractMarkovModel

    # data -
    number_of_states::Int64
    laplace_μ::Float64
    laplace_b::Float64
    boundaries::Array{Float64,1}
    transition::Array{Float64,2}
    stationary::Array{Float64,1}
    emission_μ::Array{Float64,1}
    emission_σ::Array{Float64,1}
    emission_n::Array{Int64,1}
    emission_fallback::Array{Bool,1}
    ν::Float64
    ϵ::Float64
    λ::Float64
    p_neg::Float64
    N_tail::Int64
    Δt::Float64

    # constructor -
    MyJumpHiddenMarkovModel() = new();
end

# --- Validation report ------------------------------------------------------------------------------------------------- #

"""
    mutable struct MyValidationReportModel <: AbstractReportModel

The `MyValidationReportModel` mutable struct holds the results of comparing simulated paths to the empirical growth rates.
The vector fields hold one value for each simulated path.

### Required fields
- `ks_pvalue::Array{Float64,1}`: The Kolmogorov-Smirnov p-value of each path
- `ad_pvalue::Array{Float64,1}`: The Anderson-Darling p-value of each path
- `wasserstein::Array{Float64,1}`: The Wasserstein distance between each path and the empirical growth rates
- `hellinger::Array{Float64,1}`: The Hellinger distance between each path and the empirical growth rates
- `acf_mae::Array{Float64,1}`: The mean absolute error between the autocorrelation of each path and the empirical autocorrelation
- `kurtosis_sim::Array{Float64,1}`: The excess kurtosis of each path
- `kurtosis_emp::Float64`: The excess kurtosis of the empirical growth rates
- `ks_pass_rate::Float64`: The fraction of paths with a Kolmogorov-Smirnov p-value above `α`
- `ad_pass_rate::Float64`: The fraction of paths with an Anderson-Darling p-value above `α`
- `mean_acf_mae::Float64`: The mean of `acf_mae` over the paths
- `mean_wasserstein::Float64`: The mean of `wasserstein` over the paths
- `mean_hellinger::Float64`: The mean of `hellinger` over the paths
- `mean_kurtosis::Float64`: The mean of `kurtosis_sim` over the paths
- `α::Float64`: The significance level of the tests
"""
mutable struct MyValidationReportModel <: AbstractReportModel

    # data -
    ks_pvalue::Array{Float64,1}
    ad_pvalue::Array{Float64,1}
    wasserstein::Array{Float64,1}
    hellinger::Array{Float64,1}
    acf_mae::Array{Float64,1}
    kurtosis_sim::Array{Float64,1}
    kurtosis_emp::Float64
    ks_pass_rate::Float64
    ad_pass_rate::Float64
    mean_acf_mae::Float64
    mean_wasserstein::Float64
    mean_hellinger::Float64
    mean_kurtosis::Float64
    α::Float64

    # constructor -
    MyValidationReportModel() = new();
end

# --- Copula models ----------------------------------------------------------------------------------------------------- #

"""
    mutable struct MyGaussianCopulaModel <: AbstractDependenceModel

The `MyGaussianCopulaModel` mutable struct holds a Gaussian copula for several assets.

### Required fields
- `Σ::Array{Float64,2}`: The correlation matrix of the copula
"""
mutable struct MyGaussianCopulaModel <: AbstractDependenceModel

    # data -
    Σ::Array{Float64,2}

    # constructor -
    MyGaussianCopulaModel() = new();
end

"""
    mutable struct MyStudentTCopulaModel <: AbstractDependenceModel

The `MyStudentTCopulaModel` mutable struct holds a Student-t copula for several assets.

### Required fields
- `Σ::Array{Float64,2}`: The correlation matrix of the copula
- `ν::Float64`: The degrees of freedom of the copula
"""
mutable struct MyStudentTCopulaModel <: AbstractDependenceModel

    # data -
    Σ::Array{Float64,2}
    ν::Float64

    # constructor -
    MyStudentTCopulaModel() = new();
end

"""
    mutable struct MyGaussianBivariateCopulaModel <: AbstractBivariateCopulaModel

The `MyGaussianBivariateCopulaModel` mutable struct holds a Gaussian copula for two assets.

### Required fields
- `ρ::Float64`: The correlation of the copula
"""
mutable struct MyGaussianBivariateCopulaModel <: AbstractBivariateCopulaModel

    # data -
    ρ::Float64

    # constructor -
    MyGaussianBivariateCopulaModel() = new();
end

"""
    mutable struct MyStudentTBivariateCopulaModel <: AbstractBivariateCopulaModel

The `MyStudentTBivariateCopulaModel` mutable struct holds a Student-t copula for two assets.

### Required fields
- `ρ::Float64`: The correlation of the copula
- `ν::Float64`: The degrees of freedom of the copula
"""
mutable struct MyStudentTBivariateCopulaModel <: AbstractBivariateCopulaModel

    # data -
    ρ::Float64
    ν::Float64

    # constructor -
    MyStudentTBivariateCopulaModel() = new();
end

"""
    mutable struct MyClaytonBivariateCopulaModel <: AbstractBivariateCopulaModel

The `MyClaytonBivariateCopulaModel` mutable struct holds a Clayton copula for two assets.

### Required fields
- `θ::Float64`: The dependence parameter of the copula
"""
mutable struct MyClaytonBivariateCopulaModel <: AbstractBivariateCopulaModel

    # data -
    θ::Float64

    # constructor -
    MyClaytonBivariateCopulaModel() = new();
end

"""
    mutable struct MyGumbelBivariateCopulaModel <: AbstractBivariateCopulaModel

The `MyGumbelBivariateCopulaModel` mutable struct holds a Gumbel copula for two assets.

### Required fields
- `θ::Float64`: The dependence parameter of the copula
"""
mutable struct MyGumbelBivariateCopulaModel <: AbstractBivariateCopulaModel

    # data -
    θ::Float64

    # constructor -
    MyGumbelBivariateCopulaModel() = new();
end

"""
    mutable struct MyFrankBivariateCopulaModel <: AbstractBivariateCopulaModel

The `MyFrankBivariateCopulaModel` mutable struct holds a Frank copula for two assets.

### Required fields
- `θ::Float64`: The dependence parameter of the copula
"""
mutable struct MyFrankBivariateCopulaModel <: AbstractBivariateCopulaModel

    # data -
    θ::Float64

    # constructor -
    MyFrankBivariateCopulaModel() = new();
end

# --- Vine copula ------------------------------------------------------------------------------------------------------- #

"""
    mutable struct MyVineEdgeModel <: AbstractVineEdgeModel

The `MyVineEdgeModel` mutable struct holds one edge of a vine copula. The edge links two assets, given the conditioned assets, through a bivariate copula.

### Required fields
- `first::Int64`: The index of the first asset of the edge
- `second::Int64`: The index of the second asset of the edge
- `conditioned::Array{Int64,1}`: The indices of the assets that the edge is conditioned on
- `copula::AbstractBivariateCopulaModel`: The bivariate copula of the edge
"""
mutable struct MyVineEdgeModel <: AbstractVineEdgeModel

    # data -
    first::Int64
    second::Int64
    conditioned::Array{Int64,1}
    copula::AbstractBivariateCopulaModel

    # constructor -
    MyVineEdgeModel() = new();
end

"""
    mutable struct MyVineCopulaModel <: AbstractDependenceModel

The `MyVineCopulaModel` mutable struct holds a vine copula for several assets.

### Required fields
- `number_of_assets::Int64`: The number of assets
- `order::Array{Int64,1}`: The order of the assets in the vine
- `edges::Array{Array{MyVineEdgeModel,1},1}`: The edges of the vine. Entry `k` holds the edges of tree `k`
"""
mutable struct MyVineCopulaModel <: AbstractDependenceModel

    # data -
    number_of_assets::Int64
    order::Array{Int64,1}
    edges::Array{Array{MyVineEdgeModel,1},1}

    # constructor -
    MyVineCopulaModel() = new();
end

# --- Dependence models ------------------------------------------------------------------------------------------------- #

"""
    mutable struct MySingleIndexDependenceModel <: AbstractDependenceModel

The `MySingleIndexDependenceModel` mutable struct holds a single-index model of the growth rates of several assets:
`g_i = α_i + β_i g_m + ε_i`, where `g_m` is the growth rate of the market.

### Required fields
- `α::Array{Float64,1}`: The intercept of each asset
- `β::Array{Float64,1}`: The sensitivity of each asset to the market
- `σ_ε::Array{Float64,1}`: The standard deviation of the residuals of each asset
- `residuals::Array{Float64,2}`: The residuals of the fit. Each column holds the residuals of one asset
- `residual_method::Symbol`: The method used to draw the residuals when simulating
- `market_model::MyJumpHiddenMarkovModel`: The hidden Markov model of the market
"""
mutable struct MySingleIndexDependenceModel <: AbstractDependenceModel

    # data -
    α::Array{Float64,1}
    β::Array{Float64,1}
    σ_ε::Array{Float64,1}
    residuals::Array{Float64,2}
    residual_method::Symbol
    market_model::MyJumpHiddenMarkovModel

    # constructor -
    MySingleIndexDependenceModel() = new();
end

"""
    mutable struct MyHybridSingleIndexDependenceModel <: AbstractDependenceModel

The `MyHybridSingleIndexDependenceModel` mutable struct holds a single-index model that keeps the marginal model of each asset.
A copula, fit to the residuals of the single-index fit, links the assets.

### Required fields
- `α::Array{Float64,1}`: The intercept of each asset
- `β::Array{Float64,1}`: The sensitivity of each asset to the market
- `r²::Array{Float64,1}`: The R² of the single-index fit of each asset
- `σ_market::Float64`: The annualized standard deviation of the growth rates of the market
- `marginals::Dict{String,MyJumpHiddenMarkovModel}`: The hidden Markov model of each asset, keyed by ticker
- `copula::AbstractDependenceModel`: The copula fit to the residuals
- `market_model::MyJumpHiddenMarkovModel`: The hidden Markov model of the market
- `tickers::Array{String,1}`: The ordered tickers. The order matches `α`, `β`, and `r²`
- `r2_preserve_threshold::Float64`: The R² above which the model preserves the R² of the asset
- `idiosyncratic_floor::Float64`: The smallest fraction of the variance of an asset that is idiosyncratic
"""
mutable struct MyHybridSingleIndexDependenceModel <: AbstractDependenceModel

    # data -
    α::Array{Float64,1}
    β::Array{Float64,1}
    r²::Array{Float64,1}
    σ_market::Float64
    marginals::Dict{String,MyJumpHiddenMarkovModel}
    copula::AbstractDependenceModel
    market_model::MyJumpHiddenMarkovModel
    tickers::Array{String,1}
    r2_preserve_threshold::Float64
    idiosyncratic_floor::Float64

    # constructor -
    MyHybridSingleIndexDependenceModel() = new();
end

"""
    mutable struct MySectorFactorModel <: AbstractDependenceModel

The `MySectorFactorModel` mutable struct holds a model with one market model, one model for each sector, and one own model for each asset:
`g_i(t) = α_i + β_i(t) g_m(t) + δ_i f_s(t) + ψ_i (ε_i(t) - ε_center[i])`.
The own model of an asset is fit to the residual of the single-index fit after the sector term.
The assets are linked only through the market draw and the sector draw on the same day.

To change the intercepts or the drift of `β`, use `with_parameters`. After changing the market model, a sector model, or an own model,
call `build` again with the keys `market_model`, `sector_models`, and `own_models`, so that `ε_center` is computed from the changed models.

### Required fields
- `tickers::Array{String,1}`: The assets, without the market, in fit order
- `sector::Array{String,1}`: The sector label of each asset
- `sectors::Array{String,1}`: The sorted unique sector labels
- `sector_index::Array{Int64,1}`: The position of the sector of each asset in `sectors`
- `α::Array{Float64,1}`: The intercept of each asset, in 1/year
- `β::Array{Float64,1}`: The sensitivity of each asset to the market from the single-index fit
- `δ::Array{Float64,1}`: The sector loading of each asset
- `r²::Array{Float64,1}`: The R² of the single-index fit of each asset
- `ψ::Array{Float64,1}`: The scale of the own draw of each asset, computed by `build` and `with_parameters`. Editing `ψ` by hand has no effect, because `compose_name` computes the scale again from the variances
- `σ²_own::Array{Float64,1}`: The residual variance left to the own term of each asset
- `σ²_u::Array{Float64,1}`: The variance of the training series of each own model
- `σ²_m::Float64`: The variance of the growth rate of the market over the fit window
- `ε_center::Array{Float64,1}`: The long-run mean of each own model
- `β_halflife::Float64`: The half-life of the drift of `β`, in trading days
- `β_scale::Array{Float64,1}`: The drift scale of `β` for each asset. A value of `0.0` keeps `β` fixed
- `market_model::MyJumpHiddenMarkovModel`: The hidden Markov model of the market
- `sector_models::Array{MyJumpHiddenMarkovModel,1}`: The hidden Markov model of each sector, in the order of `sectors`
- `own_models::Array{MyJumpHiddenMarkovModel,1}`: The own hidden Markov model of each asset, in the order of `tickers`
- `sector_levels::Array{Float64,2}`: The compounded sector-factor series, used when tuning
- `own_levels::Array{Float64,2}`: The compounded own-residual series, used when tuning
- `center_steps::Int64`: The length of the draw that sets the noise center of each asset
- `center_seed::Int64`: The seed offset of the centering draw. Asset `i` uses `center_seed + i`
"""
mutable struct MySectorFactorModel <: AbstractDependenceModel

    # data -
    tickers::Array{String,1}
    sector::Array{String,1}
    sectors::Array{String,1}
    sector_index::Array{Int64,1}
    α::Array{Float64,1}
    β::Array{Float64,1}
    δ::Array{Float64,1}
    r²::Array{Float64,1}
    ψ::Array{Float64,1}
    σ²_own::Array{Float64,1}
    σ²_u::Array{Float64,1}
    σ²_m::Float64
    ε_center::Array{Float64,1}
    β_halflife::Float64
    β_scale::Array{Float64,1}
    market_model::MyJumpHiddenMarkovModel
    sector_models::Array{MyJumpHiddenMarkovModel,1}
    own_models::Array{MyJumpHiddenMarkovModel,1}
    sector_levels::Array{Float64,2}
    own_levels::Array{Float64,2}
    center_steps::Int64
    center_seed::Int64

    # constructor -
    MySectorFactorModel() = new();
end

# --- Portfolio --------------------------------------------------------------------------------------------------------- #

"""
    mutable struct MyPortfolioModel <: AbstractPortfolioModel

The `MyPortfolioModel` mutable struct holds a model of several assets: one hidden Markov model for each asset and a dependence model that links the assets.

### Required fields
- `tickers::Array{String,1}`: The tickers of the assets. A single-index, hybrid, or sector factor portfolio leaves the market ticker out
- `marginals::Dict{String,MyJumpHiddenMarkovModel}`: The hidden Markov model of each asset, keyed by ticker
- `dependence::AbstractDependenceModel`: The model that links the assets
- `tickers_map::Dict{String,Int64}`: The column of each ticker in the price matrix used to fit the model
- `market_ticker::String`: The ticker of the market
- `rf::Float64`: The risk-free rate, in 1/year
- `Δt::Float64`: The time step of the growth rates, in years
"""
mutable struct MyPortfolioModel <: AbstractPortfolioModel

    # data -
    tickers::Array{String,1}
    marginals::Dict{String,MyJumpHiddenMarkovModel}
    dependence::AbstractDependenceModel
    tickers_map::Dict{String,Int64}
    market_ticker::String
    rf::Float64
    Δt::Float64

    # constructor -
    MyPortfolioModel() = new();
end
