"""
The JumpHMM package builds a hidden Markov model with jump episodes from the growth rates of a stock,
and simulates synthetic growth rates for one asset or a portfolio.
"""
module JumpHMM

    # include -
    include("Include.jl")

    # export abstract types -
    export AbstractMarkovModel, AbstractDependenceModel, AbstractBivariateCopulaModel, AbstractReportModel

    # export concrete types -
    export MyJumpHiddenMarkovModel, MyValidationReportModel
    export MyGaussianCopulaModel, MyStudentTCopulaModel
    export MyGaussianBivariateCopulaModel, MyStudentTBivariateCopulaModel
    export MyClaytonBivariateCopulaModel, MyGumbelBivariateCopulaModel, MyFrankBivariateCopulaModel
    export MyVineEdgeModel, MyVineCopulaModel
    export MySingleIndexDependenceModel, MyHybridSingleIndexDependenceModel, MySectorFactorModel
    export MyPortfolioModel

    # growth rates -
    export excess_growth_rates, prices_from_growth_rates

    # build and estimate a single-asset model -
    export build
    export laplace_boundaries, assign_states, estimate_transition, stationary_distribution
    export estimate_emissions, sample_emission

    # simulate synthetic growth rates -
    export simulate

    # tune the jump settings -
    export tune

    # validate a model against the data, and compare two samples -
    export validate, wasserstein_distance, hellinger_distance

    # decode states, filter state probabilities, and compute the log-likelihood -
    export decode, forward_filter, log_likelihood

    # draw dependent samples: uniform numbers from a copula, or growth rates from a dependence model -
    export sample_dependence

    # draw and compose the paths of a sector factor model -
    export sample_draws, compose, compose_name, with_parameters
end
