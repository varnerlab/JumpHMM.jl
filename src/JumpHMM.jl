"""
`JumpHMM` fits a hidden Markov model with jump episodes to stock growth rates
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
end
