# API Reference

## Types

### Abstract Types
```@docs
JumpHMM.AbstractMarkovModel
JumpHMM.AbstractDependenceModel
JumpHMM.AbstractValidationResult
```

### Model Types
```@docs
JumpHMM.LaplacePartition
JumpHMM.StudentTEmission
JumpHMM.JumpParameters
JumpHMM.JumpHiddenMarkovModel
```

### Simulation Types
```@docs
JumpHMM.SimulationPath
JumpHMM.SimulationResult
```

### Validation Types
```@docs
JumpHMM.PathTestResult
JumpHMM.ValidationReport
```

### Dependence Models
```@docs
JumpHMM.GaussianCopula
JumpHMM.StudentTCopula
JumpHMM.VineCopula
JumpHMM.SingleIndexModel
```

### Bivariate Copula Families
```@docs
JumpHMM.AbstractBivariateCopula
JumpHMM.GaussianBiCopula
JumpHMM.StudentTBiCopula
JumpHMM.ClaytonBiCopula
JumpHMM.GumbelBiCopula
JumpHMM.FrankBiCopula
JumpHMM.VineEdge
```

### Portfolio Types
```@docs
JumpHMM.PortfolioModel
JumpHMM.PortfolioSimulationResult
```

## Functions

### Fitting
```@docs
JumpHMM.fit(::Type{JumpHiddenMarkovModel}, ::AbstractVector{<:Real})
JumpHMM.fit(::Type{LaplacePartition}, ::AbstractVector{Float64})
JumpHMM.fit(::Type{GaussianCopula}, ::AbstractMatrix{Float64})
JumpHMM.fit(::Type{StudentTCopula}, ::AbstractMatrix{Float64})
JumpHMM.fit(::Type{VineCopula}, ::AbstractMatrix{Float64})
JumpHMM.fit(::Type{SingleIndexModel}, ::AbstractMatrix{Float64}, ::AbstractVector{Float64})
JumpHMM.fit(::Type{PortfolioModel}, ::Vector{String}, ::AbstractMatrix{<:Real})
JumpHMM.assign_states
JumpHMM.estimate_transition
JumpHMM.stationary_distribution
JumpHMM.fit_emissions
JumpHMM.sample_emission
```

### Tuning
```@docs
JumpHMM.tune(::JumpHiddenMarkovModel, ::AbstractVector{<:Real})
JumpHMM.tune(::PortfolioModel, ::AbstractMatrix{<:Real})
```

### Simulation
```@docs
JumpHMM.simulate(::JumpHiddenMarkovModel, ::Int)
JumpHMM.simulate(::PortfolioModel, ::Int)
```

### Decoding
```@docs
JumpHMM.decode
JumpHMM.forward_filter
JumpHMM.log_likelihood
```

### Validation
```@docs
JumpHMM.validate(::JumpHiddenMarkovModel, ::AbstractVector{<:Real})
JumpHMM.validate(::PortfolioModel, ::AbstractMatrix{<:Real})
```

### Dependence
```@docs
JumpHMM.sample_dependence(::GaussianCopula, ::Int)
JumpHMM.sample_dependence(::StudentTCopula, ::Int)
JumpHMM.sample_dependence(::VineCopula, ::Int)
JumpHMM.sample_dependence(::SingleIndexModel, ::Int)
```

### Finance Helpers
```@docs
JumpHMM.excess_growth_rates(::AbstractVector{<:Real})
JumpHMM.excess_growth_rates(::AbstractMatrix{<:Real})
JumpHMM.excess_growth_rates(::Dict{String, DataFrame}, ::Vector{String})
JumpHMM.prices_from_growth_rates
```
