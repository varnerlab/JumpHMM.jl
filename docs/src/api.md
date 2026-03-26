# API Reference

## Types

### Abstract Types
```@docs
AbstractMarkovModel
AbstractDependenceModel
AbstractValidationResult
```

### Model Types
```@docs
LaplacePartition
StudentTEmission
JumpParameters
JumpHiddenMarkovModel
```

### Simulation Types
```@docs
SimulationPath
SimulationResult
```

### Validation Types
```@docs
PathTestResult
ValidationReport
```

### Dependence Models
```@docs
GaussianCopula
StudentTCopula
SingleIndexModel
```

### Portfolio Types
```@docs
PortfolioModel
PortfolioSimulationResult
```

## Functions

### Fitting
```@docs
fit(::Type{JumpHiddenMarkovModel}, ::AbstractVector{<:Real})
fit(::Type{LaplacePartition}, ::AbstractVector{Float64})
fit(::Type{GaussianCopula}, ::AbstractMatrix{Float64})
fit(::Type{StudentTCopula}, ::AbstractMatrix{Float64})
fit(::Type{SingleIndexModel}, ::AbstractMatrix{Float64}, ::AbstractVector{Float64})
fit(::Type{PortfolioModel}, ::Vector{String}, ::AbstractMatrix{<:Real})
assign_states
estimate_transition
stationary_distribution
fit_emissions
sample_emission
```

### Tuning
```@docs
tune(::JumpHiddenMarkovModel, ::AbstractVector{<:Real})
tune(::PortfolioModel, ::AbstractMatrix{<:Real})
```

### Simulation
```@docs
simulate(::JumpHiddenMarkovModel, ::Int)
simulate(::PortfolioModel, ::Int)
```

### Decoding
```@docs
decode
forward_filter
log_likelihood
```

### Validation
```@docs
validate(::JumpHiddenMarkovModel, ::AbstractVector{<:Real})
validate(::PortfolioModel, ::AbstractMatrix{<:Real})
```

### Dependence
```@docs
sample_dependence(::GaussianCopula, ::Int)
sample_dependence(::StudentTCopula, ::Int)
sample_dependence(::SingleIndexModel, ::Int)
```

### Finance Helpers
```@docs
excess_growth_rates(::AbstractVector{<:Real})
excess_growth_rates(::AbstractMatrix{<:Real})
excess_growth_rates(::Dict{String, DataFrame}, ::Vector{String})
prices_from_growth_rates
```
