"""
    excess_growth_rates(prices::Array{Float64,1}; rf::Float64 = 0.0, Δt::Float64 = 1/252) -> Array{Float64,1}

Compute the annualized excess growth rates of a price series: `(1/Δt) * ln(P[t+1]/P[t]) - rf`.

### Arguments
- `prices::Array{Float64,1}`: The prices, one for each time step

### Returns
- `Array{Float64,1}`: The growth rates. The array has one fewer element than `prices`

### Optional Arguments
- `rf::Float64`: The risk-free rate, in 1/year. The default is `0.0`
- `Δt::Float64`: The time step, in years. The default is `1/252`

### Example
```julia
growth_rates = JumpHMM.excess_growth_rates([100.0, 101.0, 99.5]; rf = 0.02, Δt = 1/252);
```
"""
function excess_growth_rates(prices::Array{Float64,1}; rf::Float64 = 0.0, Δt::Float64 = 1/252)::Array{Float64,1}

    # initialize -
    number_of_prices = length(prices);
    growth_rates = Array{Float64,1}(undef, number_of_prices - 1);

    # main loop -
    for i ∈ 1:(number_of_prices - 1)
        growth_rates[i] = (1.0 / Δt) * log(prices[i+1] / prices[i]) - rf;
    end

    # return -
    return growth_rates;
end

"""
    excess_growth_rates(prices::Array{Float64,2}; rf::Float64 = 0.0, Δt::Float64 = 1/252) -> Array{Float64,2}

Compute the annualized excess growth rates of each column of a price matrix.

### Arguments
- `prices::Array{Float64,2}`: The prices. Each row is a time step and each column is an asset

### Returns
- `Array{Float64,2}`: The growth rates. The array has one fewer row than `prices`

### Optional Arguments
- `rf::Float64`: The risk-free rate, in 1/year. The default is `0.0`
- `Δt::Float64`: The time step, in years. The default is `1/252`

### Example
```julia
prices = [100.0 50.0; 101.0 49.0; 99.5 51.0];
growth_rates = JumpHMM.excess_growth_rates(prices; rf = 0.02, Δt = 1/252);
```
"""
function excess_growth_rates(prices::Array{Float64,2}; rf::Float64 = 0.0, Δt::Float64 = 1/252)::Array{Float64,2}

    # initialize -
    (number_of_prices, number_of_assets) = size(prices);
    growth_rates = Array{Float64,2}(undef, number_of_prices - 1, number_of_assets);

    # main loop -
    for j ∈ 1:number_of_assets
        for i ∈ 1:(number_of_prices - 1)
            growth_rates[i,j] = (1.0 / Δt) * log(prices[i+1,j] / prices[i,j]) - rf;
        end
    end

    # return -
    return growth_rates;
end

"""
    excess_growth_rates(data::Dict{String,DataFrame}, tickers::Array{String,1}; rf::Float64 = 0.0, Δt::Float64 = 1/252, price_col::Symbol = :close) -> Array{Float64,2}

Compute the annualized excess growth rates of the tickers in a dictionary of price tables.

### Arguments
- `data::Dict{String,DataFrame}`: The price table of each ticker. Every table must have the same number of rows
- `tickers::Array{String,1}`: The tickers to use. Column `j` of the result is `tickers[j]`

### Returns
- `Array{Float64,2}`: The growth rates. The array has one fewer row than each price table

### Optional Arguments
- `rf::Float64`: The risk-free rate, in 1/year. The default is `0.0`
- `Δt::Float64`: The time step, in years. The default is `1/252`
- `price_col::Symbol`: The column of each table that holds the prices. The default is `:close`

### Example
```julia
data = Dict("AAA" => DataFrames.DataFrame(close = [100.0, 101.0, 99.5]));
growth_rates = JumpHMM.excess_growth_rates(data, ["AAA"]);
```
"""
function excess_growth_rates(data::Dict{String,DataFrame}, tickers::Array{String,1};
    rf::Float64 = 0.0, Δt::Float64 = 1/252, price_col::Symbol = :close)::Array{Float64,2}

    # initialize -
    number_of_prices = nrow(data[tickers[1]]);
    number_of_assets = length(tickers);
    prices = Array{Float64,2}(undef, number_of_prices, number_of_assets);

    # collect the prices of each ticker into a column -
    for j ∈ 1:number_of_assets
        prices[:,j] = data[tickers[j]][!, price_col];
    end

    # return -
    return excess_growth_rates(prices; rf = rf, Δt = Δt);
end

"""
    prices_from_growth_rates(growth_rates::Array{Float64,1}, P₀::Float64; rf::Float64 = 0.0, Δt::Float64 = 1/252) -> Array{Float64,1}

Rebuild a price series from excess growth rates and the first price.

### Arguments
- `growth_rates::Array{Float64,1}`: The annualized excess growth rates
- `P₀::Float64`: The first price

### Returns
- `Array{Float64,1}`: The prices. The array has one more element than `growth_rates`

### Optional Arguments
- `rf::Float64`: The risk-free rate, in 1/year. The default is `0.0`
- `Δt::Float64`: The time step, in years. The default is `1/252`

### Example
```julia
prices = JumpHMM.prices_from_growth_rates([0.1, -0.2], 100.0; rf = 0.02, Δt = 1/252);
```
"""
function prices_from_growth_rates(growth_rates::Array{Float64,1}, P₀::Float64;
    rf::Float64 = 0.0, Δt::Float64 = 1/252)::Array{Float64,1}

    # initialize -
    number_of_steps = length(growth_rates);
    prices = Array{Float64,1}(undef, number_of_steps + 1);
    prices[1] = P₀;

    # main loop -
    for i ∈ 1:number_of_steps
        prices[i+1] = prices[i] * exp((growth_rates[i] + rf) * Δt);
    end

    # return -
    return prices;
end
