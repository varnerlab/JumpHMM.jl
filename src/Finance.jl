"""
    excess_growth_rates(prices; rf=0.0, dt=1/252) → Vector{Float64}

Compute annualized excess log-growth rates:
    G_t = (1/dt) × ln(P_t / P_{t-1}) - rf
"""
function excess_growth_rates(prices::AbstractVector{<:Real};
                             rf::Float64=0.0, dt::Float64=1/252)
    n = length(prices)
    G = Vector{Float64}(undef, n - 1)
    for i in 1:(n-1)
        G[i] = (1.0 / dt) * log(prices[i+1] / prices[i]) - rf
    end
    return G
end

"""
    excess_growth_rates(prices; rf=0.0, dt=1/252) → Matrix{Float64}

Column-wise excess growth rates from an (n_obs × n_assets) price matrix.
"""
function excess_growth_rates(prices::AbstractMatrix{<:Real};
                             rf::Float64=0.0, dt::Float64=1/252)
    n, m = size(prices)
    G = Matrix{Float64}(undef, n - 1, m)
    for j in 1:m
        for i in 1:(n-1)
            G[i, j] = (1.0 / dt) * log(prices[i+1, j] / prices[i, j]) - rf
        end
    end
    return G
end

"""
    excess_growth_rates(data, tickers; rf=0.0, dt=1/252, price_col=:close) → Matrix{Float64}

Compute excess growth rates from a Dict{String, DataFrame} of price data.
"""
function excess_growth_rates(data::Dict{String,DataFrame},
                             tickers::Vector{String};
                             rf::Float64=0.0, dt::Float64=1/252,
                             price_col::Symbol=:close)
    n = nrow(data[tickers[1]])
    m = length(tickers)
    prices = Matrix{Float64}(undef, n, m)
    for (j, ticker) in enumerate(tickers)
        prices[:, j] = data[ticker][!, price_col]
    end
    return excess_growth_rates(prices; rf=rf, dt=dt)
end

"""
    prices_from_growth_rates(G, P0; rf=0.0, dt=1/252) → Vector{Float64}

Reconstruct a price path from excess growth rates and an initial price.
"""
function prices_from_growth_rates(G::AbstractVector{Float64}, P0::Float64;
                                  rf::Float64=0.0, dt::Float64=1/252)
    n = length(G)
    P = Vector{Float64}(undef, n + 1)
    P[1] = P0
    for i in 1:n
        P[i+1] = P[i] * exp((G[i] + rf) * dt)
    end
    return P
end
