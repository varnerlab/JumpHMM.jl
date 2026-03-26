# Portfolio

## Overview
The `PortfolioModel` combines marginal Jump-HMMs with a dependence model to simulate correlated multi-asset portfolios. The interface is identical regardless of which dependence model is used.

## Copula Approach
```julia
tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
price_matrix = [...]  # (n_obs × 5)

portfolio = fit(PortfolioModel, tickers, price_matrix;
    dependence=StudentTCopula, rf=0.05, N=100)

portfolio = tune(portfolio, price_matrix)
result = simulate(portfolio, 252; n_paths=1000)
reports = validate(portfolio, price_matrix)
```

For copula-based portfolios, simulation uses the rank-reorder method: each marginal is simulated independently, then observations are reordered according to copula-sampled ranks to inject cross-asset dependence while preserving each marginal's distributional properties.

## Single-Index Model Approach
```julia
tickers = ["SPY", "AAPL", "MSFT", ...]  # include market ticker
price_matrix = [...]

portfolio = fit(PortfolioModel, tickers, price_matrix;
    dependence=SingleIndexModel, market="SPY", rf=0.05)

portfolio = tune(portfolio, price_matrix)
result = simulate(portfolio, 252; n_paths=1000)
```

For SIM portfolios, the market ticker is removed from `portfolio.tickers` but its column mapping is stored internally. This means `tune` and `validate` automatically use the correct columns from the original price matrix, including tuning the market model.

## Column Mapping
During `fit`, the `PortfolioModel` stores a `tickers_map` dictionary mapping each ticker (including the market ticker for SIM) to its column index in the original price matrix. This mapping is used automatically by `tune` and `validate`, so you can pass the same price matrix used for fitting:

```julia
# These work correctly — column mapping is automatic
portfolio = tune(portfolio, price_matrix)
reports = validate(portfolio, price_matrix)
```

You can also override the mapping explicitly:
```julia
custom_map = Dict("AAPL" => 1, "MSFT" => 2)
reports = validate(portfolio, new_prices; tickers_map=custom_map)
```

## Comparing Approaches
```julia
portfolio_copula = fit(PortfolioModel, tickers, prices;
    dependence=StudentTCopula, rf=0.05)
portfolio_sim = fit(PortfolioModel, tickers, prices;
    dependence=SingleIndexModel, market="SPY", rf=0.05)

reports_copula = validate(portfolio_copula, prices)
reports_sim = validate(portfolio_sim, prices)
```

## API
