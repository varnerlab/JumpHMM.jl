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

## Single-Index Model Approach
```julia
tickers = ["SPY", "AAPL", "MSFT", ...]  # include market ticker
price_matrix = [...]

portfolio = fit(PortfolioModel, tickers, price_matrix;
    dependence=SingleIndexModel, market="SPY", rf=0.05)

portfolio = tune(portfolio, price_matrix)
result = simulate(portfolio, 252; n_paths=1000)
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
