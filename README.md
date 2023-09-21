# Climate-risk Modelling and Portfolio Decarbonisation

This project has two parts - 
1. Climate-risk Modelling (formulates several hypotheses regarding relationships between independent financial variables and carbon emissions, and employs regression models to perform statistical tests to uncover the true relationships)
2. Portfolio Decarbonisation (uses portfolio optimisation techniques to reduce portfolio carbon footprint while maximising returns)

\
\
The `PortfolioDecarbonisation` class is a Python-based financial tool used for generating and analysing portfolios with decarbonisation constraints. The class includes a wide range of features such as downloading, processing, and grouping data, as well as the creation and analysis of multiple types of portfolios.

## Features

1. **Data Processing**: This feature takes raw financial and environmental data and processes it into a usable form. It also groups the data by sector for further analysis.

2. **Descriptive Statistics**: This feature calculates descriptive statistics for each portfolio.

3. **Portfolio Creation**: This class allows for the creation of several types of portfolios. These include:

    - Benchmark Portfolio: A market-cap weighted portfolio.
    - Decarbonised Portfolios: These portfolios aim to reduce their carbon footprint by a specified percentage.
    - Mean-Variance Efficient Portfolio: These portfolios aim to maximize return for a given level of risk.
    - Mean-Variance Efficient Decarbonised Portfolio: These portfolios aim to reduce their carbon footprint by a specified percentage while also being mean-variance efficient.

4. **Portfolio Analysis**: The class calculates and prints a wide range of portfolio statistics, including the number of companies invested in, Sharpe ratio, expected return, and standard deviation. 

5. **ESG Performance and Sector Composition Analysis**: The class also analyses and prints out the ESG (Environmental, Social, and Governance) performance and sector composition for each portfolio. 

## How to Use

The class is designed to be straightforward to use. First, an instance of the `PortfolioDecarbonisation` class is created with the filename of the data source as an argument. Then, the `run_portfolio_decarbonisation` method is called on the instance with an instance of the `BaseConvexOptimizer` class as an argument.

```python
if __name__ == "__main__":
    PD = PortfolioDecarbonisation('R3000 - V02.xlsx')
    PD.run_portfolio_decarbonisation(base_optimizer=base_optimizer)
```

## Dependencies

- pandas
- numpy
- cvxpy
- PyPortfolioOpt
