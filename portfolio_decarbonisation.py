import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import base_optimizer
import cvxpy as cp

import warnings
warnings.filterwarnings("ignore")


class PortfolioDecarbonisation:

    def __init__(self, filePath, risk_free_rate=0.0009):
        """
        Initialise the class with a file path (pointing to the data source) 
        & risk-free rate (in decimal format)
        

        Parameters:
            filePath (str): path to the Excel file
            risk_free_rate (float): Risk-free rate. Defaults to 0.0009 (0.09%)
        """
        self.filePath = filePath
        self.risk_free_rate = risk_free_rate

    def load_excel_data(self, sheetname, skiprows=None):
        """
        Load an excel file with given filename, sheetname and skiprows

        Parameters:
            sheetname (str): Sheetname in the Excel file
            skiprows (list): List of rows to skip from top. None by default

        Returns:
            DataFrame: A pandas DataFrame containing the data from the Excel file.
        """
        return pd.read_excel(self.filePath, sheet_name=sheetname, skiprows=skiprows)

    def rename_dataframe_columns(self, df, column_mapping):
        """
        Rename the columns in a DataFrame based on a mapping

        Parameters:
            df (DataFrame): DataFrame to be renamed
            column_mapping (dict): A dictionary mapping old column names to new column names

        Returns:
            DataFrame: The DataFrame with renamed columns.
        """
        df.rename(columns=column_mapping, inplace=True)
        return df

    def fix_unnamed_columns(self, df):
        """
        Fix 'Unnamed' columns in the dataframe

        Parameters:
            dataframe (DataFrame): The pandas DataFrame to be processed.

        Returns:
            DataFrame: The DataFrame with fixed 'Unnamed' columns.
        """
        df.columns = df.columns.to_series().mask(lambda x: x.str.startswith('Unnamed')).ffill()
        return df

    def create_multi_index(self, df):
        """
        Create MultiIndex df from the given df.

        Parameters:
            df (DataFrame): The pandas DataFrame to be processed.

        Returns:
            DataFrame: The MultiIndex DataFrame.
        """
        copy_df = df.copy()
        copy_df.drop(['Dates'], axis=1, inplace=True)
        multi_index = pd.MultiIndex.from_arrays([copy_df.columns, copy_df.iloc[0].values])

        df = df.drop([0]).reset_index(drop=True)
        df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce').dt.strftime('%Y-%m-%d')
        df.set_index(df['Dates'], inplace=True)
        df.drop(['Dates'], axis=1, inplace=True)
        df.columns = multi_index
        return df
    
    def _preprocess_data(self, df, new_columns, selected_columns=None):
        """
        Generic method to process financial data, emission data and ESG Score data

        Parameters:
            df (DataFrame): DataFrame to be processed
            new_columns (list): Column names for the DataFrame
            selected_columns (list): Columns to select from the processed DataFrame. None by default

        Returns:
            DataFrame: Processed DataFrame
        """
        df = df.stack(0, dropna=False)
        df.rename_axis(index=['Dates', 'Ticker'], inplace=True)
        df = df.swaplevel()
        df = df.sort_values(['Ticker', 'Dates'], ascending=[True, True])
        columns_to_convert = df.columns
        df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
        df.columns = new_columns
        
        if selected_columns is not None:
            df = df[selected_columns]
            
        return df

    def process_financial_data(self, df):
        """
        Process the financial data df to final form.

        Parameters:
            df (DataFrame): The pandas DataFrame to be processed.

        Returns:
            DataFrame: The final DataFrame.
        """
        new_columns = ['MarketCap', 'Revenue', 'TotalAssets', 'Total Equity']
        selected_columns = ['MarketCap', 'Revenue']
        return self._preprocess_data(df, new_columns, selected_columns)

    def process_emission_data(self, df):
        """
        Process the emission data dataframe to final form

        Parameters:
            df (DataFrame): The pandas DataFrame to be processed

        Returns:
            DataFrame: The final DataFrame.
        """
        new_columns = ['Scope1', 'Scope2Location', 'Scope2Market', 'Scope3']
        return self._preprocess_data(df, new_columns)

    def process_esg_score_data(self, df):
        """
        Process the ESG Score data df to final form.

        Parameters:
            df (DataFrame): The pandas DataFrame to be processed.

        Returns:
            DataFrame: The final DataFrame.
        """
        new_columns = ['ESG Score', 'Environmental Score', 'Governance Score', 'Social Score']
        return self._preprocess_data(df, new_columns)

    def data_etl(self, sheetnames=['Company List', 'Financial Data', 'GHG Emission', 'ESG Score'],
                     filter_date='2022-12-31'):
        """
        Load, process, and merge data from multiple sheets of an Excel file.

        Parameters:
            sheetnames (list): A list of sheet names in the Excel file
            filter_date (str): A date string in 'yyyy-mm-dd' format to filter the final DataFrame

        Returns:
            DataFrame: A pandas DataFrame containing processed data from all specified sheets
        """
        dataframes = {}
        for sheetname in sheetnames:
            df = self.load_excel_data(sheetname, skiprows=[0, 1, 2])
            df = self.fix_unnamed_columns(df)
            df.columns.values[0] = 'Dates'
            df = self.create_multi_index(df)
            dataframes[sheetname] = df

        # process each dataframe individually
        R3000list = self.rename_dataframe_columns(dataframes['Company List'], {'BBG Ticker': 'Ticker'})
        fin_panel = self.process_financial_data(dataframes['Financial Data'])
        emission_panel = self.process_emission_data(dataframes['GHG Emission'])
        esg_score_panel = self.process_esg_score_data(dataframes['ESG Score'])

        # merge all processed dataframes
        df1 = pd.merge(emission_panel, esg_score_panel, how='outer', on=['Ticker', 'Dates'])
        df2 = pd.merge(df1, fin_panel, how='outer', on=['Ticker', 'Dates'])
        # Filter for the specified date
        final_data = df2[df2.index.get_level_values('Dates') == filter_date]

        # Check for missing values in 'Scope1', 'Scope2Location', 'Revenue'
        drop_list = final_data[final_data[['Scope1', 'Scope2Location', 'Revenue']].isna().any(axis=1)]
        drop_list.reset_index(drop=False, inplace=True)
        drop_list = drop_list['Ticker']
        # Drop companies with missing values
        final_data = final_data.drop(drop_list, level=0, axis=0)

        return final_data

    def process_stock_price_data(self, sheetname='Stock', skiprows=[0, 1, 2]):
        """
        Load and process stock price data from an Excel file.

        Parameters:
            sheetname (str): Sheetname in Excel file
            skiprows (list): Skip first 3 rows

        Returns:
            DataFrame: Pandas DataFrame containing processed stock price data.
        """
        stock_data = self.load_excel_data(sheetname, skiprows) # Load stock price data from the Excel file
        # Fix 'Unnamed' columns
        stock_data = self.fix_unnamed_columns(stock_data)
        stock_data = stock_data.drop([0]).reset_index(drop=True)
        stock_data.columns.values[0] = 'Dates'
        # Create MultiIndex DataFrame
        stock_data = self.create_multi_index(stock_data)

        # Process the data
        stock_data = stock_data.stack(0, dropna=False)
        stock_data.rename_axis(index=['Dates', 'Ticker'], inplace=True)
        stock_data = stock_data.swaplevel()
        stock_data = stock_data.sort_values(['Ticker', 'Dates'], ascending=[True, True])
        stock_data = stock_data.astype(float)
        stock_data.columns = ['Closing Stock Price']
        stock_closing_price = stock_data.unstack().T
        stock_closing_price.reset_index(drop=False, inplace=True)
        stock_closing_price.set_index('Dates', inplace=True)
        stock_closing_price.drop('level_0', axis=1, inplace=True)
        stock_closing_price.sort_index(axis=1, inplace=True)
        ##### Checking for missing values
        stock_closing_price = stock_closing_price.dropna(axis='columns')

        return stock_closing_price
    
    def group_data_by_sector(self, financial_environmental_data, financial_stock_data):
        """
        Group the processed data by sector and sort by MarketCap

        Parameters:
            financial_environmental_data (DataFrame): DataFrame containing the financial and environmental data
            financial_stock_data (DataFrame): DataFrame containing the stock price data

        Returns:
            DataFrame: DataFrame grouped by 'GICS Sector' and 'Ticker' and sorted by 'MarketCap'
        """
        market_cap = pd.DataFrame(financial_environmental_data['MarketCap'])
        market_cap.reset_index(drop=False, inplace=True)
        market_cap.drop('Dates', axis=1, inplace=True)
        market_cap.set_index('Ticker', inplace=True)
        temp_data = market_cap.T
        R3000list = self.rename_dataframe_columns(self.load_excel_data('Company List'), {'BBG Ticker': 'Ticker'})
        common_tickers = list(set(list(financial_stock_data.columns)) & set(list(temp_data.columns)))
        temp_data = temp_data[common_tickers]
        market_cap = temp_data.T.reset_index(drop=False)
        companies_df = pd.merge(R3000list, market_cap, on='Ticker', how='inner')
        companies_df.drop(['ISIN', 'Company'], axis=1, inplace=True)
        sector_groups = pd.DataFrame(companies_df.groupby(['GICS Sector', 'Ticker'])['MarketCap'].mean()).sort_values(
            ['GICS Sector', 'MarketCap'], ascending=[True, False])

        return sector_groups
    
    def get_random_tickers(self, df, n=100):
        """
        Selects n random tickers from each sector in a DataFrame

        Parameters:
            df (DataFrame): Sector dataframe
            n (int): The total number of desired companies to form a portfolio, defaults to 100

        Returns:
            DataFrame: A new DataFrame containing only the random tickers selected from each sector.
        """
        # np.random.seed(1000)  
        # random_tickers = df.groupby(level=0).apply(lambda x: x.sample(n) if len(x) >= n else x).droplevel(0)
        # # If there are less than 100 tickers, randomly add more tickers from any sector
        # while len(random_tickers) < 100:
        #     # Choose a sector (randomly)
        #     sector = np.random.choice(df.index.get_level_values(0).unique())
        #     # Choose a ticker from the sector that is not already in the list
        #     additional_ticker = df.loc[sector].loc[~df.loc[sector].index.isin(random_tickers.index)].sample(1)
            
        #     if additional_ticker.index[0] not in random_tickers.index:
        #         # Preserve the sector information
        #         additional_ticker.index = pd.MultiIndex.from_tuples([(sector, additional_ticker.index[0])], names = ["Sector", "Ticker"])
        #         # Add the chosen ticker to the list
        #         random_tickers = pd.concat([random_tickers, additional_ticker])
        np.random.seed(64)
        random_tickers = df.sample(n=n, random_state=1)
        value_counts = random_tickers.index.get_level_values(0).value_counts()
        print(f'Sector Distribution = {value_counts.sum()}\n{value_counts}')

        return random_tickers
    
    def get_final_data(self, random_tickers, financial_environmental_data, financial_stock_data):
        """
        Filters the financial and environmental data to include only the randomly selected tickers

        Parameters:
            random_tickers (DataFrame): DataFrame containing randomly selected tickers from each sector
            financial_environmental_data (DataFrame): DataFrame containing the financial and environmental data
            financial_stock_data (DataFrame): DataFrame containing the stock price data

        Returns:
            DataFrame: DataFrame containing financial, environmental, and sector information for the randomly selected tickers
            DataFrame: DataFrame containing stock data and sector information for the randomly selected tickers
        """
        R3000list = self.rename_dataframe_columns(self.load_excel_data('Company List'), {'BBG Ticker': 'Ticker'})
        company_list = list(random_tickers.index.get_level_values('Ticker'))
        df_final = financial_environmental_data[financial_environmental_data.index.get_level_values('Ticker').isin(company_list)]
        df_final = pd.merge(df_final, R3000list[['Ticker', 'GICS Sector']], on='Ticker', how='left')
        df_final['Carbon Intensity'] = (df_final['Scope1'] + df_final['Scope2Location'])/df_final['Revenue']
        df_final.set_index('Ticker',inplace=True)
        stock_prices = financial_stock_data[company_list]
        stock_prices.sort_index(axis=1, inplace=True)

        return df_final, stock_prices
    
    def get_descriptive_statistics(self, df_final, df_stock):
        """
        Descriptive statistics for a given DataFrame

        Parameters:
            df_final (DataFrame): DataFrame for which to compute statistics
            df_stock (DataFrame): Stock data DataFrame for which to compute statistics

        Returns:
            dict: Summary stats 
        """
        market_cap = df_final['MarketCap']
        stats = {
            'mean': market_cap.mean(),
            'std_dev': market_cap.std(),
            'min': market_cap.min(),
            'max': market_cap.max(),
            'median': market_cap.median(),
            'mu': mean_historical_return(df_stock, frequency=12),
            'S': CovarianceShrinkage(df_stock, frequency=12).ledoit_wolf()
        }

        print(list(stats.items())[:5])
        return stats
    
    def get_benchmark_portfolio(self, df_final, mu):
        """
        Calculate benchmark portfolio & its weights based on market capitalization ratios.

        Parameters:
            df_final (DataFrame): Dataframe containing financial and environmental data for the 100 stocks
            mu (Series): Expected returns for all stocks
            
        Returns:
            DataFrame: Benchmark portfolio with weights, expected stock returns and scores
        """
        market_cap = df_final[['GICS Sector','MarketCap']]
        benchmark_portfolio = pd.DataFrame()
        benchmark_portfolio['GICS Sector'] = market_cap['GICS Sector']
        benchmark_portfolio['Weights'] = market_cap['MarketCap']/market_cap['MarketCap'].sum()
        benchmark_portfolio['ExpStockReturns'] = mu * benchmark_portfolio['Weights']
        portfolio_scores = ['Carbon Intensity', 'ESG Score', 'Environmental Score', 'Social Score', 'Governance Score']
        for score in portfolio_scores:
            benchmark_portfolio[f'Exp{score}'] = df_final[score] * benchmark_portfolio['Weights']
        
        return benchmark_portfolio
    
    def get_portfolio_performance(self, portfolio_weights, mu, S):
        """
        Calculate portfolio performance including expected return, volatility, sharpe ratio, and number of companies

        Parameters:
            portfolio_weights (DataFrame): Portfolio weights for each stock
            mu (Series): Expected returns for all stocks
            S (DataFrame): Covariance matrix of stock returns
            
        Returns:
            dict: A dictionary containing performance metrics
        """
        performance = base_optimizer.portfolio_performance(portfolio_weights['Weights'], mu, S, verbose=True,
                                                              risk_free_rate=self.risk_free_rate)
        return {
            'Annualised_ExpReturn': performance[0]*100,
            'Annualised_Volatility': performance[1]*100,
            'Sharpe_Ratio': performance[2],
            'Num_Companies': portfolio_weights[portfolio_weights['Weights'] >= 0].count()['Weights']
        }

    def print_portfolio_stats(self, portfolio_perf):
        """
        Print portfolio performance metrics.

        Parameters:
            portfolio_perf (dict): Portfolio performance metrics
            
        Returns:
            None
        """
        print("\n")
        print(f"Annualised Expected Return of Portfolio = {portfolio_perf['Annualised_ExpReturn']:.2f}%")
        print(f"Annualised Volatility of Portfolio = {portfolio_perf['Annualised_Volatility']:.2f}%")
        print(f"Annualised Sharpe Ratio of Portfolio = {portfolio_perf['Sharpe_Ratio']:.2f}")
        print(f"No. of companies constituting the Portfolio = {portfolio_perf['Num_Companies']}")

    def get_portfolio_scores(self, portfolio_weights, df_final, is_benchmark=False):
        """
        Calculate ESG scores for the portfolio

        Parameters:
            portfolio_weights (DataFrame): Portfolio weights for each stock
            df_final (DataFrame): Final DataFrame containing all financial and environmental data of 100 stocks
            is_benchmark (bool, optional): Whether the portfolio is a benchmark portfolio. Defaults to False.
            
        Returns:
            dict: A dictionary containing portfolio ESG scores
        """
        portfolio_scores = ['Carbon Intensity', 'ESG Score', 'Environmental Score', 'Social Score', 'Governance Score']
        scores = {}
        if is_benchmark:
            for score in portfolio_scores:
                modified_txt = "Exp" + score
                scores[score] = portfolio_weights[modified_txt].sum()
        else:  # If decarbonised portfolio
            for score in portfolio_scores:
                scores[score] = (df_final[score] * portfolio_weights).sum()
        return scores

    def print_portfolio_scores(self, portfolio_scores):
        """
        Print ESG scores for the portfolio.

        Parameters:
            portfolio_scores (dict): Portfolio ESG scores
            
        Returns:
            None
        """
        print("\n")
        print(f"Portfolio Carbon Intensity =  {portfolio_scores['Carbon Intensity']:.4f} CO2 (t)/ revenue ($mn)")
        print(f"Portfolio ESG Score =  {portfolio_scores['ESG Score']:.2f}")
        print(f"Portfolio Environmental Score = {portfolio_scores['Environmental Score']:.2f}")
        print(f"Portfolio Social Score = {portfolio_scores['Social Score']:.2f}")
        print(f"Portfolio Governance Score = {portfolio_scores['Governance Score']:.2f}")
        print("\n")
    
    def market_cap_portfolio(self, df_final, mu, S):
        """
        Calculate market cap weighted portfolio, its performance metrics, and ESG scores

        Args:
            df_final (DataFrame): Final DataFrame containing all financial and environmental data of 100 stocks
            mu (Series): Expected returns for all stocks
            S (DataFrame): Covariance matrix of stock returns
            
        Returns:
            Tuple: A tuple containing portfolio weights DataFrame, performance dictionary, and ESG scores dictionary
        """
        benchmark_portfolio = self.get_benchmark_portfolio(df_final, mu)
        benchmark_performance = self.get_portfolio_performance(benchmark_portfolio, mu, S)
        benchmark_scores = self.get_portfolio_scores(benchmark_portfolio, df_final, is_benchmark=True)
        self.print_portfolio_stats(benchmark_performance)
        self.print_portfolio_scores(benchmark_scores)
        
        return benchmark_portfolio, benchmark_performance, benchmark_scores

    def ex_ante_tracking_error(self, w, cov_matrix, benchmark_weights):
        """
        Computes ex-ante tracking error (variance) of portfolio's returns 
        with the benchmark's returns.

        Parameters:
            w (np.array): Weights of the selected portfolio
            cov_matrix (np.array): Covariance matrix of the returns of the stocks
            benchmark_weights (np.array): Weights of the benchmark portfolio
            
        Returns:
            float: A float value representing the variance of the excess returns
        """
        relative_weights = w - benchmark_weights
        variance = cp.quad_form(relative_weights, cov_matrix)
        return variance
    
    def sector_balance(self, final_data, random_tickers):
        """
        Balance the sector composition for the portfolio.

        Parameters:
            final_data (DataFrame): DataFrame containing all financial and environmental data
            random_tickers (DataFrame): DataFrame containing random stock tickers
            
        Returns:
            Tuple: A tuple containing sector mapping dictionary, lower bound for each sector, and upper bound for each sector
        """
        ticker_counts = random_tickers.groupby('GICS Sector').size()/100
        # Sector Mapping as a Dictionary
        sector_mapping = final_data.reset_index(drop=False)[['GICS Sector','Ticker']]
        sector_mapping.set_index('Ticker',inplace=True)
        sector_mapping_dict = sector_mapping.to_dict()['GICS Sector']
        lower = 0.7    # Lower Bound of change (-30%)
        upper = 1.3    # Upper Bound of change (+30%)
        sector_lower = {}
        sector_upper = {}
        for sector, count in ticker_counts.items():
            sector_lower[sector] = count * lower
            sector_upper[sector] = count * upper
        return sector_mapping_dict, sector_lower, sector_upper
         
    def decarbonised_portfolio(self, total_initial_carbon_intensity, intensity_reduction_rate, base_optimizer, stock_prices, 
                               mu, S, carbon_intensity, df_final, benchmark_weights, random_tickers=None, 
                               is_sector_balance_portfolio = False ):
        """
        Creates a portfolio with decarbonisation constraints. It takes the maximum tracking error, 
        a target carbon intensity and an optimiser as inputs and returns a DataFrame with the optimal portfolio 
        weights that minimises the portfolio's carbon intensity subject to the constraints

        Parameters:
            total_initial_carbon_intensity (float):, initial portfolio carbon intensity
            intensity_reduction_rate (float): proportion to reduce the initial portfolio carbon intensity by to achieve target carbon intensity
            base_optimizer (object): instance of the BaseConvexOptimizer class to solve the optimisation problem
            stock_prices (DataFrame): DataFrame containing stock prices data
            mu (np.array): Expected returns of each stock
            S (np.array): Covariance matrix of the returns of the stocks
            carbon_intensity (np.array): Carbon intensity of each stock
            df_final (DataFrame): Final DataFrame containing all financial and environmental data
            benchmark_weights (np.array): Weights of the benchmark portfolio
            random_tickers (list, optional): List of random tickers chosen for portfolio
            is_sector_balance_portfolio (bool): Flag to check if it's a decarbonised portfolio with sector balance adjustment

        Returns:
            DC_weights (DataFrame): Weights of the decarbonised portfolio
            DC_performance (dict): Performance of the decarbonised portfolio
            DC_scores (dict): Scores of the decarbonised portfolio
        """
        max_tracking_var = 0.01 ** 2 # Setting up maximum tracking error at 1%
        # Target carbon intensity to be intensity_rate proportion of initial value 
        target_carbon_intensity = total_initial_carbon_intensity * (1-intensity_reduction_rate)

        # Create the EfficientFrontier object
        efficient_dc = base_optimizer.BaseConvexOptimizer(
            n_assets=len(stock_prices.columns),
            tickers=stock_prices.columns,
            weight_bounds=(0, 1),
            # solver='OSQP',
        )
        sector_mapping_dict = None
        sector_lower = None
        sector_upper = None
        
        # Apply relevant constraints (as per the suggested guide) for sector balance adjustment portfolio
        if is_sector_balance_portfolio:
            efficient_dc.add_constraint(lambda w: w @ carbon_intensity == target_carbon_intensity)
            efficient_dc.add_constraint(lambda w: cp.sum(w) == 1)
            # CVXPY solver couldn't find optimal solutions with this constraint, so commenting it out (no other solvers were feasible)
            # efficient_dc.add_constraint(lambda w: self.ex_ante_tracking_error(w, S, benchmark_weights) <= max_tracking_var)
            sector_mapping_dict, sector_lower, sector_upper = self.sector_balance(df_final, random_tickers)
            efficient_dc.add_sector_constraints(sector_mapper = sector_mapping_dict, 
                               sector_lower = sector_lower,  # Current mapping * 0.7 (-30%)
                               sector_upper = sector_upper)  # Current mapping * 1.3 (+30%)
        else:
            # Add the tracking error constraint
            efficient_dc.add_constraint(lambda w: self.ex_ante_tracking_error(w, S, benchmark_weights) <= max_tracking_var)
            # Adding Carbon Constraint
            efficient_dc.add_constraint(lambda w: w @ carbon_intensity == target_carbon_intensity)
            # Adding constraint for weights to sum to 1
            efficient_dc.add_constraint(lambda w: cp.sum(w) == 1)
        
        # Adding the decarbonisation objective function
        efficient_dc.convex_objective(lambda w: w @ carbon_intensity, weights_sum_to_one=True)

        # Compute the optimal weights
        cleaned_weights_dcp = efficient_dc.clean_weights()
        DC_weights = pd.DataFrame(cleaned_weights_dcp, columns=cleaned_weights_dcp.keys(), index=['Weights']).T
        DC_weights['GICS Sector'] = df_final['GICS Sector']
        # Calculate and print portfolio performance and scores
        DC_performance = self.get_portfolio_performance(DC_weights, mu, S)
        DC_scores = self.get_portfolio_scores(DC_weights['Weights'], df_final, is_benchmark=False)
        self.print_portfolio_stats(DC_performance)
        self.print_portfolio_scores(DC_scores)
        
        return DC_weights, DC_performance, DC_scores

    def mean_variance_efficient_portfolio(self, mu, S, df_final, total_initial_carbon_intensity, carbon_intensity, 
                                          intensity_reduction_rate, is_decarbonised=False):
        """
        Creates a mean-variance efficient portfolio with or without decarbonisation constraints. 
        It takes expected returns, a covariance matrix, a DataFrame of processed data and benchmark weights as inputs 
        and returns a DataFrame with the optimal portfolio weights that maximise the Sharpe ratio

        Parameters:
            mu (np.array): Expected returns of each stock.
            S (np.array): Covariance matrix of the returns of the stocks.
            df_final (DataFrame): Final DataFrame after cleaning and pre-processing.
            total_initial_carbon_intensity (float): Initial portfolio carbon intensity.
            carbon_intensity (np.array): Carbon intensity of each stock.
            intensity_reduction_rate (float): proportion to reduce the initial portfolio carbon intensity by to achieve target carbon intensity
            is_decarbonised (bool): Whether to include decarbonisation constraints.

        Returns:
            MVP_weights (DataFrame): Weights of the mean-variance efficient portfolio.
            MVP_perf (dict): Performance of the mean-variance efficient portfolio.
            MVP_scores (dict): Scores of the mean-variance efficient portfolio.
        """
        efficient_frontier = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        if is_decarbonised:
            target_carbon_intensity = total_initial_carbon_intensity * (1-intensity_reduction_rate)
            efficient_frontier.add_constraint(lambda w: w @ carbon_intensity == target_carbon_intensity)
            efficient_frontier.add_constraint(lambda w: cp.sum(w) == 1)
            raw_weights = efficient_frontier.max_sharpe(risk_free_rate=self.risk_free_rate)
        else:
            efficient_frontier.add_constraint(lambda w: cp.sum(w) == 1)
            raw_weights = efficient_frontier.max_sharpe(risk_free_rate=self.risk_free_rate)

        cleaned_weights = efficient_frontier.clean_weights()
        MVP_weights = pd.DataFrame(cleaned_weights, columns=cleaned_weights.keys(), index=['Weights']).T
        MVP_weights['GICS Sector'] = df_final['GICS Sector']
        MVP_perf = self.get_portfolio_performance(MVP_weights, mu, S)
        MVP_scores = self.get_portfolio_scores(MVP_weights['Weights'], df_final, is_benchmark=False)
        self.print_portfolio_stats(MVP_perf)
        self.print_portfolio_scores(MVP_scores)

        return MVP_weights, MVP_perf, MVP_scores

    def create_esg_performance_df(self, benchmark_scores, dcp_1_scores, dcp_2_scores, dcp_3_scores, dcp_4_scores, 
                                  mvp_scores, mvp_dcp_5_scores, portfolios_list):
        """
        Creates a DataFrame with the ESG performance of different portfolios. It takes scores of 
        different portfolios and a list of portfolio names as inputs and returns a DataFrame with the ESG scores.

        Parameters:
            benchmark_scores (dict): Scores of the benchmark portfolio.
            dcp_1_scores (dict): Scores of decarbonised portfolio 1.
            dcp_2_scores (dict): Scores of decarbonised portfolio 2.
            dcp_3_scores (dict): Scores of decarbonised portfolio 3.
            dcp_4_scores (dict): Scores of decarbonised portfolio 4 (with sector balance adjustment).
            mvp_scores (dict): Scores of the mean-variance portfolio.
            mvp_dcp_5_scores (dict): Scores of decarbonised portfolio 5 (MVP with decarbonisation constraints).
            portfolios_list (list): Names of the portfolios.

        Returns:
            Portfolio_ESGPerf (DataFrame): ESG performance of the portfolios.
        """
        Portfolio_ESGPerf = pd.DataFrame(columns=['Mean Annual Emission', 'Avg ESG Score', 'Avg Environment Score',
                                                  'Avg Governance Score', 'Avg Social Score'], index=portfolios_list)
        Portfolio_ESGPerf['Mean Annual Emission'] = [benchmark_scores['Carbon Intensity'], dcp_1_scores['Carbon Intensity'], 
                                                     dcp_2_scores['Carbon Intensity'], dcp_3_scores['Carbon Intensity'], 
                                                     dcp_4_scores['Carbon Intensity'], mvp_scores['Carbon Intensity'], 
                                                     mvp_dcp_5_scores['Carbon Intensity']]
        Portfolio_ESGPerf['Avg ESG Score'] = [benchmark_scores['ESG Score'], dcp_1_scores['ESG Score'], 
                                                     dcp_2_scores['ESG Score'], dcp_3_scores['ESG Score'], 
                                                     dcp_4_scores['ESG Score'], mvp_scores['ESG Score'], 
                                                     mvp_dcp_5_scores['ESG Score']]
        Portfolio_ESGPerf['Avg Environment Score'] = [benchmark_scores['Environmental Score'], dcp_1_scores['Environmental Score'], 
                                                     dcp_2_scores['Environmental Score'], dcp_3_scores['Environmental Score'], 
                                                     dcp_4_scores['Environmental Score'], mvp_scores['Environmental Score'], 
                                                     mvp_dcp_5_scores['Environmental Score']]
        Portfolio_ESGPerf['Avg Governance Score'] = [benchmark_scores['Governance Score'], dcp_1_scores['Governance Score'], 
                                                     dcp_2_scores['Governance Score'], dcp_3_scores['Governance Score'], 
                                                     dcp_4_scores['Governance Score'], mvp_scores['Governance Score'], 
                                                     mvp_dcp_5_scores['Governance Score']]
        Portfolio_ESGPerf['Avg Social Score'] = [benchmark_scores['Social Score'], dcp_1_scores['Social Score'], 
                                                     dcp_2_scores['Social Score'], dcp_3_scores['Social Score'], 
                                                     dcp_4_scores['Social Score'], mvp_scores['Social Score'], 
                                                     mvp_dcp_5_scores['Social Score']]

        return Portfolio_ESGPerf
    
    def sector_analysis(self, benchmark_portfolio, DCP1_weights, DCP2_weights, DCP3_weights, DCP4_weights, MVP_weights, 
                        DCP5_weights):
        """
        Creates DataFrames with the sector composition of different portfolios. It takes portfolio 
        weights as inputs and returns a DataFrame with the sector composition.

        Parameters:
            benchmark_portfolio (DataFrame): Weights of the benchmark portfolio
            DCP1_weights (DataFrame): Weights of decarbonised portfolio 1
            DCP2_weights (DataFrame): Weights of decarbonised portfolio 2
            DCP3_weights (DataFrame): Weights of decarbonised portfolio 3
            DCP4_weights (DataFrame): Weights of decarbonised portfolio 4
            MVP_weights (DataFrame): Weights of the mean-variance portfolio
            DCP5_weights (DataFrame): Weights of decarbonised portfolio 5

        Returns:
            sector_composition (DataFrame): Sector composition of the portfolios
            sector_composition_grouped (DataFrame): Grouped sector composition of the portfolios
        """
        sector_composition = benchmark_portfolio[['GICS Sector','Weights']].copy()
        sector_composition.rename(columns={'Weights':'Benchmark'},inplace=True)
        sector_composition['DC_Portfolio1'] = DCP1_weights['Weights']
        sector_composition['DC_Portfolio2'] = DCP2_weights['Weights']
        sector_composition['DC_Portfolio3'] = DCP3_weights['Weights']
        sector_composition['DC_Portfolio4'] = DCP4_weights['Weights']
        sector_composition['MV_Portfolio']  = MVP_weights['Weights']
        sector_composition['DC_Portfolio5'] = DCP5_weights['Weights']
        sector_composition_grouped = sector_composition.groupby('GICS Sector')[['Benchmark','DC_Portfolio1','DC_Portfolio2','DC_Portfolio3','DC_Portfolio4','MV_Portfolio','DC_Portfolio5']].sum()
        return sector_composition, sector_composition_grouped
    
    def create_and_analyze_portfolios(self, base_optimizer, num_random_tickers_each_sector, portfolios_list):
        """
        Generates and analyzes various types of portfolios, including benchmark and decarbonised portfolios. 
        This function returns the portfolio weights, performance statistics, ESG performance, and sector analysis of each portfolio

        Parameters:
            base_optimizer (object): Instance of the BaseConvexOptimizer class to solve the optimisation problem
            num_random_tickers_each_sector (int): Number of initial random tickers to select from each sector
            portfolios_list (list): Names of the portfolios

        Returns:
            None. This method prints out the portfolio statistics, ESG performance, and sector composition of each portfolio
        """
        financial_environmental_data = self.data_etl()
        financial_stock_data = self.process_stock_price_data()
        final_data = self.group_data_by_sector(financial_environmental_data, financial_stock_data)
        random_tickers = self.get_random_tickers(final_data, num_random_tickers_each_sector)
        random_tickers = random_tickers.dropna()
        df_final, stock_prices = self.get_final_data(random_tickers, financial_environmental_data, financial_stock_data)

        print("----------Descriptive Statistics: (Market Cap based)----------\n")
        descriptive_statistics = self.get_descriptive_statistics(df_final, stock_prices)

        print("\n----------BENCHMARK PORTFOLIO: MARKET CAP-WEIGHTED PORTFOLIO----------\n")
        benchmark_portfolio, benchmark_perf, benchmark_scores = self.market_cap_portfolio(df_final, descriptive_statistics['mu'], descriptive_statistics['S'])
        benchmark_weights = benchmark_portfolio['Weights'].values
        carbon_intensity = df_final['Carbon Intensity'].values
        total_initial_carbon_intensity = (benchmark_weights * carbon_intensity).sum()
        print("----------DECARBONISED PORTFOLIO 1 (reduction by 50%)----------\n")
        dcp_1_weights, dcp1_perf, dcp_1_scores = self.decarbonised_portfolio(total_initial_carbon_intensity, 0.5, base_optimizer, 
                                                            stock_prices, descriptive_statistics['mu'], 
                                                            descriptive_statistics['S'], carbon_intensity, df_final, benchmark_weights)
        print("----------DECARBONISED PORTFOLIO 2 (reductiion by 25%)----------\n")
        dcp_2_weights, dcp2_perf, dcp_2_scores = self.decarbonised_portfolio(total_initial_carbon_intensity, 0.25, base_optimizer, 
                                                            stock_prices, descriptive_statistics['mu'], 
                                                            descriptive_statistics['S'], carbon_intensity, df_final, benchmark_weights)
        print("----------DECARBONISED PORTFOLIO 3 (reduction by 10%)----------\n")
        dcp_3_weights, dcp3_perf, dcp_3_scores = self.decarbonised_portfolio(total_initial_carbon_intensity, 0.1, base_optimizer, 
                                                            stock_prices, descriptive_statistics['mu'], 
                                                            descriptive_statistics['S'], carbon_intensity, df_final, benchmark_weights)
        print("----------DECARBONISED PORTFOLIO 4 (reduction by 50%, sector balance)----------\n")
        dcp_4_weights, dcp4_perf, dcp_4_scores = self.decarbonised_portfolio(total_initial_carbon_intensity, 0.5, base_optimizer,
                                                            stock_prices, descriptive_statistics['mu'], 
                                                            descriptive_statistics['S'], carbon_intensity, 
                                                            df_final, benchmark_weights, random_tickers, is_sector_balance_portfolio=True)
        print("----------MEAN-VARIANCE EFFICIENT PORTFOLIO----------\n")
        mvp_weights, mvp_perf, mvp_scores = self.mean_variance_efficient_portfolio(descriptive_statistics['mu'], descriptive_statistics['S'], 
                                                                df_final, None, None, None, is_decarbonised=False)
        print("----------MEAN-VARIANCE EFFICIENT PORTFOLIO: DECARBONISED PORTFOLIO 5 (reduction by 50%)----------\n")
        mvp_dcp_5_weights, mvp_dcp_5_perf, mvp_dcp_5_scores = self.mean_variance_efficient_portfolio(descriptive_statistics['mu'], descriptive_statistics['S'],
                                                                                df_final, total_initial_carbon_intensity, 
                                                                                carbon_intensity, intensity_reduction_rate=0.5, is_decarbonised=True)
        print("----------ESG PERFORMANCE STATS----------\n")
        print(self.create_esg_performance_df(benchmark_scores, dcp_1_scores, dcp_2_scores, dcp_3_scores, 
                                           dcp_4_scores, mvp_scores, mvp_dcp_5_scores, portfolios_list))
        print("\n----------SECTOR/INDUSTRY COMPOSITION FOR ALL CONSTRUCTED PORTFOLIOS----------\n")
        sector_composition_companies, sector_composition_grouped = self.sector_analysis(benchmark_portfolio, 
                                                                                                dcp_1_weights, dcp_2_weights, 
                                                                                                dcp_3_weights, dcp_4_weights, 
                                                                                                mvp_weights, mvp_dcp_5_weights)

        print("\n Sector Analysis by company")
        print(sector_composition_companies)
        print("\n Sector Analysis by group")
        print(sector_composition_grouped)

        portfolio_stats_df = pd.DataFrame(index=portfolios_list)
        portfolio_stats_df['Companies Invested'] = [benchmark_perf['Num_Companies'], dcp1_perf['Num_Companies'], dcp2_perf['Num_Companies'], dcp3_perf['Num_Companies'], dcp4_perf['Num_Companies'], mvp_perf['Num_Companies'], mvp_dcp_5_perf['Num_Companies']]
        portfolio_stats_df['Annual ExpReturn'] = [benchmark_perf['Annualised_ExpReturn'], dcp1_perf['Annualised_ExpReturn'], dcp2_perf['Annualised_ExpReturn'], dcp3_perf['Annualised_ExpReturn'], dcp4_perf['Annualised_ExpReturn'], mvp_perf['Annualised_ExpReturn'], mvp_dcp_5_perf['Annualised_ExpReturn']]
        portfolio_stats_df['Monthly ExpReturn'] = portfolio_stats_df['Annual ExpReturn']/12
        portfolio_stats_df['Annual Volatility'] = [benchmark_perf['Annualised_Volatility'], dcp1_perf['Annualised_Volatility'], dcp2_perf['Annualised_Volatility'], dcp3_perf['Annualised_Volatility'], dcp4_perf['Annualised_Volatility'], mvp_perf['Annualised_Volatility'], mvp_dcp_5_perf['Annualised_Volatility']]
        portfolio_stats_df['Monthly Volatility'] = portfolio_stats_df['Annual Volatility']/np.sqrt(12)
        portfolio_stats_df['Sharpe Ratio'] = [benchmark_perf['Sharpe_Ratio'], dcp1_perf['Sharpe_Ratio'], dcp2_perf['Sharpe_Ratio'], dcp3_perf['Sharpe_Ratio'], dcp4_perf['Sharpe_Ratio'], mvp_perf['Sharpe_Ratio'], mvp_dcp_5_perf['Sharpe_Ratio']]
        print("\n----------PORTFOLIO ANALYSIS----------\n")
        print(portfolio_stats_df)
    
    def run_portfolio_decarbonisation(self, base_optimizer):
        """
        Runs the portfolio decarbonisation process. This function creates and analyses various types of portfolios 
        and displays the ESG performance and sector analysis of each portfolio

        Parameters:
            base_optimizer (object): Instance of the BaseConvexOptimizer class to solve the optimisation problem.

        Returns:
            None. This method prints out the portfolio statistics, ESG performance, and sector composition of each portfolio.
        """
        portfolios_list = ['Benchmark Market Cap-Based Portfolio', 'Decarbonised Portfolio1 (reduction by 50%)',
                            'Decarbonised Portfolio2 (reduction by 25%)', 'Decarbonised Portfolio3 (reduction by 10%)',
                            'Decarbonised Portfolio4 (50%, Sector-Balanced)', 'Mean-Variance Portfolio',
                            'Decarbonised Mean-Variance Portfolio (reduction by 50%)']
        self.create_and_analyze_portfolios(base_optimizer, 100, portfolios_list)


if __name__ == "__main__":
    """
    This is the main function which instantiates the PortfolioDecarbonisation class 
    with a specified data source and runs the portfolio decarbonisation process
    """
    risk_free_rate = 0.001
    PD = PortfolioDecarbonisation('./data/R3000_data.xlsx', risk_free_rate=risk_free_rate)
    PD.run_portfolio_decarbonisation(base_optimizer=base_optimizer)   