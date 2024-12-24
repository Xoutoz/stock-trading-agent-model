import numpy as np
import pandas as pd

from src.utils.strategy import Strategy


class Backtest:

    @staticmethod
    def evaluate(strategy: Strategy, initial_investment: float) -> pd.DataFrame:
        returns = strategy.returns

        number_of_trading_days = len(strategy.data)

        risk_free_rate = strategy.risk
        cumulative_returns = (1 + returns).cumprod()[-1]

        annual_returns = (cumulative_returns) ** (252 / number_of_trading_days) - 1
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (annual_returns - risk_free_rate) / volatility
        cagr = (cumulative_returns) ** (252 / number_of_trading_days) - 1
        variance = returns.var() * 252
        data = {
            'Metric': ['Annual Returns', 'Cumulative Returns', 'Volatility (Risk)', 'Sharpe Ratio', 'CAGR', 'Variance', 'Initial Investment', 'Final Investment Product'],
            'Value': [f"{annual_returns:.2%}", f"{cumulative_returns:.2%}", f"{volatility:.2%}", f"{sharpe_ratio:.2f}", f"{cagr:.2%}", f"{variance:.2%}", f"{initial_investment:.2f}", f"{(initial_investment * cumulative_returns):.2f}"]
        }

        backtest_df = pd.DataFrame(data)
        return backtest_df

    @staticmethod
    def model_evaluation(portfolio: list) -> pd.DataFrame:

        number_of_trading_days = len(portfolio) - 1

        returns = np.diff(portfolio) / portfolio[:-1]
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        cagr = (portfolio[-1] / portfolio[0]) ** (252 / number_of_trading_days) - 1

        cumulative_returns = (portfolio[-1] - portfolio[0]) / portfolio[0]
        annual_returns = (cumulative_returns) ** (252 / number_of_trading_days) - 1
        variance = np.var(returns)

        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        data = {
            'Metric': ['Annual Returns', 'Cumulative Returns', 'Volatility (Risk)', 'Sharpe Ratio', 'CAGR', 'Variance', 'Initial Investment', 'Final Investment Product'],
            'Value': [f"{annual_returns:.2%}", f"{cumulative_returns:.2%}", f"{volatility:.2%}", f"{sharpe_ratio:.2f}", f"{cagr:.2%}", f"{variance:.2%}", f"{portfolio[0]:.2f}", f"{(portfolio[-1]):.2f}"]
        }

        backtest_df = pd.DataFrame(data)
        return backtest_df

