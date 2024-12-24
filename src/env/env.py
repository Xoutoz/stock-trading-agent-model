import pandas as pd
import gym_anytrading
from gym_anytrading.envs import StocksEnv
import yfinance as yf


class CustomStockEnv(StocksEnv):
    def __init__(self, df: pd.DataFrame, window_size: int = 20):
        """
        Initializes the environment with the given DataFrame and window size.
        Args:
            df (pd.DataFrame): The DataFrame containing the stock data.
            window_size (int, optional): The size of the window for the trading agent. Defaults to 20.
        Attributes:
            __df (pd.DataFrame): The DataFrame containing the stock data.
            __window_size (int): The size of the window for the trading agent.
            __start_index (int): The starting index for the trading window.
            __end_index (int): The ending index for the trading window.
            __frame_bound (tuple): A tuple containing the start and end indices for the trading window.
        """

        self.__df = df
        self.__window_size = window_size
        self.__start_index = window_size
        self.__end_index = len(df)
        self.__frame_bound = (self.__start_index, self.__end_index)
        super().__init__(self.__df, self.__window_size, self.__frame_bound)

    def get_max_possible_profit(self):
        """
        Calculate the maximum possible profit.
        This method returns the maximum possible profit that can be achieved
        by the trading agent. It delegates the calculation to the `max_possible_profit`
        method of the unwrapped environment.
        Returns:
            float: The maximum possible profit.
        """

        return self.unwrapped.max_possible_profit()

    @property
    def data(self):
        """
        Retrieves a copy of the internal dataframe.
        Returns:
            pandas.DataFrame: A copy of the internal dataframe.
        """

        return self.__df.copy()

    @staticmethod
    def build_from_symbol(start_date: str, end_date: str, symbol: str = "KO", window_size: int = 20):
        """
        Builds a CustomStockEnv instance using historical stock data for a given symbol.
        Args:
            start_date (str): The start date for the historical data in the format 'YYYY-MM-DD'.
            end_date (str): The end date for the historical data in the format 'YYYY-MM-DD'.
            symbol (str, optional): The stock symbol to retrieve data for. Defaults to "KO".
            window_size (int, optional): The size of the window for the environment. Defaults to 20.
        Returns:
            CustomStockEnv: An instance of the CustomStockEnv class initialized with the historical data.
        """

        return CustomStockEnv(yf.Ticker(symbol).history(start=start_date, end=end_date), window_size)


if __name__ == '__main__':
    # Simple test to check if the environment is working
    df = yf.Ticker('KO').history(start='2019-01-01', end='2024-10-31')
    print(len(df))
    print('----------------------------------------')

    env = CustomStockEnv(df)
    print(env.metadata)
    print(env.get_max_possible_profit())

    print('----------------------------------------')

    env = CustomStockEnv.build_from_symbol('2019-01-01', '2024-12-31')
    print(env.metadata)
    print(env.get_max_possible_profit())
