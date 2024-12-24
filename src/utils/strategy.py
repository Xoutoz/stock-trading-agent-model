from abc import ABC, abstractmethod

from src.env.env import CustomStockEnv


class Strategy(ABC):
    def __init__(self, env: CustomStockEnv):
        """
        Initializes the strategy with the given stock trading environment.
        Args:
            env (CustomStockEnv): The custom stock trading environment instance.
        Attributes:
            _data (pd.DataFrame): The data obtained from the environment.
        """

        self._data = env.data

    @property
    def returns(self):
        """
        Calculate the percentage change of the data.
        Returns:
            pandas.DataFrame or pandas.Series: The percentage change of the data.
        """

        return self._data['Close'].pct_change()

    @property
    def risk(self):
        """
        Calculate the risk of the trading strategy.
        Risk is defined as the standard deviation of the returns.
        Returns:
            float: The standard deviation of the returns.
        """

        return self.returns.std()

    @property
    def data(self):
        """
        Retrieve the stored data.
        Returns:
            Any: The data stored in the instance.
        """

        return self._data

    @abstractmethod
    def entry_points(self):
        """
        Abstract method to get entry points for a trading strategy.
        This method should be implemented by subclasses to define the logic
        for identifying entry points in a trading strategy.
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """

        raise NotImplementedError

    @abstractmethod
    def apply_strategy(self, initial_investment: float):
        """
        Apply the trading strategy to the given initial investment.
        This method should be implemented by subclasses to define the specific
        trading strategy logic.
        Args:
            initial_investment (float): The initial amount of money to invest.
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """

        raise NotImplementedError