from src.utils.strategy import Strategy
from src.utils.action import Action


class MACDStrategy(Strategy):
    def __init__(self, env):
        super().__init__(env)

    def apply_strategy(self, initial_investment: float) -> None:
        # Initialize the portfolio with cash and no shares
        shares = [0]
        portfolio = [initial_investment]
        current_balance = initial_investment

        # Calculate the 12-period EMA
        self._data['EMA12'] = self._data['Close'].ewm(span=12, adjust=False).mean()

        # Calculate the 26-period EMA
        self._data['EMA26'] = self._data['Close'].ewm(span=26, adjust=False).mean()

        # Calculate MACD (the difference between 12-period EMA and 26-period EMA)
        self._data['MACD'] = self._data['EMA12'] - self._data['EMA26']

        # Calculate the 9-period EMA of MACD (Signal Line)
        self._data['Signal_Line'] = self._data['MACD'].ewm(span=9, adjust=False).mean()

        # Calculate the MACD Histogram
        self._data['Histogram'] = self._data['MACD'] - self._data['Signal_Line']

        self._data['Action'] = Action.HOLD

        previous_market_day = None
        for market_day in self._data.itertuples():
            idx = market_day.Index

            if previous_market_day:

                # Golden Cross: MACD crosses above the Signal Line
                if current_balance >= market_day.Close and market_day.MACD > market_day.Signal_Line \
                        and previous_market_day and previous_market_day.MACD < previous_market_day.Signal_Line:
                    self._data.at[idx, 'Action'] = Action.BUY
                    total_shares = portfolio[-1] // market_day.Close
                    shares.append(total_shares)
                    portfolio.append(portfolio[-1])
                    current_balance -= total_shares * market_day.Close

                # Death Cross: MACD crosses below the Signal Line
                elif shares[-1] > 0 and market_day.MACD < market_day.Signal_Line and \
                        previous_market_day and previous_market_day.MACD > previous_market_day.Signal_Line:
                    self._data.at[idx, 'Action'] = Action.SELL
                    current_balance += shares[-1] * market_day.Close
                    portfolio.append(portfolio[-1] + (current_balance - portfolio[-1]))
                    shares.append(0)

                # Othwerwise, hold
                else:
                    self._data.at[idx, 'Action'] = Action.HOLD
                    portfolio.append(portfolio[-1])
                    shares.append(shares[-1])

            previous_market_day = market_day
        
        self._data['Portfolio'] = portfolio
        self._data['Shares'] = shares

    @property
    def entry_points(self):
        return {str(record.Index)[:10]: record.Action for record in self._data[self._data['Action'] != Action.HOLD].itertuples()}


if __name__ == '__main__':
    from src.env.env import CustomStockEnv


    strategy = MACDStrategy(CustomStockEnv.build_from_symbol(start_date="2019-01-01", end_date="2024-10-31"))
    strategy.apply_strategy(10_000)

    print(strategy.entry_points)
