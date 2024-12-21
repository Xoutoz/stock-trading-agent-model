import numpy as np
import pandas as pd


def calculate_entry_points(data: pd.DataFrame, strategy: str, ta_hyperparameters: tuple, loopback_period: int = 14):
    """
    Calculate entry points for stock trading based on the given strategy.

    Parameters:
    data (pd.DataFrame): The stock price data.
    strategy (str): The trading strategy to use ('RSI' or 'EMA').
    ta_hyperparameters (tuple): Hyperparameters for the technical analysis indicators.
                                For RSI, provide the oversold and overbought thresholds, respectively.
                                For EMA, provide the short and long rolling windows, respectively.
    loopback_period (int, optional): The period for calculating the RSI. Default is 14.

    Returns:
    dict: A dictionary with dates as keys and 'BUY' or 'SELL' signals as values.
    """
    if strategy == 'RSI':
        rsi_calculated_signals = calculate_rsi(
            data,
            loopback_period,
            oversold_threshold=ta_hyperparameters[0],
            overbought_threshold=ta_hyperparameters[1]
        )
        rsi_signals = {}
        for date, value in rsi_calculated_signals.items():
            if value == 1.0:
                rsi_signals[date.strftime('%Y-%m-%d')] = 'BUY'
            elif value == -1.0:
                rsi_signals[date.strftime('%Y-%m-%d')] = 'SELL'
        return rsi_signals

    elif strategy == 'EMA':
        crossover = calculate_crossover(
            data,
            short_rolling_window=ta_hyperparameters[0],
            long_rolling_window=ta_hyperparameters[1]
        )

        crossover_signals = {}
        for date, value in crossover.items():
            if value == 1.0:
                crossover_signals[date.strftime('%Y-%m-%d')] = 'BUY'
            elif value == -1.0:
                crossover_signals[date.strftime('%Y-%m-%d')] = 'SELL'

        return crossover_signals


def calculate_crossover(data, short_rolling_window, long_rolling_window):
    data['ema_short'] = data['Close'].ewm(span=short_rolling_window).mean()
    data['ema_long'] = data['Close'].ewm(span=long_rolling_window).mean()

    data['bullish'] = np.where(data['ema_short'] > data['ema_long'], 1.0, 0.0)
    data['crossover'] = data['bullish'].diff()

    return data['crossover'][(data['crossover'] != 0)].dropna()


def calculate_rsi(data, loopback_period, oversold_threshold, overbought_threshold):
    returns = data.pct_change()
    data['returns'] = returns['Close']
    data['rsi'] = get_rsi(data['returns'], loopback_period)

    data['signal'] = 0
    data.loc[data['rsi'] > overbought_threshold, 'signal'] = -1  # Sell signal
    data.loc[data['rsi'] < oversold_threshold, 'signal'] = 1    # Buy signal

    return data['signal'][(data['signal'] != 0)].dropna()


def get_rsi(returns, loopback_period):
    up_returns = pd.Series(np.where(returns > 0, returns, 0))
    down_returns = pd.Series(np.where(returns < 0, abs(returns), 0))

    up_ewm = up_returns.ewm(alpha=1/loopback_period,
                            min_periods=loopback_period, adjust=False).mean()
    down_ewm = down_returns.ewm(
        alpha=1/loopback_period, min_periods=loopback_period, adjust=False).mean()

    rs = up_ewm / down_ewm
    rsi = 100 - (100 / (1 + rs))

    rsi_df = pd.DataFrame(rsi).rename(columns={0: 'rsi'}) \
        .set_index(returns.index)
    return rsi_df[14:]


if __name__ == '__main__':
    import json
    from env import CustomStockEnv

    OVERBOUGHT_THRESHOLD = 80
    OVERSOLD_THRESHOLD = 20
    SHORT_ROLLING_WINDOW = 50
    LONG_ROLLING_WINDOW = 200

    env = CustomStockEnv.build_from_symbol('TSLA', '2020-01-01', '2021-01-01')
    data = env.df.copy()

    print(json.dumps(calculate_entry_points(data, 'RSI', (OVERSOLD_THRESHOLD, OVERBOUGHT_THRESHOLD)), indent=4))
    print('----------------------------------------')
    print(calculate_entry_points(data, 'EMA', (SHORT_ROLLING_WINDOW, LONG_ROLLING_WINDOW)))
