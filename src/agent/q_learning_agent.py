import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple

import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.action import Action
from src.utils.state import State
from src.env.env import CustomStockEnv


class QLearningAgent:
    def __init__(self, env: CustomStockEnv, entry_points: Tuple[str, Action], alpha: float = 0.2, gamma: float = 0.9,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995, lambda_min: float = 0.01,
                 lambda_decay: float = 0.995, symbol_risk_free_rate: float = 0.0, model_name: str = 'q_learning'):
        """
        Initialize the QLearningAgent.
        Parameters:
        env (CustomStockEnv): The custom stock trading environment.
        entry_points (Tuple[str, Action]): The entry points for the agent (human feedback).
        alpha (float, optional): Learning rate for Q-learning. Default is 0.2.
        gamma (float, optional): Discount factor for Q-learning. Default is 0.9.
        epsilon_min (float, optional): Minimum exploration rate. Default is 0.01.
        epsilon_decay (float, optional): Decay rate for exploration. Default is 0.995.
        lambda_min (float, optional): Minimum value for human feedback parameter. Default is 0.01.
        lambda_decay (float, optional): Decay rate for human feedback parameter. Default is 0.995.
        daily_risk_free_rate (float, optional): Daily risk-free rate. Default is 0.0.
        """


        self.__env = env
        self.__entry_points = entry_points

        # Agent possible states and actions
        self.__actions: List[Action] = [Action.BUY, Action.SELL]
        self.__states: List[State] = [State.IN_MARKET, State.NOT_IN_MARKET]

        # Q-learning hyper parameters
        self.__alpha: float = alpha  # Learning rate
        self.__gamma: float = gamma  # Discount factor
        self.__epsilon: float = 1.0  # Exploration rate
        self.__epsilon_min: float = epsilon_min  # Minimum exploration rate
        self.__epsilon_decay: float = epsilon_decay  # Exploration decay rate

        # Human feedback hyper parameters
        self.__lambda: float = 1.0
        self.__lambda_min: float = lambda_min
        self.__lambda_decay: float = lambda_decay

        # Financial hyper parameters
        self.__symbol_risk_free_rate: float = symbol_risk_free_rate # Risk-free rate
        self.__daily_risk_free_rate: float = symbol_risk_free_rate / 365 # Risk-free rate

        # Acumulated reward
        self.__total_reward: float = 0.0

        # Q-Table
        self.__q_table: np.ndarray = np.zeros((len(self.__states), len(self.__actions)))

        # Technical parameters
        self.__q_table_filename: str = f"models/q_learning/{model_name}_q_table.pkl"

    def __update_q_table(self, state: State, action: Action, reward: float, next_state: State):
        """
        Update the Q-table using the Q-learning algorithm.
        Args:
            state (State): The current state of the environment.
            action (Action): The action taken in the current state.
            reward (float): The reward received after taking the action.
            next_state (State): The state of the environment after taking the action.
        Returns:
            None
        """

        max_q_value = np.max(self.__q_table[next_state.value])
        self.__q_table[state.value, action.value] += self.__alpha * (reward + self.__gamma * max_q_value - self.__q_table[state.value, action.value])

    def __choose_action(self, state: State):
        """
        Choose an action based on the current state using an epsilon-greedy policy.
        Args:
            state (State): The current state of the environment.
        Returns:
            Action: The action chosen based on the epsilon-greedy policy. With probability
                    `self.__epsilon`, a random action is chosen. Otherwise, the action with
                    the highest Q-value for the given state is chosen.
        """

        return np.random.choice(self.__actions) if np.random.uniform() < self.__epsilon else Action(np.argmax(self.__q_table[state.value]))

    def __calculate_reward(self, current_portfolio_value: float, previous_portfolio_value: float):
        """
        Calculate the reward based on the current and previous portfolio values.
        The reward is calculated as the Sharpe ratio, which is the expected return
        minus the daily risk-free rate, divided by the daily volatility.
        Args:
            current_portfolio_value (float): The value of the portfolio at the current time step.
            previous_portfolio_value (float): The value of the portfolio at the previous time step.
        Returns:
            float: The calculated reward. If the expected return is zero, the reward is zero.
        """

        expected_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        daily_volatility = np.std([previous_portfolio_value, current_portfolio_value])

        return (expected_return - self.__daily_risk_free_rate) / daily_volatility if expected_return != 0.0 else 0

    def __execute_action(self, action: Action, state: State, portfolio_balance: float, shares: int, market_day: pd.Series):
        """
        Execute the action based on the current state.
        Args:
            action (Action): The action to execute.
            state (State): The current state of the environment.
            portfolio_balance (float): The current balance of the portfolio.
            shares (int): The number of shares held in the portfolio.
            market_day (pd.Series): The current market day data.
        Returns:
            Tuple[float, int]: A tuple containing the updated portfolio balance and number of shares.
        """

        if action == Action.BUY and state == State.NOT_IN_MARKET:
            state = State.IN_MARKET
            if portfolio_balance >= market_day.Close:
                shares += portfolio_balance // market_day.Close
                portfolio_balance -= shares * market_day.Close

        elif action == Action.SELL and state == State.IN_MARKET:
            state = State.NOT_IN_MARKET
            portfolio_balance += shares * market_day.Close
            shares = 0

        return state, portfolio_balance, shares


    # Define the trading strategy with Q-learning and human feedback
    def learn(self, initial_investment: float, num_episodes: int = 10_000, verbose: bool = False):
        """
        Train the Q-learning agent on stock data.
        Parameters:
        initial_investment (float): Initial amount of money to start trading with.
        num_episodes (int, optional): Number of training episodes. Default is 10,000.
        Returns:
        tuple: A tuple containing:
            - total_reward (float): The total reward accumulated over all episodes.
            - portfolio_value (list): List of portfolio values over time.
            - trades (list): List of executed trades with details.
        """

        data = self.__env.data

        for episode in range(num_episodes):
            self.__total_reward = 0
            portfolio_balance = initial_investment
            portfolio_value = [portfolio_balance]
            trades = []
            shares = 0

            reward = 0
            previous_market_day = None

            # Iterate time-series
            for market_day in data.itertuples():
                # Get market day
                day = str(market_day.Index)[:10]

                # Get the current state
                state = State.NOT_IN_MARKET if portfolio_balance < market_day.Close or previous_market_day is None else State.IN_MARKET

                # Get the recommended action
                if day in self.__entry_points and np.random.uniform() < self.__lambda:
                    action = self.__entry_points[day]

                    # Execute the recommended action
                    state, portfolio_balance, shares = self.__execute_action(action, state, portfolio_balance, shares, market_day)
                    trades.append((day, action, round(market_day.Close, 2), shares, round(portfolio_balance, 2)))

                # Calculate the reward
                reward = self.__calculate_reward(portfolio_balance, portfolio_value[-1]) if previous_market_day else 0
                self.__total_reward += reward

                # Choose an action based on the current state
                action = self.__choose_action(state)

                # Execute the action
                state, portfolio_balance, shares = self.__execute_action(action, state, portfolio_balance, shares, market_day)
                trades.append((day, action, round(market_day.Close, 2), shares, round(portfolio_balance, 2)))

                # Update the Q-table
                next_state = State.NOT_IN_MARKET if state == State.NOT_IN_MARKET else State.IN_MARKET  # Next state
                self.__update_q_table(state, action, reward, next_state)

                # Update the portfolio value
                portfolio_value.append(shares * market_day.Close + portfolio_balance)

                previous_market_day = market_day

            # Decay exponentially epsilon value
            if self.__epsilon > self.__epsilon_min:
                self.__epsilon *= self.__epsilon_decay

            # Decay exponentially human feedback parameter
            if self.__lambda > self.__lambda_min:
                self.__lambda -= self.__lambda_decay

            if verbose and (episode == 0 or episode % 500 == 499):
                print(f"Episode {episode + 1} of {num_episodes} finished.")

        return self.__total_reward, portfolio_value, trades

    def save_model(self):
        """
        Saves the Q-table to a file using pickle.
        The Q-table is saved to the file specified by the attribute
        `self.__q_table_filename`. The file is opened in binary write mode
        and the Q-table is serialized using the pickle module.
        Raises:
            Exception: If there is an issue with file writing or serialization.
        """

        with open(self.__q_table_filename, "wb") as file:
            pickle.dump(self.__q_table, file)

    def load_model(self):
        """
        Loads the Q-table model from a file.
        This method attempts to load the Q-table from a file specified by
        the `__q_table_filename` attribute. If the file is not found, it
        prints a message indicating that the model needs to be trained first.
        Raises:
            FileNotFoundError: If the Q-table file does not exist.
        """

        try:
            with open(self.__q_table_filename, "rb") as file:
                self.__q_table = pickle.load(file)
        except FileNotFoundError:
            print("Model not found. Please train the model first.")

    @property
    def q_table(self):
        """
        Retrieve the Q-table used by the agent.
        Returns:
            dict: The Q-table, which is a dictionary mapping states to actions and their respective Q-values.
        """

        return self.__q_table

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    from src.env.env import CustomStockEnv
    from src.utils.macd_strategy import MACDStrategy


    INITIAL_INVESTMENT = 10_000

    env = CustomStockEnv.build_from_symbol(start_date="2019-01-01", end_date="2024-10-31")
    strategy = MACDStrategy(env)
    strategy.apply_strategy(initial_investment=INITIAL_INVESTMENT)

    data = env.data

    # Best parameters -> Learning Rate: 0.77, Gamma: 0.4, Minimum Epsilon: 0.99, Epsilon Decay: 0.12, Minimun Lambda: 0.69, Lambda Decay: 0.889
    agent = QLearningAgent(
        env=env,
        entry_points=strategy.entry_points,
        alpha=0.77,
        gamma=0.4,
        epsilon_min=0.99,
        epsilon_decay=0.12,
        lambda_min=0.69,
        lambda_decay=0.889,
        symbol_risk_free_rate=strategy.risk)
    total_reward, portfolio, trades = agent.learn(initial_investment=INITIAL_INVESTMENT, num_episodes=10_000, verbose=True)

    print(f"Portfolio final value: {portfolio[-1]}")
    print()
    print(f"Q-Table: {agent.q_table}")


    plt.figure(figsize=(10, 6))
    plt.plot(data.index, portfolio[1:], label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Rewards')
    plt.legend()
    plt.grid()
    plt.savefig('plots/portfolio_performance.png')