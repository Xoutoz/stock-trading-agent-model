import numpy as np
import pandas as pd
import pickle

from action import Action
from state import State

class QLearningAgent:
    def __init__(self, alpha: float = 0.2, gamma: float = 0.9, epsilon: float = 0.1, risk_free_rate: float = 0.0):
        self.__actions = [Action.HOLD, Action.BUY, Action.SELL]
        self.__states = [State.IN_MARKET, State.NOT_IN_MARKET]
        self.__alpha = alpha  # Learning rate
        self.__gamma = gamma  # Discount factor
        self.__epsilon = epsilon  # Exploration rate
        self.__risk_free_rate = risk_free_rate # Risk-free rate
        self.__q_table = np.zeros((len(self.__states), len(self.__actions)))
        self.__q_table_filename = f"models/q_learning/q_table_alpha_{self.__alpha}_gamma_{self.__gamma}_epsilon_{self.__epsilon}_risk_{self.__risk_free_rate}.pickle"

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

    def __calculate_reward(self, current_stock_price: float, current_portfolio_value: float, previous_portfolio_value: float):
        expected_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        if expected_return != 0.0:
            volatility = np.sqrt((current_stock_price / expected_return) ** 2)
            return (expected_return - self.__risk_free_rate) / volatility
        else:
            return 0.0

    # Define the trading strategy with Q-learning and human feedback
    def train(self, data: pd.Series, entry_points: dict, initial_investment: float, num_episodes: int = 1000):

        for episode in range(num_episodes):
            total_reward = 0.0                      # Track total reward
            portfolio_value = [initial_investment]  # Track portfolio value
            trades = []                             # Track executed trades
            reward = 0.0                            # Initialize reward
            shares = 0                              # Initialize shares

            for market_day in data.itertuples():
                day = str(market_day.Index)[:10]

                if market_day.Index == data.index[0]:
                    state = State.NOT_IN_MARKET
                    action = Action.BUY
                else:
                    state = State.NOT_IN_MARKET if portfolio_value[-1] == 0 else State.IN_MARKET  # Current state

                    if day in entry_points:
                        if entry_points[day] == 'BUY':
                            action = Action.BUY
                            state = State.NOT_IN_MARKET
                        else:
                            action = Action.SELL
                            state = State.IN_MARKET
                    else:
                        action =  self.__choose_action(state)

                if action == Action.BUY and state == State.NOT_IN_MARKET and shares == 0:
                    state = State.IN_MARKET
                    shares += portfolio_value[-1] // market_day.Close  # Buy as many shares as possible
                    # print(f"Shares: {shares}")
                    portfolio_value.append(portfolio_value[-1] - (shares * market_day.Close))
                    trades.append((action, day, market_day.Close, shares))

                elif action == Action.SELL and state == State.IN_MARKET and shares > 0:
                    state = State.NOT_IN_MARKET  # Exit the market
                    portfolio_value.append(portfolio_value[-1] + (shares * market_day.Close))
                    shares = 0  # Sell all shares
                    trades.append((action, day, market_day.Close, shares))

                else:
                    state = State.NOT_IN_MARKET
                    action = Action.HOLD
                    portfolio_value.append(portfolio_value[-1])

                # trades.append((action, day, market_day.Close, shares))


                if market_day.Index != data.index[0]:
                    # reward = self.__calculate_reward(market_day.Close, portfolio_value[-1], portfolio_value[-2])
                    reward = (portfolio_value[-1] - portfolio_value[-2]) / portfolio_value[-2] if portfolio_value[-2] != 0 else 0.0
                total_reward += reward

                # Update the Q-table
                next_state = State.NOT_IN_MARKET if state == State.NOT_IN_MARKET else State.IN_MARKET  # Next state
                self.__update_q_table(state, action, reward, next_state)

        return total_reward, portfolio_value, trades


        # # Iterate over each trading day
        # for index in range(len(data) - 1):
        #     state = State.NOT_IN_MARKET if portfolio_value[-1] == 0 else State.IN_MARKET  # Current state

        #     # Check if an entry point exists for the current day
        #     if data.index[index] in entry_points:
        #         # Get the recommended action for the current day from human feedback
        #         action = entry_points[data.index[index]]

        #         # Execute the recommended action
        #         if action == Action.BUY and state == State.NOT_IN_MARKET:
        #             state = State.IN_MARKET  # Enter the market
        #             shares = portfolio_value[-1] // data['Close'][index]  # Buy as many shares as possible
        #             trades.append((Action.BUY, data.index[index], data['Close'][index], shares))

        #         elif action == Action.SELL and state == State.IN_MARKET:
        #             state = State.NOT_IN_MARKET  # Exit the market
        #             shares = portfolio_value[-1] // data['Close'][index]  # Sell all shares
        #             trades.append((Action.SELL, data.index[index], data['Close'][index], shares))

        #     # Calculate the reward for the current day
        #     if index > 0:
        #         reward = self.__calculate_reward(data['Close'][index], portfolio_value[-1], portfolio_value[-2])
        #     total_reward += reward

        #     # Choose the action using epsilon-greedy policy
        #     action = self.__choose_action(state)

        #     # Execute the action
        #     if action == Action.BUY and state == State.NOT_IN_MARKET:
        #         state = State.IN_MARKET  # Enter the market
        #         shares = portfolio_value[-1] // data['Close'][index]  # Buy as many shares as possible
        #         trades.append((Action.BUY, data.index[index], data['Close'][index], shares))

        #     elif action == Action.SELL and state == State.IN_MARKET:
        #         state = State.NOT_IN_MARKET  # Exit the market
        #         shares = portfolio_value[-1] // data['Close'][index]  # Sell all shares
        #         trades.append((Action.SELL, data.index[index], data['Close'][index], shares))

        #     # Update the Q-table
        #     next_state = State.NOT_IN_MARKET if state == State.NOT_IN_MARKET else State.IN_MARKET  # Next state
        #     self.__update_q_table(state, action, reward, next_state)

        #     # Update the portfolio value
        #     portfolio_value.append(shares * data['Close'][index])

        # return total_reward, portfolio_value, trades

    def save_model(self):
        with open(self.__q_table_filename, "wb") as file:
            pickle.dump(self.__q_table, file)

    def load_model(self):
        try:
            with open(self.__q_table_filename, "rb") as file:
                self.__q_table = pickle.load(file)
        except FileNotFoundError:
            print("Model not found. Please train the model first.")

    def get_q_table(self):
        return self.__q_table

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    from env import CustomStockEnv
    from utils import calculate_entry_points


    agent = QLearningAgent()

    agent.save_model()
    agent.load_model()

    env = CustomStockEnv.build_from_symbol(start_date="2019-01-01", end_date="2024-10-31")
    agent = QLearningAgent()

    data = env.get_data()
    entry_points = calculate_entry_points(data, "EMA", (50, 200))

    print(entry_points)
    print()

    total_reward, portfolio, trades = agent.train(data, entry_points, initial_investment=30_000, num_episodes=1_000)

    print(f"Trades: {trades}")
    print()
    print(f"Portfolio final value: {portfolio[-1]}")


    plt.figure(figsize=(10, 6))
    plt.plot(data.index, portfolio[1:], label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Portfolio Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('portfolio_performance.png')