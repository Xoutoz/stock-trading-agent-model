import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple

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
            self.__total_reward = 0.0               # Track total reward
            portfolio_value = [initial_investment]  # Track portfolio value
            current_balance = initial_investment
            trades = []                             # Track executed trades
            reward = 0.0                            # Initialize reward
            shares = 0                              # Initialize shares
            previous_day_quotation = 0
            update_reward = False
            rewards = []

            # Iterate time-series
            for market_day in data.itertuples():
                day = str(market_day.Index)[:10]

                # Steps:
                # 1. Define the current state
                # 2. Choose recommended action if exists
                # 2. Choose an action based on the current state
                # 3. Update the Q-table
                # 4. Calculate the reward
                # 5. Update the portfolio value

                if market_day.Index == data.index[0] or current_balance < market_day.Close:
                    state = State.NOT_IN_MARKET
                else:
                    state = State.IN_MARKET  # Current state

                    if day in self.__entry_points and np.random.uniform() < self.__lambda:
                        if self.__entry_points[day] == Action.BUY:
                            action = Action.BUY
                        else:
                            action = Action.SELL
                    else:
                        action =  self.__choose_action(state)

                if action == Action.BUY and state == State.NOT_IN_MARKET and current_balance > market_day.Close:
                    state = State.IN_MARKET
                    shares += current_balance // market_day.Close  # Buy as many shares as possible
                    shares_liquidation = -1 * (shares * market_day.Close)
                    current_balance += shares_liquidation
                    portfolio_value.append(portfolio_value[-1])
                    trades.append((action.name, day, round(market_day.Close, 2), shares, 0))
                    update_reward = True

                elif action == Action.SELL and state == State.IN_MARKET and shares > 0:
                    state = State.NOT_IN_MARKET  # Exit the market
                    shares_liquidation = shares * market_day.Close
                    profit = shares_liquidation - current_balance
                    current_balance += shares_liquidation
                    portfolio_value.append(portfolio_value[-1] + (current_balance - portfolio_value[-1]))
                    trades.append((action.name, day, round(market_day.Close, 2), shares, profit, shares_liquidation, current_balance))
                    shares = 0  # Sell all shares
                    update_reward = True
                    number_of_days_holding = 0

                else:
                    update_reward = False

                # elif action == Action.HOLD:
                #     state = State.NOT_IN_MARKET
                #     action = Action.HOLD
                #     portfolio_value.append(portfolio_value[-1])

                #     # Calculate daily return signal
                #     daily_return = (market_day.Close - previous_day_quotation.Close) / previous_day_quotation.Close if previous_day_quotation != 0 else 0

                #     # Extract integer signal
                #     integer_signal = 1 if daily_return > 0 else -1 if daily_return < 0 else 0
                #     # Penalization holding formula: return_signal * (-e^risk_free_rate * t) + 1 -> exponential growth
                #     # reward = daily_return * ((-np.exp(self.__symbol_risk_free_rate * number_of_days_holding)) + 2)
                #     reward = daily_return
                #     update_reward = False
                #     number_of_days_holding += 1


                if market_day.Index != data.index[0] and update_reward:
                    reward = self.__calculate_reward(previous_day_quotation.Close, market_day.Close)
                else:
                    reward = 0
                    # reward = (market_day.Close - previous_day_quotation.Close) / previous_day_quotation.Close
                
                rewards.append(reward)
                # if reward != 0:
                #     reward = np.log(reward) if reward > 0 else -np.log(-reward)
                # else:
                #     reward = 0
                # self.__total_reward += reward

                # if verbose:
                #     print(reward)

                # Update the Q-table
                next_state = State.NOT_IN_MARKET if state == State.NOT_IN_MARKET else State.IN_MARKET  # Next state
                self.__update_q_table(state, action, reward, next_state)

                previous_day_quotation = market_day

            # break
            if self.__epsilon > self.__epsilon_min:
                self.__epsilon -= self.__epsilon_decay

            if self.__lambda > self.__lambda_min:
                self.__lambda -= self.__lambda_decay

            if verbose and episode == 0 or episode % 500 == 499:
                print(f"Episode {episode + 1} of {num_episodes} finished.")
        print(self.__total_reward)
        return self.__total_reward, portfolio_value, trades, rewards

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

    agent = QLearningAgent(env=env, entry_points=strategy.entry_points, symbol_risk_free_rate=strategy.risk)
    # agent = QLearningAgent(env=env, entry_points={}, symbol_risk_free_rate=strategy.risk)
    total_reward, portfolio, trades, rewards = agent.learn(initial_investment=INITIAL_INVESTMENT, num_episodes=1_000, verbose=True)

    print(f"Trades: {trades}")
    print()
    print(f"Portfolio final value: {portfolio[-1]}")
    print()
    print(f"Q-Table: {agent.q_table}")


    plt.figure(figsize=(10, 6))
    plt.plot(data.index, rewards, label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Rewards')
    plt.legend()
    plt.grid()
    plt.savefig('plots/portfolio_performance.png')