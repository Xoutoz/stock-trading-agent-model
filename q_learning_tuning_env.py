import numpy as np
import gymnasium as gym
from itertools import product

from q_learning_agent import QLearningAgent
from env import CustomStockEnv
from utils import calculate_entry_points

class QLearningTuningEnv(gym.Env):
    def __init__(self, max_episodes):
        super(QLearningTuningEnv, self).__init__()

        self.__task_env = CustomStockEnv.build_from_symbol(start_date="2019-01-01", end_date="2024-10-31")
        self.__entry_points = calculate_entry_points(self.__task_env.get_data(), "EMA", (50, 200))
        self.__best_reward = -np.inf
        self.__initial_investment = 10_000
        self.__risk_free_rate = 0.0456 / 365

        # Meta-episode tracking
        self.__max_episodes = max_episodes
        self.current_episode = 0  # Track the current episode number

        # Define hyperparameter space
        # alpha, gamma, epsilon_min, epsilon_decay, lambda_min, lambda_decay
        alpha = np.linspace(0.01, 0.99, 10)
        gamma = np.linspace(0.1, 0.99, 10)
        epsilon_min = np.linspace(0.1, 0.99, 10)
        epsilon_decay = np.linspace(0.01, 0.999, 10)
        lambda_min = np.linspace(0.1, 0.99, 10)
        lambda_decay = np.linspace(0.01, 0.999, 10)

        # All combinations of hyperparameters
        self.action_combinations = list(product(
            alpha,
            gamma,
            epsilon_min,
            epsilon_decay,
            lambda_min,
            lambda_decay
        ))
        self.action_space = gym.spaces.Discrete(len(self.action_combinations))

        # Dummy observation space (e.g., cumulative reward of Q-learning agent)
        self.observation_space = gym.spaces.Box(low=0, high=1e6, shape=(1,), dtype=np.float64)

    def reset(self, seed=None):
        # Reset meta-environment state
        super().reset(seed=seed)
        self.current_episode = 0
        self.__best_reward = -np.inf
        return np.array([0.0]), {}

    def step(self, action):
        # Map action to Q-learning hyperparameters
        alpha, gamma, epsilon_min, epsilon_decay, lambda_min, lambda_decay = self.action_combinations[action]

        print(f"Episode: {self.current_episode + 1}/{self.__max_episodes}, Learning Rate: {round(alpha, 2)}, Gamma: {round(gamma, 2)}, Minimum Epsilon: {round(epsilon_min, 2)}, Epsilon Decay: {round(epsilon_decay, 3)}, Minimun Lambda: {round(lambda_min, 2)}, Lambda Decay: {round(lambda_decay, 3)}")

        # Train Q-learning agent
        q_agent = QLearningAgent(
            alpha=alpha,
            gamma=gamma,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            lambda_min=lambda_min,
            lambda_decay=lambda_decay,
            daily_risk_free_rate=self.__risk_free_rate
        )
        total_reward, _, __ = q_agent.train(self.__task_env.get_data(), self.__entry_points, initial_investment=self.__initial_investment, num_episodes=500)

        # Update episode counter
        self.current_episode += 1

        # Update best reward
        self.__best_reward = max(self.__best_reward, total_reward)

        # Reward for PPO agent
        reward = total_reward

        # Observation and done flag
        obs = np.array([total_reward])

        # Check if maximum episodes are reached
        done = self.current_episode >= self.__max_episodes

        return obs, reward, done, False, {}


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch

    # Verify MPS device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")


    # Create the meta-environment
    meta_env = DummyVecEnv([lambda: QLearningTuningEnv(max_episodes=10)])

    # Create and train the PPO agent
    ppo_agent = PPO(
        "MlpPolicy",
        meta_env,
        n_steps=256,      # Reduced rollout length
        device=device,
        verbose=1
    )
    ppo_agent.learn(total_timesteps=100)

    # Get the best hyperparameters from the trained PPO agent
    obs = meta_env.envs[0].observation_space.sample()
    action, _ = ppo_agent.predict(obs, deterministic=True)

    # Extract hyperparameters using the action index
    best_hyperparameters = meta_env.envs[0].action_combinations[action]
    alpha, gamma, epsilon_min, epsilon_decay, lambda_min, lambda_decay = best_hyperparameters

    print("Best Hyperparameters Found:")
    print(f"Learning Rate: {round(alpha, 2)}, Gamma: {round(gamma, 2)}, Minimum Epsilon: {round(epsilon_min, 2)}, Epsilon Decay: {round(epsilon_decay, 3)}, Minimun Lambda: {round(lambda_min, 2)}, Lambda Decay: {round(lambda_decay, 3)}")