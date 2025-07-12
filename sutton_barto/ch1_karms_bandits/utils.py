# implement k-armed bandit problem utilities

import numpy as np


class KArmedBandit:
    def __init__(self, k: int, q_mean=0.0, reward_std: float = 1.0):
        """
        Initialize a k-armed bandit problem.

        Args:
            k (int): Number of arms.
            seed (int): Random seed for reproducibility.
        """
        self.k = k
        self.hidden_q_values = np.random.randn(k) + q_mean  # True action values
        self.reward_std = reward_std

    def get_reward(self, action: int) -> float:
        """
        Get a reward for taking an action.

        Args:
            action (int): The action taken (arm pulled).

        Returns:
            float: The reward received.
        """
        if action < 0 or action >= self.k:
            raise ValueError(f"Action must be in range [0, {self.k - 1}]")

        # Sample reward from a normal distribution centered at the true value
        return np.random.normal(self.hidden_q_values[action], self.reward_std)


class BanditSolverAgent:
    def __init__(
        self,
        bandit: KArmedBandit,
        greedy_epsilon: float = 0.1,
        n_steps: int = 1000,
        initial_q: float = 0.0,
    ):
        """
        Initialize the agent for the k-armed bandit problem.

        Args:
            bandit (KArmedBandit): The bandit environment.
        """
        # np.random.seed(seed)
        self.bandit = bandit
        self.greedy_epsilon = greedy_epsilon
        self.initial_q = initial_q
        self.q_values = np.ones(bandit.k) * initial_q  # Estimated action values
        # self.action_counts = np.zeros(bandit.k)  # Count of actions taken
        self.chosen_actions = np.full(n_steps, -1, dtype=int)  # Store chosen actions
        self.past_rewards = np.full(n_steps, np.nan)  # Store past rewards

        self.current_step = 0

    def select_action(self) -> int:
        """
        Select an action using epsilon-greedy strategy.

        Args:
            epsilon (float): Probability of selecting a random action.

        Returns:
            int: The selected action.
        """
        random_value = np.random.rand()
        if random_value < self.greedy_epsilon:
            # print(f"Exploring, random value {random_value} < epsilon {self.greedy_epsilon}")
            return np.random.randint(self.bandit.k)  # Explore
        else:
            return np.argmax(self.q_values)  # Exploit

    def update_q_values(self):
        """
        Update the estimated action values based on the received reward.

        Args:
            action (int): The action taken.
            reward (float): The reward received.
        """
        for i in range(self.bandit.k):
            rewards_for_action = self.past_rewards[self.chosen_actions == i]
            if len(rewards_for_action) > 0:
                self.q_values[i] = np.mean(rewards_for_action)

    def reset(self):
        """
        Reset the agent's state.
        """
        self.q_values.fill(self.initial_q)
        self.past_rewards.fill(np.nan)
        self.chosen_actions.fill(-1)
        self.current_step = 0

    def make_a_move(self):
        """
        Take an action and update the agent's state.

        Args:
            action (int): The action taken.
        """
        action = self.select_action()
        reward = self.bandit.get_reward(action)
        self.chosen_actions[self.current_step] = action
        self.past_rewards[self.current_step] = reward
        self.current_step += 1
        self.update_q_values()
