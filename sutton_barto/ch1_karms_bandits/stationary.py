from utils import KArmedBandit, BanditSolverAgent
from dataclasses import dataclass
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import numpy as np


@dataclass
class BanditParams:
    k: int = 10
    q_mean: float = 0.0
    reward_std: float = 1.0


@dataclass
class AgentParams:
    greedy_epsilon: float = 0.1
    initial_q: float = 0.0


n_steps = 1000
n_runs = 2000

agents_dict = {
    "eps0.1": AgentParams(greedy_epsilon=0.1),
    "eps0.01": AgentParams(greedy_epsilon=0.01),
    "greedy": AgentParams(greedy_epsilon=0.0),
}

results_rewards_dict = {
    k: np.nan * np.ones((n_runs, n_steps)) for k in agents_dict.keys()
}
results_optimal_actions_dict = {
    k: np.nan * np.ones((n_runs, n_steps)) for k in agents_dict.keys()
}

for n_run in trange(n_runs):
    seed = np.random.randint(0, 1000000)
    bandit = KArmedBandit(
        k=BanditParams.k, q_mean=BanditParams.q_mean, reward_std=BanditParams.reward_std
    )
    true_optimal_action = np.argmax(bandit.hidden_q_values)

    for agent_name, agent_params in agents_dict.items():
        results_rewards = np.zeros((n_runs, n_steps))
        np.random.seed(seed)
        agent = BanditSolverAgent(
            bandit,
            greedy_epsilon=agent_params.greedy_epsilon,
            initial_q=agent_params.initial_q,
            n_steps=n_steps,
        )
        for _ in range(n_steps):
            agent.make_a_move()

        results_rewards[n_run] = agent.past_rewards

        average_rewards = np.mean(results_rewards, axis=0)
        results_rewards_dict[agent_name][n_run, :] = average_rewards
        results_optimal_actions_dict[agent_name][n_run, :] = (
            agent.chosen_actions == true_optimal_action
        )


fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
for agent_name, average_optimal_actions in results_optimal_actions_dict.items():
    ax[0].plot(np.mean(average_optimal_actions, 0), label=agent_name)

for agent_name, average_rewards in results_rewards_dict.items():
    ax[1].plot(np.mean(average_rewards, 0), label=agent_name)

ax[0].set_xlabel("Steps")
ax[1].set_xlabel("Steps")
ax[0].set_ylabel("Optimal Action Selection Rate")
ax[1].set_ylabel("Average Reward")
ax[0].legend(frameon=False)
plt.tight_layout()
plt.show()
