import gymnasium as gym 
import numpy as np
import typing as tt
import torch
from torch import nn, optim
from dataclasses import dataclass

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions, **kwargs):
        super().__init(**kwargs)

        self.pipe = nn.Sequential(nn.Linear(obs_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, n_actions))

    def forward(self, x:torch.Tensor):
        return self.pipe(x)
    
    

@dataclass
class EpisodeStep:
    observation: np.array
    action: int

@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]


def iterate_batches(env: gym.Env, net: Net, batch_size:int) -> tt.Generator[tt.List[Episode], None, None]:
    batch = []
    episode_reward = 0.
    episodes_list = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)

    i=0
    while True:
        obs_v = torch.Tensor(obs, dtype=torch.float32)
        logits = net(obs_v.unsqueeze(0))
        act_probs_v = sm(logits)
        act_probs = act_probs_v.data.numpy()[0, :]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_over, is_trunc, _ = env.step(action)

        episode_reward += reward

        step_log = EpisodeStep(obs=obs_v, action=action)

        episodes_list.append(step_log)
        

        if is_done or is_trunc:
            new_episode = Episode(reward=episode_reward, steps=episodes_list)
            batch.append(new_episode)
            # reset all:
            episode_reward = 0.0
            episodes_list = []

            next_obs, _ = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []


