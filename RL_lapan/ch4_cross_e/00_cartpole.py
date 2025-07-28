from pickletools import optimize
import gymnasium as gym 
import numpy as np
import typing as tt
import torch
from torch import nn, optim
from dataclasses import dataclass
from torch.utils.tensorboard.writer import SummaryWriter


import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions, **kwargs):
        super().__init__(**kwargs)

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


def iterate_batches(env: gym.Env, net: Net, batch_size: int) -> tt.Generator[tt.List[Episode], None, None]:
    batch = []
    episode_reward = 0.
    episodes_list = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)

    # i=0
    while True:
        obs_v = torch.tensor(obs, dtype=torch.float32)
        logits = net(obs_v.unsqueeze(0))
        act_probs_v = sm(logits)
        act_probs = act_probs_v.data.numpy()[0, :]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)

        episode_reward += reward

        step_log = EpisodeStep(observation=obs_v, action=action)

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

        obs = next_obs


def filter_batch(batch: tt.List[Episode], percentile: float) -> tt.Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    rewards = list(map(lambda x: x.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = np.mean(rewards)

    train_obs: tt.List[np.array] = []
    train_act: tt.List[int] = []

    for episode in batch:
        if episode.reward >= reward_bound:
            train_obs.extend(map(lambda x: x.observation, episode.steps))
            train_act.extend(map(lambda x: x.action, episode.steps))


    train_obs_v = torch.FloatTensor(np.vstack(train_obs))
    train_act_v = torch.LongTensor(train_act)

    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="/Users/vigji/Desktop/new_folder")
    obs_space_shape = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    net = Net(obs_size=obs_space_shape, n_actions=n_actions,
              hidden_size=HIDDEN_SIZE)
    
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    writer = SummaryWriter(comment="-cartpole")

    for iter_n, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, act_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        print(obs_v.shape)
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, act_v)
        loss_v.backward()
        optimizer.step()

        print(f"{iter_n}: loss: {loss_v:.3f}, rw_mean:{reward_m:.1f}, rw_bound:{reward_b:.1f}")
        writer.add_scalar("loss", loss_v.item(), iter_n)
        writer.add_scalar("reward_b", reward_b, iter_n)
        writer.add_scalar("reward_m", reward_m, iter_n)

        if reward_m > 475:
            print("GENIOOO")
            break

    writer.close()

