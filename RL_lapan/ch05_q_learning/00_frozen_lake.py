import typing as tt
import gymnasium as gym
from collections import defaultdict, Counter
from torch.utils.tensorboard.writer import SummaryWriter


ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20

State = int
Action = int
RewardKey = tt.Tuple[State, Action, State]
TransitKey = tt.Tuple[State, Action]


class Agent:
    def __init__(self):
        self.env: gym.Env = gym.make(ENV_NAME)

        self.state, _ = self.env.reset()
        self.rewards: tt.Dict[RewardKey, float] = defaultdict(float)
        self.transitions: tt.Dict[TransitKey, Counter] = defaultdict(Counter)

        self.values: tt.Dict[State, float] = defaultdict(float)

    
    def play_n_ramdom(self, n : int):
        for _ in range(n):
            action = self.env.action_space.sample()

            new_state, reward, term_flag, trunc_flag, _ = self.env.step(action)

            rw_key = (self.state, action, new_state)
            self.rewards[rw_key] = float(reward)

            trans_key = (self.state, action)
            self.transitions[trans_key][new_state] += 1

            if term_flag or trunc_flag:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def calculate_action_value(self, state, action):
        possible_states_counts = self.transitions[(state, action)]
        total = sum(possible_states_counts.values())
        acc_value = 0

        for possible_state, count in possible_states_counts.items():
            reward = self.rewards[(state, action, possible_state)]
            state_value = self.values[possible_state]
            p_weight = count / total
            acc_value += p_weight * (reward + GAMMA * state_value)

        return acc_value
    
    def select_action(self, state: State):
        best_action = None
        best_action_val = None

        for action in range(self.env.action_space.n):
            action_val = self.calculate_action_value(state, action)
            if not best_action_val or action_val > best_action_val:
                best_action = action
                best_action_val = action_val

        return best_action
    
    def play_episode(self, env: gym.Env):
        total_reward = 0.0

        state, _ = env.reset()

        while True:
            print("!")
            action = self.select_action(state)
            state, reward, end_flag, trunc_flag, _ = env.step(action)
            total_reward += reward
            if end_flag or trunc_flag:
                break

        return total_reward
        


if __name__ == "__main__":
    from pprint import pprint
    self = Agent()

    self.play_n_ramdom(100000)
    pprint(self.rewards)
    # pprint(self.transitions)
    state = 14
    action = 3

    
    print(self.calculate_action_value(state, action))
    print(self.select_action(14))
    env = gym.make(ENV_NAME)

    print(self.play_episode(env))
