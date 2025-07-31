import typing as tt
import gymnasium as gym
from collections import defaultdict, Counter
from torch.utils.tensorboard.writer import SummaryWriter
from pprint import pprint


ENV_NAME = "FrozenLake-v1" # "FrozenLake-v1"
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
        self.transits: tt.Dict[TransitKey, Counter] = defaultdict(Counter)

        self.q_values: tt.Dict[TransitKey, float] = defaultdict(float)

    
    def play_n_random(self, n : int):
        for _ in range(n):
            action = self.env.action_space.sample()

            new_state, reward, term_flag, trunc_flag, _ = self.env.step(action)
            rw_key = (self.state, action, new_state)
            self.rewards[rw_key] = float(reward)
            trans_key = (self.state, action)
            self.transits[trans_key][new_state] += 1
            if term_flag or trunc_flag:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    # def calc_action_value(self, state: State, action: Action) -> float:
    #     target_counts = self.transits[(state, action)]
    #     # total = sum(target_counts.values())
    #     # action_value = 0.0
    #     # for tgt_state, count in target_counts.items():
    #     #     rw_key = (state, action, tgt_state)
    #     reward = self.rewards[rw_key]
    #     val = reward + GAMMA * self.values[(state, action)]
    #         action_value += (count / total) * val
    #     return action_value
    
    def select_action(self, state: State) -> Action:
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.q_values[(state, action)] # self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action
                
    def play_episode(self, env: gym.Env):
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, end_flag, trunc_flag, _ = env.step(action)
            total_reward += reward
            rw_key = (state, action, new_state)
            self.rewards[rw_key] = float(reward)
            trans_key = (state, action)
            self.transits[trans_key][new_state] += 1
            if end_flag or trunc_flag:
                break
            state = new_state
        #print(rew_list)

        return total_reward
    
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                total_value = 0.0
                target_counts = self.transits[(state, action)]
                total_count = sum(k for k in target_counts.values())
                for next_state, count in target_counts.items():
                    reward = self.rewards[(state, action, next_state)]
                    new_action = self.select_action(next_state)

                    total_value += (reward + GAMMA * self.q_values[(next_state, new_action)]) * count / total_count


                self.q_values[(state, action)] = total_value


        #pprint(self.values)

        
        


if __name__ == "__main__":
    from pprint import pprint
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration-v2")
    test_env = gym.make(ENV_NAME)

    best_reward = 0.
    i_iter = 0
    while True:
        # print(i_iter)
        agent.play_n_random(100)
        agent.value_iteration()

        # test part
        total_reward = 0.
        for _ in range(TEST_EPISODES):
            total_reward += agent.play_episode(env=test_env)
        total_reward /= TEST_EPISODES
        #print(total_reward)

        if total_reward > best_reward:
            print(f"{i_iter}: Best reward updated {best_reward:.3} -> {total_reward:.3}")
            best_reward = total_reward

        writer.add_scalar("test", total_reward, i_iter)

        if total_reward >= 0.8:
            print(f"Solved: reward {total_reward} after {i_iter} iterations")
            break

        i_iter += 1

    writer.close()