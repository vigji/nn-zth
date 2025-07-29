import gymnasium as gym
import random

e = gym.make("CartPole-v1")

obs, info = e.reset()
print(obs)
print(info)

print(e.action_space)

print(e.observation_space)

total_reward = 0
while True:
    action = e.action_space.sample()
    obs, r, done, truncated, _ = e.step(action)
    total_reward += r
    if done or truncated:
        break

print("Total reward: ", total_reward)


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, epsilon: float = 0.1):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action: gym.core.WrapperActType) -> gym.core.WrapperActType:

        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            print(f"Random action {action}")

        return action
