import random
from typing import List


class Environment:
    def __init__(self):
        self.steps_left = 10

    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0, 0.0]

    def get_actions(self) -> List[int]:
        return [0, 1]

    def is_done(self):
        return self.steps_left == 0

    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game ended!")

        self.steps_left -= 1
        return random.random()  # in this example only random


class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env: Environment):
        observations = env.get_observation()
        action = env.get_actions()

        # here, random for the example:
        choosen_action = random.choice(action)
        reward = env.action(choosen_action)
        self.total_reward += reward


if __name__ == "__main__":

    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print(f"Total: {agent.total_reward:.4f}")
