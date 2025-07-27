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
    
    