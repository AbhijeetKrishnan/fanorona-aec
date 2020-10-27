import random

from .agent import FanoronaAgent

class RandomAgent(FanoronaAgent):

    def __init__(self, seed=None):
        super(RandomAgent, self).__init__()
        random.seed(seed)

    def move(self, env):
        moves = env.get_valid_moves()
        return random.choice(moves)