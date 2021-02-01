from gym_fanorona.envs.fanorona_env import FanoronaEnv
from gym_fanorona.envs.action import FanoronaMove


class FanoronaAgent:
    def __init__(self):
        self.reward = 0
        self.side = None

    def move(self, env: FanoronaEnv) -> FanoronaMove:
        pass

    def receive_reward(self, reward: int) -> None:
        self.reward += reward
