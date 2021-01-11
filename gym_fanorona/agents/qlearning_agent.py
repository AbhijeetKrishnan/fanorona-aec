import random
from typing import Dict, Tuple

from gym_fanorona.envs.action import FanoronaMove
from gym_fanorona.envs.node import FanoronaTreeNode

from .agent import FanoronaAgent


class QlearningAgent(FanoronaAgent):
    def __init__(
        self,
        learning_rate: float = 0.9,
        reward_decay: float = 0.9,
        e_greedy: float = 0.9,
        seed: int = 1,
    ):
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table: Dict[FanoronaTreeNode, Dict[FanoronaMove, float]] = dict()

        random.seed(seed)

    def move(self, node: FanoronaTreeNode) -> FanoronaMove:
        if random.random() < self.epsilon:
            # select action with max q-value in current state
            best_move = sorted(self.q_table[node].items(), key=lambda item: item[1])[0][
                0
            ]
        else:
            # select random action
            best_move = random.choice(node.actions())
        return best_move
