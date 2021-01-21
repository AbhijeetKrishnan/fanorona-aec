from collections import defaultdict
from typing import Dict, Tuple, Optional, cast, Callable

from gym_fanorona.envs.action import FanoronaMove
from gym_fanorona.envs.node import FanoronaTreeNode
from gym_fanorona.envs.fanorona_env import FanoronaEnv

from .agent import FanoronaAgent


class QlearningAgent(FanoronaAgent):
    def __init__(
        self,
        Ne: int = 5,
        Rplus: float = 2,
        alpha: Optional[Callable] = None,
        gamma: float = 0.9,
    ):
        super(QlearningAgent, self).__init__()
        self.alpha: Callable
        self.gamma = gamma  # reward decay
        self.Ne = Ne  # iteration limit in exploration function
        self.Rplus = Rplus  # large value to assign before iteration limit
        self.Q: Dict[
            Tuple[FanoronaTreeNode, Optional[FanoronaMove]], float
        ] = defaultdict(float)
        self.Nsa: Dict[
            Tuple[FanoronaTreeNode, Optional[FanoronaMove]], float
        ] = defaultdict(int)
        self.s: Optional[FanoronaTreeNode] = None  # previous state
        self.a: Optional[FanoronaMove] = None  # previous action
        self.r: Optional[float] = None  # previous reward

        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1.0 / (1 + n)  # udacity video

    def f(self, u, n):
        """Exploration function. Returns fixed Rplus until
        agent has visited state, action a Ne number of times.
        Same as ADP agent in book."""
        if n < self.Ne:
            return self.Rplus
        else:
            return u

    def move(self, env: FanoronaEnv) -> FanoronaMove:
        node = FanoronaTreeNode(env)
        if node.terminal_test():
            self.Q[node, None] = node.utility()
        if self.s:
            self.Nsa[self.s, self.a] += 1
            max_q = max(self.Q[node, action] for action in node.actions())
            self.Q[self.s, self.a] += self.alpha(self.Nsa[self.s, self.a]) * (
                cast(float, self.r) + self.gamma * max_q - self.Q[self.s, self.a]
            )
        self.s, self.r = node, node.utility()
        self.a = max(
            node.actions(),
            key=lambda action: self.f(self.Q[node, action], self.Nsa[node, action]),
        )
        return self.a
