from copy import deepcopy
from typing import List, Optional

from .action import FanoronaMove
from .fanorona_env import FanoronaEnv


class FanoronaTreeNode:
    def __init__(self, env: FanoronaEnv):
        self.env = env
        self.parent: "FanoronaTreeNode"
        self.depth = 0

    def terminal_test(self) -> bool:
        return self.env.state.is_done()

    def actions(self) -> List[FanoronaMove]:
        return self.env.get_valid_moves()

    def utility(self) -> float:
        return float(self.env.state.utility())

    def result(self, action: FanoronaMove) -> "FanoronaTreeNode":
        env_copy = deepcopy(self.env)
        env_copy.step(action)
        result_node = FanoronaTreeNode(env_copy)
        result_node.parent = self
        result_node.depth = self.depth + 1
        return result_node
