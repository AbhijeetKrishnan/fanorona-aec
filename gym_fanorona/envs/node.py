from copy import deepcopy
from typing import List, Optional

from .action import FanoronaMove
from .fanorona_env import FanoronaEnv
from .enums import Piece


class FanoronaTreeNode:
    def __init__(self, env: FanoronaEnv):
        self.env = env
        self.state = env.state
        self.parent: "FanoronaTreeNode"
        self.depth = 0

    def terminal_test(self) -> bool:
        return self.state.is_done()

    def actions(self) -> List[FanoronaMove]:
        self.env.state = self.state
        return self.env.get_valid_moves()

    def utility(self, side: Piece) -> float:
        return float(self.state.utility(side))

    def result(self, action: FanoronaMove) -> "FanoronaTreeNode":
        self.env.state = self.state
        self.env.step(action)
        state_copy = deepcopy(self.env.state)
        result_node = FanoronaTreeNode(self.env)
        result_node.parent = self
        result_node.depth = self.depth + 1
        result_node.state = state_copy
        return result_node
