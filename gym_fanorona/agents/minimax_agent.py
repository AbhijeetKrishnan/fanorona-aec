from .agent import FanoronaAgent
from gym_fanorona.envs import FanoronaEnv
from gym_fanorona.envs.node import FanoronaTreeNode
from gym_fanorona.envs.action import FanoronaMove
from gym_fanorona.envs.state import FanoronaState
from gym_fanorona.envs.constants import MOVE_LIMIT

from typing import Callable, Optional, cast

INF = float("inf")


class MinimaxAgent(FanoronaAgent):
    """Reference: Stuart Russell and Peter Norvig. 2009. Artificial Intelligence: A Modern Approach (3rd. ed.). Prentice Hall Press, USA.
    
    terminal_test(node: FanoronaTreeNode) -> bool: node is an end state for the game
    utility(node: FanoronaTreeNode, side: Piece) -> float: value of the end state (this is dependent on who's POV it is)
    actions(node: FanoronaTreeNode) -> List[FanoronaMove]: list of legal moves possible from given state
    """

    def __init__(
        self,
        cutoff: Optional[int] = None,
        heuristic: Optional[Callable[[FanoronaTreeNode], float]] = None,
    ):
        super(MinimaxAgent, self).__init__()
        if cutoff is not None and heuristic is None:
            raise Exception("cutoff provided without supplying a heuristic.")
        self.cutoff = cutoff
        self.heuristic = cast(Callable[[FanoronaTreeNode], float], heuristic)

    def move(self, env: FanoronaEnv) -> FanoronaMove:
        def max_value(node: FanoronaTreeNode) -> float:
            print(f"exploring state {node.env.state}")
            if node.terminal_test():
                return node.utility(self.side)
            elif self.cutoff is not None and node.depth >= self.cutoff:
                return self.heuristic(node)
            value = -INF
            for action in node.actions():
                result = node.result(action)
                if result.env.state.turn_to_play == node.env.state.turn_to_play:
                    value = max(value, max_value(result))
                else:
                    value = max(value, min_value(result))
            return value

        def min_value(node: FanoronaTreeNode) -> float:
            print(f"exploring state {node.env.state}")
            if node.terminal_test():
                return node.utility(self.side)
            elif self.cutoff is not None and node.depth >= self.cutoff:
                return self.heuristic(node)
            value = INF
            for action in node.actions():
                result = node.result(action)
                if result.env.state.turn_to_play == node.env.state.turn_to_play:
                    value = min(value, min_value(result))
                else:
                    value = min(value, max_value(result))
            return value

        value, best_value = -INF, -INF
        root = FanoronaTreeNode(env)
        for action in root.actions():
            print(f"Evaluating action {action} in state {root.env.state}")
            result = root.result(action)
            if (
                result.env.state.turn_to_play == root.env.state.turn_to_play
            ):  # same player's move
                value = max(value, max_value(result))
            else:
                value = max(value, min_value(result))  # other player's move
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
