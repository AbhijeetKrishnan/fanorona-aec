from typing import Optional
from enum import IntEnum

from .utils import Direction, Position


class MoveType(IntEnum):
    PAIKA = 0
    APPROACH = 1
    WITHDRAWAL = 2
class FanoronaMove:
    def __init__(
        self,
        position: Position,
        direction: Direction,
        capture_type: MoveType,
        end_turn: bool,
    ):
        self.position = position
        self.direction = direction
        self.capture_type = capture_type
        self.end_turn = end_turn

    def __repr__(self):
        return f"<FanoronaMove: position={str(self.position.to_human())}, \
            direction={str(self.direction)},                              \
            capture_type={self.capture_type},                             \
            end_turn={self.end_turn}>"

    def __str__(self):
        return f"{self.position.to_human()}{self.direction.value}{self.capture_type}{int(self.end_turn)}"

    def to_action(self) -> int:
        "Return integer encoding of FanoronaMove object"
        pass

    @staticmethod
    def action_to_move(action: int) -> "FanoronaMove":
        "Converts NN-style action to a FanoronaMove object"
        pass

    @staticmethod
    def str_to_move(action_string: str) -> Optional["FanoronaMove"]:
        """
        Return FanoronaMove object from string representation of move.

        Move is represented by 'FFDCE', where
            FF - initial position of piece to be moved in human-readable coordinates (e.g. A3, G4)
            D  - direction in which piece is moved
            C  - capture type (paika (0), approach (1) or withdrawal (2))
            E  - end turn (yes (1) or no (0))
        """
        import re

        action_pattern = re.compile(
            r"(?P<from>[A-I][1-5])(?P<direction>[0-8])(?P<capture_type>[0-2])(?P<end_turn>[01])"
        )
        match = action_pattern.match(action_string)
        if not match:
            ret_val = None
        else:
            ret_val = FanoronaMove(
                Position(match.group("from")),
                Direction(int(match.group("direction"))),
                MoveType(match.group("capture_type")),
                bool(int(match.group("end_turn"))),
            )
        return ret_val