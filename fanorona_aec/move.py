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
        move_type: MoveType,
        end_turn: bool,
    ):
        self.position = position
        self.direction = direction
        self.move_type = move_type
        self.end_turn = end_turn

    def __repr__(self):
        return f"<FanoronaMove: pos={str(self.position.to_human())}, dir={str(self.direction)}, type={self.move_type.name}, end?={self.end_turn}>"

    def __str__(self):
        return f"{self.position.to_human()}{self.direction.value}{self.move_type.value}{int(self.end_turn)}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FanoronaMove):
            return NotImplemented
        elif (
            self.position == other.position
            and self.direction == other.direction
            and self.move_type == other.move_type
            and self.end_turn == other.end_turn
        ):
            return True
        else:
            return False

    def to_action(self) -> int:
        "Return integer encoding of FanoronaMove object"
        if self.end_turn:
            return 5 * 9 * 8 * 3
        else:
            pos_int: int = self.position.to_pos()
            dir_int: int = self.direction.value - 1 - (
                1 if self.direction.value >= 5 else 0
            )  # to account for Direction.X
            move_type_int: int = self.move_type.value
            return pos_int * 8 * 3 + dir_int * 3 + move_type_int

    @staticmethod
    def action_to_move(action: int) -> "FanoronaMove":
        "Converts integer-encoded action to a FanoronaMove object"
        action = int(action)  # to handle np.int type actions
        result = FanoronaMove(Position("I5"), Direction.NE, MoveType.WITHDRAWAL, True)
        if action != 5 * 9 * 8 * 3:
            result.end_turn = False
            move_type_int = action % 3
            action = (action - move_type_int) // 3
            dir_int = action % 8
            action = (action - dir_int) // 8
            if dir_int >= 4:
                dir_int += 1  # to account for Direction.X
            dir_int += 1  # Direction enum starts from 1
            pos_int = action
            result.position = Position(pos_int)
            result.direction = Direction(dir_int)
            result.move_type = MoveType(move_type_int)
        return result

    @staticmethod
    def str_to_move(action_string: str) -> Optional["FanoronaMove"]:
        """
        Return FanoronaMove object from string representation of move.

        Move is represented by 'FFDCE', where
            FF - initial position of piece to be moved in human-readable coordinates e.g. A3, G4 (case-insensitive)
            D  - direction in which piece is moved
            C  - move type (paika (0), approach (1) or withdrawal (2))
            E  - end turn (yes (1) or no (0))
        """
        import re

        action_pattern = re.compile(
            r"(?P<from>[a-iA-I][1-5])(?P<direction>[1-9])(?P<move_type>[0-2])(?P<end_turn>[01])"
        )
        match = action_pattern.match(action_string)
        if not match:
            ret_val = None
        else:
            ret_val = FanoronaMove(
                Position(match.group("from")),
                Direction(int(match.group("direction"))),
                MoveType(int(match.group("move_type"))),
                bool(int(match.group("end_turn"))),
            )
        return ret_val


# End turn move needs to encode to a value of 5 * 9 * 8 * 3
END_TURN = FanoronaMove(Position("I5"), Direction.NE, MoveType.WITHDRAWAL, True)
