import re
from enum import IntEnum
from typing import Optional, TypeAlias

from .utils import Direction, Position

ActionType: TypeAlias = int

END_TURN_ACTION: ActionType = 5 * 9 * 8 * 3  # end turn action encoding
ACTION_PATTERN = re.compile(
    r"(?P<from>[a-iA-I][1-5])(?P<direction>[1-9])(?P<move_type>[0-2])(?P<end_turn>[01])"
)


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
    ) -> None:
        """
        Initialize a Fanorona move object.

        Args:
            position (Position): The position of the move.
            direction (Direction): The direction of the move.
            move_type (MoveType): The type of the move.
            end_turn (bool): Indicates whether the move ends the turn.

        Returns:
            None
        """

        self.position = position
        self.direction = direction
        self.move_type = move_type
        self.end_turn = end_turn

    def __repr__(self) -> str:
        """
        Returns a string representation of the FanoronaMove object.

        The string includes the position, direction, move type, and whether it ends the turn.

        Returns:
            str: A string representation of the FanoronaMove object.
        """

        return f"<FanoronaMove: pos={str(self.position.to_human())}, dir={str(self.direction)}, type={self.move_type.name}, end?={self.end_turn}>"

    def __str__(self) -> str:
        """
        Returns a string representation of the FanoronaMove object.

        The string representation includes the position, direction, move type, and end turn information.

        Returns:
            str: The string representation of the FanoronaMove object.
        """

        return f"{self.position.to_human()}{self.direction.value}{self.move_type.value}{int(self.end_turn)}"

    def __eq__(self, other: object) -> bool:
        """
        Check if two FanoronaMove objects are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
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

    def to_action(self) -> ActionType:
        """
        Return integer encoding of FanoronaMove object.

        Returns:
            int: The integer encoding of the FanoronaMove object.
        """
        if self.end_turn:
            return END_TURN_ACTION
        else:
            pos_int = self.position.to_pos()
            dir_int = (
                self.direction.value
                - 1
                - (1 if self.direction.value >= 5 else 0)
            )  # to account for Direction.X
            move_type_int: int = self.move_type.value
            return pos_int * 8 * 3 + dir_int * 3 + move_type_int

    @staticmethod
    def from_action(action: ActionType) -> "FanoronaMove":
        """
        Converts integer-encoded action to a FanoronaMove object.

        Args:
            action (ActionType): The integer-encoded action.

        Returns:
            FanoronaMove: The corresponding FanoronaMove object.
        """
        action = int(action)  # to handle np.int type actions
        if action != END_TURN_ACTION:
            move_type_int = action % 3
            action = (action - move_type_int) // 3
            dir_int = action % 8
            action = (action - dir_int) // 8
            pos_int = action

            end_turn = False
            position = Position(pos_int)
            direction = Direction.from_raw_int(dir_int)
            move_type = MoveType(move_type_int)
        else:
            end_turn = True
            position = Position("I5")
            direction = Direction.NE
            move_type = MoveType.WITHDRAWAL
        result = FanoronaMove(position, direction, move_type, end_turn)
        return result

    @staticmethod
    def from_str(action_string: str) -> Optional["FanoronaMove"]:
        """
        Return FanoronaMove object from string representation of move.

        Move is represented by `FFDCE`, where
            FF - initial position of piece to be moved in human-readable coordinates e.g. A3, G4 (case-insensitive)
            D  - direction in which piece is moved
            C  - move type (paika (0), approach (1) or withdrawal (2))
            E  - end turn (yes (1) or no (0))

        Args:
            action_string (str): The string representation of the move.

        Returns:
            Optional["FanoronaMove"]: The FanoronaMove object created from the string representation.
        """

        match = ACTION_PATTERN.match(action_string)
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
END_TURN = FanoronaMove(
    Position("I5"), Direction.NE, MoveType.WITHDRAWAL, True
)
