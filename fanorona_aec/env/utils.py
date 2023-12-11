from enum import IntEnum
from typing import Iterator, List, Tuple, Union

BOARD_ROWS = 5
BOARD_COLS = 9
MOVE_LIMIT = 44

BOARD_SQUARES = BOARD_ROWS * BOARD_COLS


class Piece(IntEnum):
    def __str__(self):
        return str(self.name)[0]  # just the first letter

    def other(self) -> "Piece":
        match self:
            case Piece.WHITE:
                return Piece.BLACK
            case Piece.BLACK:
                return Piece.WHITE
            case _:
                raise ValueError(f"Cannot define `other()` for {str(self)}")

    WHITE = 0
    BLACK = 1
    EMPTY = 2


class Direction(IntEnum):
    "Uses numpad coordinates to represent directions"

    def __str__(self):
        if self.name[0] == "X":
            return "-"
        else:
            return self.name

    @staticmethod
    def from_str(dir_str: str) -> "Direction":
        try:
            if dir_str == "-":
                return Direction.X
            else:
                return Direction[dir_str]
        except KeyError:
            raise ValueError(f"Invalid direction: {dir_str}")

    def opposite(self) -> "Direction":
        "Return the direction of opposite orientation to the current one e.g. NE.opposite() == SW"
        return Direction(10 - self.value)

    def as_vector(self) -> Tuple:
        "Return the unit vector representation of the direction (with tail assumed at (0, 0))"
        # fmt: off
        DISPLACEMENT_VECTORS = {
            1: (-1, -1),
            2: (-1,  0),
            3: (-1,  1),
            4: ( 0, -1),
            5: ( 0,  0),
            6: ( 0,  1),
            7: ( 1, -1),
            8: ( 1,  0),
            9: ( 1,  1),
        }
        # fmt: on
        return DISPLACEMENT_VECTORS[self.value]

    @staticmethod
    def dir_range() -> Iterator["Direction"]:
        for i in range(9):
            yield Direction(i)

    # fmt: off
    SW = 1
    S  = 2
    SE = 3
    W  = 4
    X  = 5  # No direction
    E  = 6
    NW = 7
    N  = 8
    NE = 9
    # fmt: on


class Position:
    def __init__(self, pos: Union[Tuple[int, int], str, int]):
        self.row: int = 0
        self.col: int = 0
        if isinstance(pos, tuple):
            self.row, self.col = pos
        elif isinstance(pos, str):
            col_str, row_str = list(pos)
            self.row = int(row_str) - 1
            self.col = ord(col_str) - ord("A")
        elif isinstance(pos, int):
            self.col = pos % BOARD_COLS
            self.row = (pos - self.col) // BOARD_COLS
        else:
            raise Exception(
                f"Cannot create a Position object from an object of type \
                    {type(pos)}"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            raise NotImplementedError(
                f"Cannot compare {type(self)} with object of type \
                    {type(other)}"
            )
        return self.row == other.row and self.col == other.col

    def __repr__(self):
        return f"<Position: {self.to_human()}>"

    @staticmethod
    def pos_range() -> Iterator["Position"]:
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                yield Position((row, col))

    @staticmethod
    def coord_range() -> Iterator[Tuple[int, int]]:
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                yield row, col

    @staticmethod
    def human_range() -> Iterator[str]:
        for pos in Position.pos_range():
            yield pos.to_human()

    def to_pos(self) -> int:
        return self.row * BOARD_COLS + self.col

    def to_coords(self) -> Tuple[int, int]:
        return self.row, self.col

    def to_human(self) -> str:
        return f"{chr(65 + self.col)}{self.row + 1}"

    def is_valid(self) -> bool:
        "Returns True if the position represents a valid square on the Fanorona board."
        return 0 <= self.row < BOARD_ROWS and 0 <= self.col < BOARD_COLS

    def displace(self, direction: Direction) -> "Position":
        """
        Returns the resultant position obtained from adding a given unit
        direction vector (given by 'direction') to pos.

        Parameters:
            direction (Direction): The direction in which to displace the position.

        Returns:
            Position: The resultant position after displacement.
        """
        del_row, del_col = direction.as_vector()
        res = (self.row + del_row, self.col + del_col)
        return Position(res)

    def get_valid_dirs(self) -> List[Direction]:
        """Get list of valid directions available from a given board position.

        Returns:
            List[Direction]: A list of valid directions available from the current board position.

        Raises:
            ValueError: If unexpected coordinates are encountered.
        """
        row, col = self.row, self.col
        match (row, col):
            case (0, 0):  # bottom-left corner
                dir_list = [Direction.N, Direction.NE, Direction.E]
            case (2, 0):  # middle-left
                dir_list = [
                    Direction.S,
                    Direction.SE,
                    Direction.E,
                    Direction.NE,
                    Direction.N,
                ]
            case (4, 0):  # top-left corner
                dir_list = [Direction.S, Direction.SE, Direction.E]
            case (0, 8):  # bottom-right corner
                dir_list = [Direction.W, Direction.NW, Direction.N]
            case (2, 8):  # middle-right
                dir_list = [
                    Direction.S,
                    Direction.SW,
                    Direction.W,
                    Direction.NW,
                    Direction.N,
                ]
            case (4, 8):  # top-right corner
                dir_list = [Direction.S, Direction.SW, Direction.W]
            case (0, col) if col % 2 == 1:  # bottom edge 1
                dir_list = [Direction.W, Direction.N, Direction.E]
            case (0, col) if col % 2 == 0:  # bottom edge 2
                dir_list = [
                    Direction.W,
                    Direction.NW,
                    Direction.N,
                    Direction.NE,
                    Direction.E,
                ]
            case (4, col) if col % 2 == 1:  # top edge 1
                dir_list = [Direction.W, Direction.S, Direction.E]
            case (4, col) if col % 2 == 0:  # top edge 2
                dir_list = [
                    Direction.W,
                    Direction.SW,
                    Direction.S,
                    Direction.SE,
                    Direction.E,
                ]
            case (_, 0):  # left edge
                dir_list = [Direction.S, Direction.E, Direction.N]
            case (_, 8):  # right edge
                dir_list = [Direction.S, Direction.W, Direction.N]
            case (row, col) if (row + col) % 2 == 0:  # 8-point
                dir_list = [
                    Direction.S,
                    Direction.SW,
                    Direction.W,
                    Direction.NW,
                    Direction.N,
                    Direction.NE,
                    Direction.E,
                    Direction.SE,
                ]
            case (row, col) if (row + col) % 2 == 1:  # 4-point
                dir_list = [Direction.S, Direction.W, Direction.N, Direction.E]
            case _:
                raise ValueError(
                    f"Unexpected coords encountered: row=\
                                 {row}, col={col}"
                )
        return dir_list
