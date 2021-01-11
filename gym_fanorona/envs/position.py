from typing import Iterator, List, Tuple, Union

from .constants import BOARD_COLS, BOARD_ROWS, BOARD_SQUARES
from .enums import Direction


class Position:
    def __init__(self, pos: Union[Tuple[int, int], str]):
        self.row: int = 0
        self.col: int = 0
        if isinstance(pos, tuple):
            self.row, self.col = pos
        elif isinstance(pos, str):
            col_str, row_str = list(pos)
            self.row = int(row_str) - 1
            self.col = ord(col_str) - ord("A")
        else:
            raise Exception(f"Object of type {type(pos)} was passed")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            raise NotImplementedError
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
        """Returns the resultant position obtained from adding a given unit direction vector (given by 'direction') to pos."""
        DISPLACEMENT_VECTORS = {
            0: (-1, -1),  # SW
            1: (-1, 0),  # S
            2: (-1, 1),  # SE
            3: (0, -1),  # W
            4: (0, 0),  # -
            5: (0, 1),  # E
            6: (1, -1),  # NW
            7: (1, 0),  # N
            8: (1, 1),  # NE
        }
        del_row, del_col = DISPLACEMENT_VECTORS[direction]
        res = (self.row + del_row, self.col + del_col)
        return Position(res)

    def get_valid_dirs(self) -> List[Direction]:
        """Get list of valid directions available from a given board position."""
        assert self.is_valid()
        row, col = self.row, self.col
        if row == 0 and col == 0:  # bottom-left corner
            dir_list = [Direction.N, Direction.NE, Direction.E]
        elif row == 2 and col == 0:  # middle-left
            dir_list = [
                Direction.S,
                Direction.SE,
                Direction.E,
                Direction.NE,
                Direction.N,
            ]
        elif row == 4 and col == 0:  # top-left corner
            dir_list = [Direction.S, Direction.SE, Direction.E]
        elif row == 0 and col == 8:  # bottom-right corner
            dir_list = [Direction.W, Direction.NW, Direction.N]
        elif row == 2 and col == 8:  # middle-right
            dir_list = [
                Direction.S,
                Direction.SW,
                Direction.W,
                Direction.NW,
                Direction.N,
            ]
        elif row == 4 and col == 8:  # top-right corner
            dir_list = [Direction.S, Direction.SW, Direction.W]
        elif row == 0 and col % 2 == 1:  # bottom edge 1
            dir_list = [Direction.W, Direction.N, Direction.E]
        elif row == 0 and col % 2 == 0:  # bottom edge 2
            dir_list = [
                Direction.W,
                Direction.NW,
                Direction.N,
                Direction.NE,
                Direction.E,
            ]
        elif row == 4 and col % 2 == 1:  # top edge 1
            dir_list = [Direction.W, Direction.S, Direction.E]
        elif row == 4 and col % 2 == 0:  # top edge 2
            dir_list = [
                Direction.W,
                Direction.SW,
                Direction.S,
                Direction.SE,
                Direction.E,
            ]
        elif col == 0:  # left edge
            dir_list = [Direction.S, Direction.E, Direction.N]
        elif col == 8:  # right edge
            dir_list = [Direction.S, Direction.W, Direction.N]
        elif (row + col) % 2 == 0:  # 8-point
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
        elif (row + col) % 2 == 1:  # 4-point
            dir_list = [Direction.S, Direction.W, Direction.N, Direction.E]
        return dir_list
