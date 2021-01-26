from enum import IntEnum


class Piece(IntEnum):
    def __str__(self):
        return self.name[0]  # just the first letter

    WHITE = 0
    BLACK = 1
    EMPTY = 2


class Direction(IntEnum):
    def __str__(self):
        if self.value == 4:
            return "-"
        else:
            return self.name

    def opposite(self):
        "Return the direction of opposite orientation to the current one e.g. NE.opposite() == SW"
        return Direction(8 - self.value)

    @staticmethod
    def dir_range():
        for i in range(9):
            yield Direction(i)

    SW = 0
    S = 1
    SE = 2
    W = 3
    X = 4  # No direction
    E = 5
    NW = 6
    N = 7
    NE = 8


class Reward(IntEnum):
    PAIKA = 0
    END_TURN = 0
    ILLEGAL_MOVE = -1
    LOSS = -1
    WIN = 1
    DRAW = 0.5
    NONE = 0
