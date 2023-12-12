import pytest

from fanorona_aec.env.fanorona_move import END_TURN, FanoronaMove, MoveType
from fanorona_aec.env.utils import Direction, Position


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            FanoronaMove(Position("A1"), Direction.SE, MoveType.PAIKA, False),
            "<FanoronaMove: pos=A1, dir=SE, type=PAIKA, end?=False>",
        ),
        (
            FanoronaMove(
                Position("C3"), Direction.W, MoveType.APPROACH, False
            ),
            "<FanoronaMove: pos=C3, dir=W, type=APPROACH, end?=False>",
        ),
        (
            FanoronaMove(
                Position("G5"), Direction.N, MoveType.WITHDRAWAL, False
            ),
            "<FanoronaMove: pos=G5, dir=N, type=WITHDRAWAL, end?=False>",
        ),
        (
            END_TURN,
            "<FanoronaMove: pos=I5, dir=NE, type=WITHDRAWAL, end?=True>",
        ),
    ],
)
def test_repr(test_input, expected):
    "Test that move has the expected representation"
    assert repr(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            FanoronaMove(Position("A1"), Direction.SE, MoveType.PAIKA, False),
            "A1300",
        ),
        (
            FanoronaMove(
                Position("C3"), Direction.W, MoveType.APPROACH, False
            ),
            "C3410",
        ),
        (
            FanoronaMove(
                Position("G5"), Direction.N, MoveType.WITHDRAWAL, False
            ),
            "G5820",
        ),
        (END_TURN, "I5921"),
    ],
)
def test_str(test_input, expected):
    "Test that action has the expected string representation"
    assert str(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (FanoronaMove(Position("A1"), Direction.SW, MoveType.PAIKA, False), 0),
        (
            FanoronaMove(
                Position("C3"), Direction.W, MoveType.APPROACH, False
            ),
            490,
        ),
        (
            FanoronaMove(
                Position("G5"), Direction.N, MoveType.WITHDRAWAL, False
            ),
            1028,
        ),
        (END_TURN, 5 * 9 * 8 * 3),
    ],
)
def test_to_action(test_input, expected):
    "Test that move has the correct integer encoding"
    assert test_input.to_action() == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (0, FanoronaMove(Position("A1"), Direction.SW, MoveType.PAIKA, False)),
        (
            490,
            FanoronaMove(
                Position("C3"), Direction.W, MoveType.APPROACH, False
            ),
        ),
        (
            1028,
            FanoronaMove(
                Position("G5"), Direction.N, MoveType.WITHDRAWAL, False
            ),
        ),
        (5 * 9 * 8 * 3, END_TURN),
    ],
)
def test_from_action(test_input, expected):
    "Test that input integer is correctly returned as a FanoronaMove object"
    assert FanoronaMove.from_action(test_input) == expected


@pytest.mark.parametrize("test_input", range(5 * 9 * 8 * 3 + 1))
def test_all_action_encodings(test_input):
    "Test that all actions decode and encode back to the same integer"
    assert FanoronaMove.from_action(test_input).to_action() == test_input


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            "A1300",
            FanoronaMove(Position("A1"), Direction.SE, MoveType.PAIKA, False),
        ),
        (
            "C3410",
            FanoronaMove(
                Position("C3"), Direction.W, MoveType.APPROACH, False
            ),
        ),
        (
            "G5820",
            FanoronaMove(
                Position("G5"), Direction.N, MoveType.WITHDRAWAL, False
            ),
        ),
        ("I5921", END_TURN),
    ],
)
def test_from_str(test_input, expected):
    "Test that input string generates the correct move object"
    assert FanoronaMove.from_str(test_input) == expected
