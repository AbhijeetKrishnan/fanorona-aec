import pytest

from fanorona_aec.env.utils import Direction, Position

# fmt: off
POS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
)

HUMAN = (
    "A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "I1",
    "A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "I2",
    "A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3", "I3",
    "A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4", "I4",
    "A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5", "I5",
)
# fmt: on


@pytest.mark.parametrize("test_input,expected", zip(POS, HUMAN))
def test_convert_coords_to_human(test_input, expected):
    """Test that converting to human-readable coords upon initialization using row-col coords returns
    the correct values for all valid board coords
    """
    assert Position(test_input).to_human() == expected


@pytest.mark.parametrize("test_input,expected", zip(HUMAN, POS))
def test_convert_human_to_coords(test_input, expected):
    """Test that converting to coords upon initialization using human-readable coords returns the
    correct values for all valid board coords
    """
    assert Position(test_input).to_coords() == expected


@pytest.mark.parametrize("test_input,expected", zip(POS, range(45)))
def test_convert_coords_to_pos(test_input, expected):
    "Test that initializing using coords returns the correct values for all valid board coords"
    assert Position(test_input).to_pos() == expected


@pytest.mark.parametrize("test_input,expected", zip(range(45), POS))
def test_convert_pos_to_coords(test_input, expected):
    """
    Test that converting to coords upon initialization using pos returns the correct values for
    all valid board positions
    """
    assert Position(test_input).to_coords() == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ((0, 0), [Direction.N, Direction.NE, Direction.E]),
        (
            (2, 0),
            [
                Direction.S,
                Direction.SE,
                Direction.E,
                Direction.NE,
                Direction.N,
            ],
        ),
        ((4, 0), [Direction.S, Direction.SE, Direction.E]),
        ((0, 8), [Direction.W, Direction.NW, Direction.N]),
        (
            (2, 8),
            [
                Direction.S,
                Direction.SW,
                Direction.W,
                Direction.NW,
                Direction.N,
            ],
        ),
        ((4, 8), [Direction.S, Direction.SW, Direction.W]),
        ((0, 1), [Direction.W, Direction.N, Direction.E]),
        (
            (0, 2),
            [
                Direction.W,
                Direction.NW,
                Direction.N,
                Direction.NE,
                Direction.E,
            ],
        ),
        pytest.param(
            (4, 1), [Direction.W, Direction.N, Direction.E], marks=pytest.mark.xfail
        ),
        (
            (4, 2),
            [
                Direction.W,
                Direction.SW,
                Direction.S,
                Direction.SE,
                Direction.E,
            ],
        ),
        ((1, 0), [Direction.S, Direction.E, Direction.N]),
        ((1, 8), [Direction.S, Direction.W, Direction.N]),
        (
            (1, 1),
            [
                Direction.S,
                Direction.SW,
                Direction.W,
                Direction.NW,
                Direction.N,
                Direction.NE,
                Direction.E,
                Direction.SE,
            ],
        ),
        ((1, 2), [Direction.S, Direction.W, Direction.N, Direction.E]),
    ],
)
def test_get_valid_dirs(test_input, expected):
    "Verify that Position.get_valid_dirs() returns the right directions for all possible types of positions"
    assert sorted(Position(test_input).get_valid_dirs()) == sorted(expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (1, (-1, -1)),
        (2, (-1, 0)),
        (3, (-1, 1)),
        (4, (0, -1)),
        (5, (0, 0)),
        (6, (0, 1)),
        (7, (1, -1)),
        (8, (1, 0)),
        (9, (1, 1)),
    ],
)
def test_displace(test_input, expected):
    "Test that displace() returns the right result for all possible directions from a position"
    assert Position((0, 0)).displace(Direction(test_input)) == Position(expected)
