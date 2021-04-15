from fanorona_aec.move import FanoronaMove
from fanorona_aec.utils import Position, Direction

# TODO: fill in these tests
def test_repr():
    "Test that action has the expected representation"
    pass


def test_str():
    "Test that action has the expected string representation"
    pass


def test_to_action():
    "Test that move has the correct integer encoding"
    pass


def test_action_to_move():
    "Test that input integer is correctly returned as a FanoronaMove object"
    pass


def test_str_to_move():
    move_str = "A2700"
    move = FanoronaMove.str_to_move(move_str)
    assert move.position == Position("A2")
    assert move.direction == Direction["N"]
    assert move.capture_type == 0
    assert not move.end_turn
