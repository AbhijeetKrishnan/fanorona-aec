from gym_fanorona.envs import FanoronaMove, Position, Direction
import pytest


def test_get_action():
    action_str = "A2A300"
    action = FanoronaMove.get_action(action_str)
    assert action.position == Position("A2")
    assert action.direction == Direction["N"]
    assert action.capture_type == 0
    assert not action.end_turn
