import random

import gym
from gym_fanorona.envs import (
    FanoronaMove,
    FanoronaState,
    Position,
    Direction,
    Reward,
    Piece,
)
import pytest

TEST_STATES = [
    "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0",  # start state
    "WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB B - - 1",  # capturing seq after D2->E3 approach
    "9/9/3W1B3/9/9 W - - 49",  # random endgame state
]


@pytest.fixture(scope="function")
def env():
    env = gym.make("fanorona-v0")
    env.reset()
    yield env
    env.close()


def test_is_valid(env):
    "Test 2 opening moves for validity"
    action1 = FanoronaMove(Position("D2"), Direction["NE"], 1, False)
    action2 = FanoronaMove(
        Position("E1"), Direction["NW"], 1, False
    )  # cannot move different piece in capturing sequence - illegal move
    assert action1.is_valid(env.state)
    env.step(action1)
    assert not action2.is_valid(env.state)


def test_is_done(env):
    "Test that is_done method is correctly identifying end of game states."
    board_str = "9/9/3W1B3/9/9 W - - 30"
    env.state = FanoronaState.set_from_board_str(board_str)
    action = FanoronaMove(Position("D3"), Direction["E"], 1, False)
    assert action.is_valid(env.state)
    _, _, done, _ = env.step(action)
    assert done


def test_utility(env):
    "Test that utility() method is correctly returning the verdict of end of game states."
    board_str = "9/9/3W1B3/9/9 W - - 30"
    env.state = FanoronaState.set_from_board_str(board_str)
    action = FanoronaMove(Position("D3"), Direction["E"], 1, False)
    assert action.is_valid(env.state)
    env.step(action)
    utility = env.state.utility(Piece.BLACK)
    print(env.state)
    assert utility is not None
    assert utility == Reward.LOSS


def test_end_turn(env):
    "Test end turn action by making an opening capturing move and testing all possible end turn actions in action space"
    action = FanoronaMove(
        Position("D2"), Direction(8), 1, False
    )  # D2 -> E3 approach capture
    env.step(action)
    action = FanoronaMove(
        Position("E5"), Direction["SE"], 1, False
    )  # Black: E5 -> F4 approach capture
    assert action.is_valid(env.state)
    env.step(action)
    for pos in Position.pos_range():
        for direction in Direction.dir_range():
            for capture_type in range(3):
                end_turn = True
                action = FanoronaMove(pos, direction, capture_type, end_turn)
                assert action.is_valid(env.state), f"Action: {action}"


def test_set_state_from_board_str(env):
    "Verify that set_state_from_board_str() sets the state correctly"
    for board_str in TEST_STATES:
        env.state = FanoronaState.set_from_board_str(board_str)
        assert env.state.get_board_str() == board_str


def test_get_board_str(env):
    "Test that get_board_str() works with randomly sampled observations without error"
    for _ in range(10):
        env.state = FanoronaState(env.observation_space.sample())
        print(env.state.get_board_str())


def test_in_capturing_seq(env):
    states = TEST_STATES
    env.state = FanoronaState.set_from_board_str(states[0])
    assert not env.state.in_capturing_seq()
    env.state = FanoronaState.set_from_board_str(states[1])
    assert not env.state.in_capturing_seq()
    env.state = FanoronaState.set_from_board_str(states[2])
    assert not env.state.in_capturing_seq()


def test_reset_visited_pos(env):
    states = TEST_STATES
    env.state = FanoronaState.set_from_board_str(states[0])
    for row, col in Position.coord_range():
        assert not env.state.visited[row][col]
    env.state = FanoronaState.set_from_board_str(states[1])
    env.state.reset_visited_pos()
    for row, col in Position.coord_range():
        assert not env.state.visited[row][col]
    env.state = FanoronaState.set_from_board_str(states[2])
    env.state.reset_visited_pos()
    for row, col in Position.coord_range():
        assert not env.state.visited[row][col]


def test_get_piece(env):
    "Verify that state.get_piece() works correctly"
    assert env.state.get_piece(Position("A1")) == Piece.WHITE
    assert env.state.get_piece(Position("A5")) == Piece.BLACK
    assert env.state.get_piece(Position("E3")) == Piece.EMPTY


def test_other_side(env):
    "Verify that state.other_side() works correctly"
    assert env.state.other_side() == Piece.BLACK


def test_piece_exists(env):
    "Verify that state.piece_exists() works correctly"
    assert env.state.piece_exists(Piece.WHITE)
    assert env.state.piece_exists(Piece.BLACK)
    assert env.state.piece_exists(Piece.EMPTY)

    env.state = FanoronaState.set_from_board_str("9/4W4/9/9/9 W - - 30")
    assert env.state.piece_exists(Piece.WHITE)
    assert not env.state.piece_exists(Piece.BLACK)
