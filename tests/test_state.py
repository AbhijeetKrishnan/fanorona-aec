from fanorona_aec import fanorona_v0

from fanorona_aec.move import FanoronaMove
from fanorona_aec.state import FanoronaState
from fanorona_aec.utils import (
    Position,
    Direction,
    Piece,
)
import pytest

TEST_STATES = [
    "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - - 0",  # start state
    "WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB B - - - 1",  # capturing seq after D2->E3 approach
    "9/9/3W1B3/9/9 W - - - 49",  # random endgame state
]

# TODO: fix and update these tests


@pytest.fixture(scope="function")
def env():
    env = fanorona_v0.env()
    env.reset()
    yield env
    env.close()


def test_is_game_over(env):
    "Test that is_done method is correctly identifying end of game states."
    board_str = "9/9/3W1B3/9/9 W - - - 30"
    env.state.set_from_board_str(board_str)
    move = FanoronaMove(Position("D3"), Direction["E"], 1, False)
    env.state.push(move)
    assert env.state.is_game_over()


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
