import numpy as np
import pytest

from fanorona_aec.env.fanorona_move import FanoronaMove
from fanorona_aec.env.fanorona_state import FanoronaState
from fanorona_aec.env.utils import Piece, Position

TEST_STATE_STRS = [
    "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - - 0",  # start state
    # capturing seq after D2->E3 approach
    "WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB B - - - 1",
    "9/9/3W1B3/9/9 W - - - 44",  # random endgame state
    "9/4W4/9/9/9 W - - - 30",  # terminal state
    "3W1WW2/5W3/7WW/2W6/9 B - - - 17",  # pathologic endgame state
]


@pytest.fixture(scope="function")
def test_state_list():
    yield map(
        lambda board_str: FanoronaState().set_from_board_str(board_str), TEST_STATE_STRS
    )


@pytest.fixture(scope="function")
def start_state():
    state = FanoronaState()
    state.reset()
    yield state


def test_str(test_state_list):
    "Test that state has correct string representation"
    for test_state, expected_str in zip(test_state_list, TEST_STATE_STRS):
        assert str(test_state) == expected_str


def test_to_svg(test_state_list):
    "Test that state is correctly output as an svg file"
    for test_state in test_state_list:
        test_state.to_svg()


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("A1", Piece.WHITE),
        ("A5", Piece.BLACK),
        ("E3", Piece.EMPTY),
    ],
)
def test_get_piece(start_state, test_input, expected):
    "Verify that state.get_piece() works correctly"
    assert start_state.get_piece(Position(test_input)) == expected


@pytest.mark.parametrize(
    "state_str,piece",
    [
        ("WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - - 0", Piece.WHITE),
        ("WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - - 0", Piece.BLACK),
        ("WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - - 0", Piece.EMPTY),
        ("9/4W4/9/9/9 W - - - 30", Piece.WHITE),
        pytest.param(
            "9/4W4/9/9/9 W - - - 30",
            Piece.BLACK,
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_piece_exists(state_str, piece):
    "Verify that state.piece_exists() works correctly"
    state = FanoronaState().set_from_board_str(state_str)
    assert state.piece_exists(piece)


def test_push(test_state_list):
    """Test that state moves to a valid successor state upon calling push() with a move until end of
    the game
    """
    for state in test_state_list:
        while not state.done:
            action = np.random.default_rng(seed=0).choice(state.legal_moves)
            move = FanoronaMove.from_action(action)
            state.push(move)


def test_done(test_state_list, expected_list=[False, False, True, True, True]):
    "Test that done property is correctly identifying end of game states."
    for test_state, expected in zip(test_state_list, expected_list):
        assert test_state.done == expected


def test_reset(test_state_list, start_state):
    "Test that reset() correctly resets the state to the start position"
    for test_state in test_state_list:
        test_state.reset()
        assert test_state == start_state


def test_set_from_board_str(test_state_list, test_board_str_list=TEST_STATE_STRS):
    "Verify that set_from_board_str() sets the state correctly"
    for test_str, expected in zip(test_board_str_list, test_state_list):
        assert FanoronaState().set_from_board_str(test_str) == expected


def test_get_observation(test_state_list):
    "Test that the correct observation is returned which represents the state"
    for state in test_state_list:
        obs = state.get_observation(0)
        obs = state.get_observation(1)


@pytest.mark.skip(reason="Not implemented")
def test_is_valid():
    "Test that moves are correctly validated with respect to the current state"
    pass


@pytest.mark.skip(reason="Not implemented")
def test_legal_moves():
    "Test that state.legal_moves correctly represents the list of legal moves available from the state"
    pass
