from fanorona_aec.move import FanoronaMove, MoveType
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

# TODO: investigate using fixtures to initialize states dynamically for use as parameters to test functions


@pytest.fixture(scope="function")
def start_state():
    state = FanoronaState()
    state.reset()
    yield state


# TODO: write test
def test_str():
    "Test that state has correct string representation"
    pass


# TODO: write test
def test_to_svg():
    "Test that state is correctly output as an svg file"
    pass


@pytest.mark.parametrize(
    "test_input,expected",
    [("A1", Piece.WHITE), ("A5", Piece.BLACK), ("E3", Piece.EMPTY),],
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
        pytest.param("9/4W4/9/9/9 W - - - 30", Piece.BLACK, marks=pytest.mark.xfail,),
    ],
)
def test_piece_exists(state_str, piece):
    "Verify that state.piece_exists() works correctly"
    state = FanoronaState().set_from_board_str(state_str)
    assert state.piece_exists(piece)


def test_push():
    "Test that state moves to correct successor state upon calling push() with a move"
    pass


# TODO: rewrite this using a parametrized test
def test_is_game_over(start_state):
    "Test that is_done method is correctly identifying end of game states."
    board_str = "9/9/3W1B3/9/9 W - - - 30"
    start_state.set_from_board_str(board_str)
    move = FanoronaMove(Position("D3"), Direction["E"], MoveType.APPROACH, False)
    start_state.push(move)
    assert start_state.is_game_over()


def test_get_result():
    "Test that the correct result is returned for done games"
    pass


def test_reset():
    "Test that reset() correctly resets the state to the start position"
    pass


# TODO: add more test states
def test_set_state_from_board_str(start_state):
    "Verify that set_state_from_board_str() sets the state correctly"
    for board_str in TEST_STATES:
        start_state.set_from_board_str(board_str)
        assert str(start_state) == board_str


def test_get_observation():
    "Test that the correct observation is returned which represents the state"
    pass


def test_is_valid():
    "Test that moves are correctly validated with respect to the current state"
    pass


def test_legal_moves():
    "Test that state.legal_moves correctly represents the list of legal moves available from the state"
    pass
