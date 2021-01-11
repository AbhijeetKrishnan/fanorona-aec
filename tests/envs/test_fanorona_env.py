import os
import random

import gym
import pytest
from gym_fanorona.envs import (
    MOVE_LIMIT,
    Direction,
    FanoronaMove,
    FanoronaState,
    Position,
)

TEST_STATES = [
    "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0",  # start state
    "WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB B - - 1",  # capturing seq after D2->E3 approach
    "9/9/3W1B3/9/9 W - - 49",  # random endgame state
]


def test_make():
    "Verify that gym.make() executes without error"
    env = gym.make("fanorona-v0")
    env.close()


def test_reset(env):
    "Verify that reset() executes without error"
    env.reset()


@pytest.fixture(scope="function")
def env():
    env = gym.make("fanorona-v0")
    env.reset()
    yield env
    env.close()


def test_reset_starting(env):
    "Verify that reset() sets the board state to the correct starting position"
    assert (
        env.state.get_board_str()
        == "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0"
    )  # starting position


def test_render(env):
    "Verify that render() executes without error for default parameters"
    env.render()


def test_render_svg(env, tmpdir):
    "Verify that render() executes without error for svg"
    name = "test_img.svg"
    img = tmpdir.mkdir("imgs").join(name)
    env.render(mode="svg", filename=img)


# TODO: convert test_step() into a class, add methods to test step() with different moves from different initial states
def test_step(env):
    "Test step() with a variety of moves from different initial states"
    action = FanoronaMove(
        Position("D2"), Direction(8), 1, False
    )  # D2 -> E3 approach capture
    _, reward, done, _ = env.step(action)
    assert (
        env.state.get_board_str()
        == "WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB B - - 1"
    )  # turn automatically skips due to no available moves
    assert reward == 0
    assert not done
    action = FanoronaMove(
        Position("A1"), Direction(0), 0, True
    )  # end turn action can be of the form (X, X, X, 1)
    assert not action.is_valid(env.state)


def test_game(env, tmpdir):
    "Test environment with a whole game and svg rendering"
    action_list = [
        "D2E310",
        "G4G520",
        "G5F410",
        "A1A101",
        "E2E310",
        "E3D220",
        "D4E310",
        "E3E210",
        "E2F220",
        "D3E310",
        "E3D420",
        "D4E420",
        "C3D320",
        "D3D210",
        "D2E320",
        "E3E220",
        "A1A210",
        "H3G320",
        "G3F420",
        "I2I310",
        "H4G520",
        "F1G100",
        "E2F200",
        "B1C100",
        "F2E320",
        "E3D210",
        "H1G100",
        "D2C300",
        "G1F100",
        "F4G300",
        "F1E100",
        "G3F210",
        "A2A100",
        "C3B210",
    ]
    actions = [FanoronaMove.get_action(action_string) for action_string in action_list]
    ctr = 0
    img = tmpdir.mkdir("imgs").join(f"board-{ctr:03d}.svg")
    env.render(mode="svg", filename=img)
    for action in actions:
        assert action.is_valid(env.state)
        env.step(action)
        ctr += 1
        env.render(mode="svg", filename=img)


def test_get_valid_moves(env):
    "Verify that get_valid_moves() returns the correct valid moves from a given state"
    state_str = (
        "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0"  # start state
    )
    env.state = FanoronaState.set_from_board_str(state_str)
    valid_moves = env.get_valid_moves()
    assert len(valid_moves) == 5


def test_random_valid_moves(env):
    "Test random valid moves sampled from action space to verify that they run without error"
    valid_moves = env.get_valid_moves()
    while valid_moves:
        print(valid_moves)
        action = random.choice(valid_moves)
        assert action.is_valid(env.state)
        env.step(action)
        valid_moves = env.get_valid_moves()


@pytest.mark.skip(
    reason="Inexplicably taking too long to find valid moves after two of them"
)
def test_all_moves(env):
    "Test all possible moves by systematically exploring action space and verify that they run without error"
    valid = 0
    while valid < 3:
        for pos in Position.pos_range():
            for dir in range(9):
                for capture_type in range(3):
                    for end_turn in range(2):
                        action = (pos, dir, capture_type, end_turn)
                        valid = env.is_valid(action)
                        if valid:
                            print(env.get_board_str(), action)
                            valid += 1
                        env.step(action)


@pytest.mark.skip(reason="Not implemented")
def test_play_game(env):
    "Test play_game() function"
    pass
