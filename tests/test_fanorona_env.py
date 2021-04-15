import random

import gym
import pytest
from gym_fanorona.envs import (
    Direction,
    FanoronaMove,
    FanoronaState,
    Position,
)
from gym_fanorona.agents import RandomAgent

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
    assert reward == 2
    assert not done
    action = FanoronaMove(
        Position("A1"), Direction(0), 0, True
    )  # end turn action can be of the form (X, X, X, 1)
    assert not action.is_valid(env.state)


def test_game(env, tmpdir):
    "Test environment with a whole game and svg rendering"
    action_list = [
        "E2710",
        "F4620",
        "E5110",
        "G2710",
        "F3310",
        "A1001",
        "D2710",
        "E3620",
        "D4720",
        "E1610",
        "H4620",
        "A1001",
        "G3710",
        "A1001",
        "E4510",
        "A1001",
        "C2710",
        "F4010",
        "E3310",
        "A1001",
        "H1710",
        "D3110",
        "D2310",
        "A1710",
        "A1001",
        "I4110",
        "H2020",
        "C2310",
        "B2720",
        "F1700",
        "B3300",
        "F2500",
        "I5000",
        "G1800",
        "H4110",
        "G2300",
        "D5100",
        "F2610",
        "H5300",
        "E3810",
        "A1001",
        "B5100",
        "F4120",
        "A1001",
        "H3310",
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


def test_play_game(env):
    "Verify that play_game() works without error"
    env.set_white_player(RandomAgent())
    env.set_black_player(RandomAgent())
    move_list = env.play_game()
    print([str(move) for move in move_list])
