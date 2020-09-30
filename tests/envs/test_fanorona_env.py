import random

import gym
import pytest
from gym_fanorona.envs import (MOVE_LIMIT, Direction, FanoronaMove,
                               FanoronaState, Position)

TEST_STATES = [
    'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0', # start state
    'WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB B - - 1', # capturing seq after D2->E3 approach
    '9/9/3W1B3/9/9 W - - 49', # random endgame state
]

def test_make():
    "Verify that gym.make() executes without error"
    env = gym.make('fanorona-v0')
    env.close()

def test_reset():
    "Verify that reset() executes without error"
    env = gym.make('fanorona-v0')
    env.reset()
    env.close()

def test_reset_starting():
    "Verify that reset() sets the board state to the correct starting position"
    env = gym.make('fanorona-v0')
    env.reset()
    assert env.state.get_board_str() == 'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0' # starting position
    env.close()

def test_render():
    "Verify that render() executes without error for default parameters"
    env = gym.make('fanorona-v0')
    env.reset()
    env.render()
    env.close()

def test_render_svg():
    "Verify that render() executes without error for svg"
    env = gym.make('fanorona-v0')
    env.reset()
    env.render(mode='svg')
    env.close()

# TODO: convert test_step() into a class, add methods to test step() with different moves from different initial states
def test_step():
    "Test step() with a variety of moves from different initial states"
    env = gym.make('fanorona-v0')
    env.reset()
    action = FanoronaMove(Position(12), Direction(8), 1, False) # D2 -> E3 approach capture
    obs, reward, done, info = env.step(action)
    assert env.state.get_board_str() == 'WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB B - - 1' # turn automatically skips due to no available moves
    assert reward == 0
    assert not done
    action = FanoronaMove(Position(0), Direction(0), 0, True) # end turn action can be of the form (X, X, X, 1)
    assert not action.is_valid(env.state)
    env.close()

def test_get_valid_moves():
    "Verify that get_valid_moves() returns the correct valid moves from a given state"
    env = gym.make('fanorona-v0')
    state_str = 'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0' # start state
    env.state = FanoronaState.set_from_board_str(state_str)
    valid_moves = env.get_valid_moves()
    assert len(valid_moves) == 5
    env.close()

def test_random_valid_moves():
    "Test random valid moves sampled from action space to verify that they run without error"
    ITERATIONS = 100
    env = gym.make('fanorona-v0')
    env.reset()
    valid_moves = env.get_valid_moves()
    while valid_moves:
        print(valid_moves)
        action = random.choice(valid_moves) 
        assert action.is_valid(env.state)
        env.step(action)
        valid_moves = env.get_valid_moves()
    env.close()

@pytest.mark.skip(reason='Inexplicably taking too long to find valid moves after two of them')
def test_all_moves():
    "Test all possible moves by systematically exploring action space and verify that they run without error"
    env = gym.make('fanorona-v0')
    env.reset()
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
    env.close()
