import random

import gym
from gym_fanorona.envs import FanoronaMove, FanoronaState, Position, Direction
import pytest

TEST_STATES = [
    'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0', # start state
    'WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB B - - 1', # capturing seq after D2->E3 approach
    '9/9/3W1B3/9/9 W - - 49', # random endgame state
]

def test_is_valid():
    "Test 2 opening moves for validity"
    env = gym.make('fanorona-v0')
    env.reset()
    action1 = FanoronaMove(Position('D2'), Direction['NE'], 1, False)
    action2 = FanoronaMove(Position('E1'), Direction['NW'], 1, False) # cannot move different piece in capturing sequence - illegal move
    assert action1.is_valid(env.state)
    env.step(action1)
    assert not action2.is_valid(env.state)
    env.close()

def test_is_done():
    "Test that is_done method is correctly identifying end of game states."
    board_str = '9/9/3W1B3/9/9 W - - 49'
    env = gym.make('fanorona-v0')
    env.state = FanoronaState.set_from_board_str(board_str)
    action = FanoronaMove(Position('D3'), Direction['E'], 1, False)
    assert action.is_valid(env.state)
    _, _, done, _ = env.step(action)
    assert done
    env.close()

def test_end_turn():
    "Test end turn action by making an opening capturing move and testing all possible end turn actions in action space"
    env = gym.make('fanorona-v0')
    env.reset()
    action = FanoronaMove(Position('D2'), Direction(8), 1, False) # D2 -> E3 approach capture
    env.step(action)
    action = FanoronaMove(Position('E5'), Direction['SE'], 1, False) # Black: E5 -> F4 approach capture
    assert(action.is_valid(env.state))
    env.step(action)
    for pos in Position.pos_range():
            for direction in Direction.dir_range():
                for capture_type in range(3):
                    end_turn = True
                    action = FanoronaMove(pos, direction, capture_type, end_turn)
                    assert action.is_valid(env.state), f'Action: {action}'
    env.close()

def test_set_state_from_board_str():
    "Verify that set_state_from_board_str() sets the state correctly"
    env = gym.make('fanorona-v0')
    for board_str in TEST_STATES:
        env.state = FanoronaState.set_from_board_str(board_str)
        assert env.state.get_board_str() == board_str
    env.close()

def test_get_board_str():
    "Test that get_board_str() works with randomly sampled observations without error"
    env = gym.make('fanorona-v0')
    env.reset()
    for _ in range(10):
        env.state = FanoronaState(env.observation_space.sample())
        print(env.state.get_board_str())
    env.close()

def test_capture_exists():
    env = gym.make('fanorona-v0')
    env.reset()
    states = TEST_STATES
    env.state = FanoronaState.set_from_board_str(states[0])
    assert env.state.capture_exists()
    env.state = FanoronaState.set_from_board_str(states[1])
    assert env.state.capture_exists()
    env.state = FanoronaState.set_from_board_str(states[2])
    assert env.state.capture_exists()

def test_in_capturing_seq():
    env = gym.make('fanorona-v0')
    env.reset()
    states = TEST_STATES
    env.state = FanoronaState.set_from_board_str(states[0])
    assert not env.state.in_capturing_seq()
    env.state = FanoronaState.set_from_board_str(states[1])
    assert not env.state.in_capturing_seq()
    env.state = FanoronaState.set_from_board_str(states[2])
    assert not env.state.in_capturing_seq()

def test_reset_visited_pos():
    env = gym.make('fanorona-v0')
    env.reset()
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

@pytest.mark.skip(reason='Not implemented')
def test_get_piece():
    "TODO: "
    pass

@pytest.mark.skip(reason='Not implemented')
def test_other_side():
    "TODO: "
    pass

@pytest.mark.skip(reason='Not implemented')
def test_piece_exists():
    "TODO: "
    pass
