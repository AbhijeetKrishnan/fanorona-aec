import random

import gym
import gym_fanorona
import pytest
from gym_fanorona.envs.fanorona_env import FanoronaEnv

TEST_STATES = [
    'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0', # start state
    'WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB W NE D2,E3 0', # capturing seq after D2->E3 approach
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
    assert env.get_board_str() == 'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0' # starting position
    env.close()

def test_render():
    "Verify that render() executes without error"
    env = gym.make('fanorona-v0')
    env.reset()
    env.render()
    env.close()

def test_is_valid():
    "Test 2 opening moves for validity"
    env = gym.make('fanorona-v0')
    env.reset()
    action1 = (12, 8, 1, 0)
    action2 = (24, 6, 1, 0) # cannot move different piece in capturing sequence - illegal move
    assert env.is_valid(action1)
    env.step(action1)
    assert not env.is_valid(action2)
    env.close()

# TODO: convert test_step() into a class, add methods to test step() with different moves from different initial states
def test_step():
    "Test step() with a variety of moves from different initial states"
    env = gym.make('fanorona-v0')
    env.reset()
    action = (12, 8, 1, 0) # D2 -> E3 approach capture
    obs, reward, done, info = env.step(action)
    assert env.get_board_str() == 'WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB W NE D2,E3 0'
    assert reward == 0
    assert not done
    action = (0, 0, 0, 1) # end turn action can be of the form (X, X, X, 1)
    assert env.is_valid(action)
    obs, reward, done, info = env.step(action)
    assert env.get_board_str() == 'WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB B - - 1'
    assert reward == 0
    assert not done
    env.close()

def test_is_done():
    "Test that is_done method is correctly identifying end of game states."
    board_str = '9/9/3W1B3/9/9 W - - 49'
    env = gym.make('fanorona-v0')
    env.set_state_from_board_str(board_str)
    action = (21, 5, 1, 0)
    obs, reward, done, info = env.step(action)
    assert done
    env.close()

def test_end_turn():
    "Test end turn action by making an opening capturing move and testing all possible end turn actions in action space"
    env = gym.make('fanorona-v0')
    env.reset()
    action = (12, 8, 1, 0) # D2 -> E3 approach capture
    env.step(action)
    assert env.get_board_str() == 'WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB W NE D2,E3 0'
    for pos in range(45):
            for dir in range(9):
                for capture_type in range(3):
                    end_turn = 1
                    action = (pos, dir, capture_type, end_turn)
                    assert env.is_valid(action), f'Action: {action}'
    env.close()

def test_get_valid_moves():
    "Verify that get_valid_moves() returns the correct valid moves from a given state"
    env = gym.make('fanorona-v0')
    state = 'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0' # start state
    env.set_state_from_board_str(state)
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
        action = random.choice(valid_moves) 
        assert env.is_valid(action)
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
        for pos in range(45):
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

def test_set_state_from_board_str():
    "Verify that set_state_from_board_str() sets the state correctly"
    env = gym.make('fanorona-v0')
    for board_str in TEST_STATES:
        env.set_state_from_board_str(board_str)
        assert env.get_board_str() == board_str
    env.close()

def test_get_board_str():
    "Test that get_board_str() works with randomly sampled observations"
    env = gym.make('fanorona-v0')
    env.reset()
    for i in range(10):
        env.state = env.observation_space.sample()
        print(env.get_board_str())
    env.close()

def test_capture_exists():
    env = gym.make('fanorona-v0')
    env.reset()
    states = TEST_STATES
    env.set_state_from_board_str(states[0])
    assert env.capture_exists()
    env.set_state_from_board_str(states[1])
    assert not env.capture_exists()
    env.set_state_from_board_str(states[2])
    assert env.capture_exists()

def test_convert_coords_to_human():
    "Test that convert_coords_to_human returns the correct values"
    answer = {
        (0, 0): 'A1', (0, 1): 'B1', (0, 2): 'C1', (0, 3): 'D1', (0, 4): 'E1', (0, 5): 'F1', (0, 6): 'G1', (0, 7): 'H1', (0, 8): 'I1',
        (1, 0): 'A2', (1, 1): 'B2', (1, 2): 'C2', (1, 3): 'D2', (1, 4): 'E2', (1, 5): 'F2', (1, 6): 'G2', (1, 7): 'H2', (1, 8): 'I2',
        (2, 0): 'A3', (2, 1): 'B3', (2, 2): 'C3', (2, 3): 'D3', (2, 4): 'E3', (2, 5): 'F3', (2, 6): 'G3', (2, 7): 'H3', (2, 8): 'I3',
        (3, 0): 'A4', (3, 1): 'B4', (3, 2): 'C4', (3, 3): 'D4', (3, 4): 'E4', (3, 5): 'F4', (3, 6): 'G4', (3, 7): 'H4', (3, 8): 'I4',
        (4, 0): 'A5', (4, 1): 'B5', (4, 2): 'C5', (4, 3): 'D5', (4, 4): 'E5', (4, 5): 'F5', (4, 6): 'G5', (4, 7): 'H5', (4, 8): 'I5',
    }
    for row in range(5):
        for col in range(9):
            assert FanoronaEnv.convert_coords_to_human(row, col) == answer[(row, col)]

def test_convert_human_to_coords():
    "Test that convert_human_to_coords returns the correct values"
    answer = {
        'A1': (0, 0), 'B1': (0, 1), 'C1': (0, 2), 'D1': (0, 3), 'E1': (0, 4), 'F1': (0, 5), 'G1': (0, 6), 'H1': (0, 7), 'I1': (0, 8),
        'A2': (1, 0), 'B2': (1, 1), 'C2': (1, 2), 'D2': (1, 3), 'E2': (1, 4), 'F2': (1, 5), 'G2': (1, 6), 'H2': (1, 7), 'I2': (1, 8),
        'A3': (2, 0), 'B3': (2, 1), 'C3': (2, 2), 'D3': (2, 3), 'E3': (2, 4), 'F3': (2, 5), 'G3': (2, 6), 'H3': (2, 7), 'I3': (2, 8),
        'A4': (3, 0), 'B4': (3, 1), 'C4': (3, 2), 'D4': (3, 3), 'E4': (3, 4), 'F4': (3, 5), 'G4': (3, 6), 'H4': (3, 7), 'I4': (3, 8),
        'A5': (4, 0), 'B5': (4, 1), 'C5': (4, 2), 'D5': (4, 3), 'E5': (4, 4), 'F5': (4, 5), 'G5': (4, 6), 'H5': (4, 7), 'I5': (4, 8),
    }
    for row in range(1, 6):
        for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
            human = col + str(row)
            assert FanoronaEnv.convert_human_to_coords(human) == answer[human]

@pytest.mark.skip(reason='Not implemented')
def test_coords_to_pos():
    "TODO: "
    pass

@pytest.mark.skip(reason='Not implemented')
def test_pos_to_coords():
    "TODO: "
    pass

@pytest.mark.skip(reason='Not implemented')
def test_displace_piece():
    "TODO: "
    pass

@pytest.mark.skip(reason='Not implemented')
def test_get_piece():
    "TODO: "
    pass

@pytest.mark.skip(reason='Not implemented')
def test_get_valid_dirs():
    "TODO: "
    pass

@pytest.mark.skip(reason='Not implemented')
def test_in_capturing_seq():
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

@pytest.mark.skip(reason='Not implemented')
def test_reset_visited_pos():
    "TODO: "
    pass
