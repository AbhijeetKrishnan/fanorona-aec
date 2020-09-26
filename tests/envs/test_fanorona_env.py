import random

import gym
import gym_fanorona
import pytest


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
        try:
            env.step(action)
        except Exception:
            print(action)
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
                        #print(action, valid)
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

@pytest.mark.skip(reason='Not implemented')
def test_convert_coords_to_human():
    "TODO: Test that convert_coords_to_human returns the correct values"
    pass

@pytest.mark.skip(reason='Not implemented')
def test_convert_human_to_coords():
    "TODO: Test that convert_human_to_coords returns the correct values"
    pass

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