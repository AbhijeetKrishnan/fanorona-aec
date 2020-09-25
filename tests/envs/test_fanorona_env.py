import pytest

import gym
import gym_fanorona

def test_make():
    env = gym.make('fanorona-v0')
    env.close()

def test_reset():
    env = gym.make('fanorona-v0')
    env.reset()
    env.close()

def test_get_board_string_starting():
    env = gym.make('fanorona-v0')
    env.reset()
    assert env.get_board_str() == 'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0' # starting position
    env.close()

def test_render():
    env = gym.make('fanorona-v0')
    env.reset()
    env.render()
    env.close()

def test_is_valid():
    env = gym.make('fanorona-v0')
    env.reset()
    action = (12, 8, 1, 0)
    assert env.is_valid(action)

def test_is_valid2():
    "Test 2 opening moves for validity"
    env = gym.make('fanorona-v0')
    env.reset()
    action1 = (12, 8, 1, 0)
    action2 = (24, 6, 1, 0) # cannot move different piece in capturing sequence - illegal move
    assert env.is_valid(action1)
    env.step(action1)
    assert not env.is_valid(action2)

def test_step():
    "Test simple opening move."
    env = gym.make('fanorona-v0')
    env.reset()
    action = (12, 8, 1, 0) # D2 -> E3 approach capture
    obs, reward, done, info = env.step(action)
    assert env.get_board_str() == 'WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB W NE D2,E3 0'
    assert reward == 0
    assert not done
    env.close()

def test_step2():
    "Test end_turn after opening move."
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


@pytest.mark.skip(reason='Too many invalid moves sampled compared to valid moves')
def test_random_moves():
    "Test random moves sampled from action space"
    env = gym.make('fanorona-v0')
    env.reset()
    valid, invalid = 0, 0
    while True:
        action = env.action_space.sample()
        if env.is_valid(action):
            valid += 1
            print('Valid')
        else:
            invalid += 1
            print('Invalid')
        try:
            env.step(action)
        except Exception:
            print(action)
        if valid >= 100:
            break
    assert valid == 100
    env.close()

@pytest.mark.skip(reason='Inexplicably taking too long to find valid moves after two of them')
def test_all_moves():
    "Test all possible moves by systematically exploring action space"
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

def test_set_state_from_board_str():
    board_str = '9/9/3W1B3/9/9 W - - 49'
    env = gym.make('fanorona-v0')
    env.set_state_from_board_str(board_str)
    assert env.get_board_str() == board_str

def test_random_obs():
    env = gym.make('fanorona-v0')
    env.reset()
    for i in range(10):
        env.state = env.observation_space.sample()
        print(env.get_board_str())