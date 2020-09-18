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
    assert env.get_board_string() == 'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0' # starting position
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

def test_step():
    env = gym.make('fanorona-v0')
    env.reset()
    action = (12, 8, 1, 0) # D2 -> E3 approach capture
    obs, reward, done, info = env.step(action)
    assert env.get_board_string() == 'WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB W NE D2,E3 0'
    assert reward == 0
    assert not done
    env.close()