import pytest

import gym
import gym_fanorona

def test_make():
    env = gym.make('fanorona-v0')

def test_reset():
    env = gym.make('fanorona-v0')
    env.reset()

def test_get_board_string_starting():
    env = gym.make('fanorona-v0')
    env.reset()
    assert env.get_board_string() == 'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - -' # starting position

def test_render():
    env = gym.make('fanorona-v0')
    env.reset()
    env.render()