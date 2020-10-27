import pytest

import gym
import gym_fanorona
from gym_fanorona.agents.random_agent import RandomAgent

def test_game():
    """Test that creating an env with two RandomAgents works correctly."""
    white, black = RandomAgent(), RandomAgent(1)
    env = gym.make('fanorona-v0', white_player=white, black_player=black)
    env.play_game()