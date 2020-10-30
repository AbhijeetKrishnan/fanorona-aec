import random

import gym
import gym_fanorona
import pytest
from gym_fanorona.agents.minimax_agent import MinimaxAgent
from gym_fanorona.agents.random_agent import RandomAgent


def test_game():
    """Test that creating an env with one MinimaxAgent and one RandomAgent works correctly."""
    white, black = MinimaxAgent(cutoff=2, heuristic=lambda node: random.choice([-1, 1])), RandomAgent()
    env = gym.make('fanorona-v0', white_player=white, black_player=black)
    env.play_game()
