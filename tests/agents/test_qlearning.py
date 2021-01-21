import random

import gym
import gym_fanorona
import pytest
from gym_fanorona.agents.qlearning_agent import QlearningAgent
from gym_fanorona.agents.random_agent import RandomAgent


def test_game():
    """Test that creating an env with one QLearningAgent and one RandomAgent works correctly."""
    white, black = (
        QlearningAgent(Ne=5, Rplus=2, alpha=lambda n: 60 / (59 + n), gamma=0.9),
        RandomAgent(),
    )
    env = gym.make("fanorona-v0", white_player=white, black_player=black)
    env.play_game()
