from enum import Enum

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

BOARD_WIDTH  = 5 # number of rows
BOARD_HEIGHT = 9 # number of columns

NUM_SQUARES  = BOARD_WIDTH * BOARD_HEIGHT

class Piece(Enum):
    WHITE = 0
    BLACK = 1
    EMPTY = 2

class Direction(Enum):
    SW = 0
    S  = 1
    SE = 2
    W  = 3
    X  = 4 # No direction
    E  = 5
    NW = 6
    N  = 7
    NE = 8 

class FanoronaEnv(gym.Env):
    """
    Description:
        Implements the Fanorona board game following the 5x9 Fanoron Tsivy
        variation. A draw is declared if 50 half-moves have been exceeded 
        since the start of the game. Consecutive captures count as one move

    References: 
        https://www.mindsports.nl/index.php/the-pit/528-fanorona
        https://en.wikipedia.org/wiki/Fanorona

    Observation:
        Type: Tuple(
            Box(low=0, high=2, (5, 9)): board state (board x Piece)
            Discrete(2): turn to play (White/Black)
            Discrete(9): last direction moved (Direction)
            Box(low=0, high=1, (5, 9)): positions used (board state x Boolean)
            Box(low=0, high=50): number of half-moves since start of game
        )

    Action:
        Type: Tuple(
            Discrete(45): from
            Discrete(45): to
            Discrete(45 + 1): start of capturing line (+ 1 in case of paika)
        )
    
    Reward:
        +1: win
        0 : draw
        -1: loss

    Starting State:
        Starting board position for Fanorona (see https://en.wikipedia.org/wiki/Fanorona#/media/File:Fanorona-1.svg)

    Episode Termination:
        Game ends in a win, draw or loss
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, side=Piece.WHITE):
        super(FanoronaEnv, self).__init__()
        self.action_space = spaces.Tuple((
            spaces.Discrete(NUM_SQUARES),    # from: Position
            spaces.Discrete(NUM_SQUARES),    # to: Position
            spaces.Discrete(NUM_SQUARES + 1) # start of capturing line (+ 1 for paika move): Position
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=2, shape=(BOARD_WIDTH, BOARD_HEIGHT), dtype=np.int8), # board state: (9 x 5) x Piece
            spaces.Discrete(2),                                                          # turn to play: (WHITE, BLACK)
            spaces.Discrete(len(Direction)),                                             # last direction used: Direction 
            spaces.Box(low=0, high=1, shape=(BOARD_WIDTH, BOARD_HEIGHT), dtype=np.int8), # positions used: (9 x 5) x (True, False)
            spaces.Box(low=0, high=50 dtype=np.int8)                                     # number of half-moves
        ))

        self.side = side

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        error_msg = f"{action} ({type(action)}) invalid"
        assert self.action_space.contains(action), error_msg
        # compute return values based on action taken
        done = True
        info = {}
        return observation, reward, done, info

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass
