from enum import IntEnum

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

BOARD_ROWS  = 5
BOARD_COLS = 9
MOVE_LIMIT = 50

NUM_SQUARES  = BOARD_ROWS * BOARD_COLS

class Piece(IntEnum):
    WHITE = 0
    BLACK = 1
    EMPTY = 2

PIECE_STRINGS = ['W', 'B', 'E']

class Direction(IntEnum):
    SW = 0
    S  = 1
    SE = 2
    W  = 3
    X  = 4 # No direction
    E  = 5
    NW = 6
    N  = 7
    NE = 8 

DIR_STRINGS = ['SW', 'S', 'SW', 'W', '-', 'E', 'NW', 'N', 'NE']

class FanoronaEnv(gym.Env):
    """
    Description:
        Implements the Fanorona board game following the 5x9 Fanoron Tsivy
        variation. A draw is declared if 50 half-moves have been exceeded 
        since the start of the game. Consecutive captures count as one move.

    References: 
        https://www.mindsports.nl/index.php/the-pit/528-fanorona
        https://en.wikipedia.org/wiki/Fanorona

    Observation:
        Type: Tuple(
            Box(low=0, high=2, (5, 9)): board state (board x Piece)
            Discrete(2): turn to play (White/Black)
            Discrete(9): last direction moved (Direction)
            Box(low=0, high=1, (5, 9)): positions used (board state x Boolean)
            Discrete(MOVE_LIMIT + 1): number of half-moves since start of game
        )

    Action:
        Type: Tuple(
            Discrete(45): from
            Discrete(45): to
            Discrete(45 + 1): start of capturing line (+ 1 in case of paika)
        )
    
    Reward:
        +1: win
         0: draw
        -1: loss

    Starting State:
        Starting board position for Fanorona (see https://en.wikipedia.org/wiki/Fanorona#/media/File:Fanorona-1.svg)

    Episode Termination:
        Game ends in a win, draw or loss
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, side=Piece.WHITE):
        """
        side: Which side to play as (White/Black)
        """
        super(FanoronaEnv, self).__init__()
        self.action_space = spaces.Tuple((
            spaces.Discrete(NUM_SQUARES),    # from: Position
            spaces.Discrete(NUM_SQUARES),    # to: Position
            spaces.Discrete(NUM_SQUARES + 1) # start of capturing line (+ 1 for paika move): Position
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=2, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8), # board state: (9 x 5) x Piece
            spaces.Discrete(2),                                                       # turn to play: (WHITE, BLACK)
            spaces.Discrete(len(Direction)),                                          # last direction used: Direction 
            spaces.Box(low=0, high=1, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8), # positions used: (9 x 5) x (True, False)
            spaces.Discrete(MOVE_LIMIT + 1)                                           # number of half-moves
        ))

        self.seed()
        self.state = None
        self.side = side

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        error_msg = f"{action} ({type(action)}) invalid"
        assert self.action_space.contains(action), error_msg
        # TODO: compute return values based on action taken

        done = True
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.state = tuple((
            np.array(
                [
                    [int(Piece.WHITE)] * BOARD_COLS,
                    [int(Piece.WHITE)] * BOARD_COLS,
                    [int(Piece.BLACK), int(Piece.WHITE), int(Piece.BLACK), int(Piece.WHITE), int(Piece.EMPTY), int(Piece.BLACK), int(Piece.WHITE), int(Piece.BLACK), int(Piece.WHITE)],
                    [int(Piece.BLACK)] * BOARD_COLS,
                    [int(Piece.BLACK)] * BOARD_COLS,
                    
                ],
                dtype=np.int8,

            ),
            int(Piece.WHITE),
            int(Direction.X),
            np.array(
                [
                    [0] * BOARD_COLS,
                    [0] * BOARD_COLS,
                    [0] * BOARD_COLS,
                    [0] * BOARD_COLS,
                    [0] * BOARD_COLS,
                ],
                dtype=np.int8,

            ),
        ))

        return self.state

    def get_board_string(self):
        """     
        ●─●─●─●─●─●─●─●─●
        │╲│╱│╲│╱│╲│╱│╲│╱│
        ●─●─●─●─●─●─●─●─●
        │╱│╲│╱│╲│╱│╲│╱│╲│
        ●─○─●─○─┼─●─○─●─○  
        │╲│╱│╲│╱│╲│╱│╲│╱│
        ○─○─○─○─○─○─○─○─○
        │╱│╲│╱│╲│╱│╲│╱│╲│
        ○─○─○─○─○─○─○─○─○
        """
        _board_state, _who_to_play, _last_dir, _visited_pos = self.state

        board_string = ''
        count = 0
        for row in _board_state:
            for col in row:
                if col == Piece.EMPTY:
                    count += 1
                else:
                    if count > 0:
                        board_string += str(count)
                        count = 0
                    board_string += PIECE_STRINGS[col]
            if count > 0:
                board_string += str(count)
            board_string += '/'
        board_string = board_string.rstrip('/')
        if count > 0:
                board_string += str(count)

        who_to_play = PIECE_STRINGS[_who_to_play]
        last_dir = DIR_STRINGS[_last_dir]
        
        visited_pos = []
        for row_idx, row in enumerate(_visited_pos):
            for col_idx, col in enumerate(row):
                if col:
                    visited_pos.append(f'{chr(65 + col_idx)}{row_idx}')
        visited_pos = ','.join(visited_pos)
        if visited_pos == '':
            visited_pos = '-'

        return ' '.join([board_string, who_to_play, last_dir, visited_pos])

    def render(self, mode='human', close=False):
        print(self.get_board_string())
