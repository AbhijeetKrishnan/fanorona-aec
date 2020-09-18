from enum import IntEnum
from typing import List, Dict

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

DIR_STRINGS = ['SW', 'S', 'SE', 'W', '-', 'E', 'NW', 'N', 'NE']

class Reward(IntEnum):
    PAIKA = 0
    END_TURN = 0
    ILLEGAL_MOVE = -1
    LOSS = -1
    WIN = 1
    DRAW = 0
    NONE = 0

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
            Discrete(9): direction
            Discrete(3): capture type (none, approach, withdrawal)
            Discrete(2): end turn
        )
    
    Reward:
        +1: win
         0: draw
        -1: loss

    Starting State:
        Starting board setup for Fanorona (see https://en.wikipedia.org/wiki/Fanorona#/media/File:Fanorona-1.svg)

    Episode Termination:
        Game ends in a win, draw or loss
    """

    metadata = {'render.modes': ['human']}

    def __init__(self: FanoronaEnv) -> None:

        super(FanoronaEnv, self).__init__()
        # would dynamically changing action space be better than current implementation?
        self.action_space = spaces.Tuple((
            spaces.Discrete(NUM_SQUARES),    # from
            spaces.Discrete(len(Direction)), # direction 
            spaces.Discrete(3)               # capture type (none=0, approach=1, withdrawal=2)
            spaces.Discrete(2)               # end turn (0 for no, 1 for yes)
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=2, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8), # board state: (9 x 5) x Piece
            spaces.Discrete(2),                                                       # turn to play: (WHITE, BLACK)
            spaces.Discrete(len(Direction)),                                          # last direction used: Direction 
            spaces.Box(low=0, high=1, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8), # positions used: (9 x 5) x (True, False)
            spaces.Discrete(MOVE_LIMIT + 1)                                           # number of half-moves
        ))

        # self.seed()
        self.state = None

    @staticmethod
    def pos_to_coords(position: int) -> Tuple[int, int]:
        """Converts an integer board coordinate into (row, col) format."""
        return (position // BOARD_COLS, position % BOARD_COLS)

    @staticmethod
    def coords_to_pos(coords: Tuple[int, int]) -> int:
        """Converts (row, col) tuple to integer board coordinate."""
        row, col = coords
        return row * (BOARD_COLS) + col

    @staticmethod
    def displace_pos(pos: int, dir: Direction) -> int:
        """Adds unit direction vector (given by dir) to pos."""
        DIR_VALS = {
            0: (-1, -1), # SW
            1: (-1,  0), # S
            2: (-1,  1), # SE
            3: (0,  -1), # W
            4: (0,   0), # -
            5: (0,   1), # E
            6: (1,  -1), # NW
            7: (1,   0), # N
            8: (1,   1)  # NE
        }
        res_row, res_col = FanoronaEnv.pos_to_coords(pos)
        mod_row, mod_col = DIR_VALS[_dir]
        res = (res_row + mod_row, res_col + mod_col)
        return FanoronaEnv.coords_to_pos(res)

    def seed(self, seed=None) -> List[float]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_piece(self: FanoronaEnv, position: int) -> Piece:
        """Return type of piece at given position (specified in integer coordinates)."""
        _board_state, _, _, _, _ = self.state
        row, col = pos_to_coords(position)
        return Piece(_board_state[row][col])

    def other_side(self: FanoronaEnv) -> Piece:
        """Return the color of the opponent's pieces."""
        _, _who_to_play, _, _, _ = self.state
        if _who_to_play == Piece.WHITE:
            return Piece.BLACK
        else:
            return Piece.WHITE

    @staticmethod
    def get_valid_dirs(pos: int) -> List[Direction]:
        """Get list of valid directions available from a given board position."""
        row, col = FanoronaEnv.pos_to_coords(pos)
        if row == 0 and col == 0: # bottom-left corner
            return [Direction.N, Direction.NE, Direction.E]
        elif row == 2 and col == 0: # middle-left
            return [Direction.S, Direction.SE, Direction.E, Direction.NE, Direction.N]
        elif row == 4 and col == 0: # top-left corner
            return [Direction.S, Direction.SE, Direction.E]
        elif col = 0: # left edge
            return [Direction.S, Direction.E, Direction.N]
        elif row == 0 and col % 2 == 1: # bottom edge 1
            return [Direction.W, Direction.N, Direction.E]
        elif row == 0 and col % 2 == 0: # bottom edge 2
            return [Direction.W, Direction.NW, Direction.N, Direction.NE, Direction.E]
        elif row == 4 and col % 2 == 1: # top edge 1
            return [Direction.W, Direction.S, Direction.E]
        elif row == 4 and col % 2 == 0: # top edge 2
            return [Direction.W, Direction.SW, Direction.S, Direction.SE, Direction.E]
        elif row == 0 and col == 8: # bottom-right corner
            return [Direction.W, Direction.NW, Direction.N]
        elif row == 2 and col == 8: # middle-right
            return [Direction.S, Direction.SW, Direction.W, Direction.NW, Direction.N]
        elif row == 4 and col == 8: # top-right corner
            return [Direction.S, Direction.SW, Direction.W]
        elif col == 8: # right edge
            return [Direction.S, Direction.W Direction.N]
        elif (row + col) % 2 == 0: # 8-point
            return [Direction.S, Direction.SW, Direction.W, Direction.NW, Direction.N, Direction.NE, Direction.E, Direction.SE]
        elif (row + col) % 2 == 1: # 4-point
            return [Direction.S, Direction.W, Direction.N, Direction.E]

    def in_capturing_seq(self):
        """Returns True if current state is part of a capturing sequence i.e. at least one capture has already been made."""
        _, _, last_dir_used, _, _ = self.state
        return last_dir_used != Direction.X:

    def capture_exists(self):
        """Returns True if capturing move exists in the current state."""
        _, _who_to_play, _, _, _ = self.state

        # Capturing move exists if -
        # a) a piece belonging to the side to play exists
        # b) it has an adjacent empty space
        # c) the opposite color piece exists on approach or withdrawal.
        for pos in range(NUM_SQUARES):
            if self.get_piece(pos) == Piece(_who_to_play):
                for dir in FanoronaEnv.get_valid_dirs(pos):
                    if self.get_piece(FanoronaEnv.displace_pos(_to, dir)) == self.other_side() # approach
                        return True
                    elif self.get_piece(FanoronaEnv.displace_pos(_from, 8 - dir)) == self.other_side # withdrawal
                        return True
        return False

    def is_valid(self: FanoronaEnv, action) -> bool:
        _from, _dir, _capture_type, _end_turn = action
        _board_state, _who_to_play, _last_dir, _visited_pos, _half_moves = self.state

        _to = FanoronaEnv.displace_pos(_from, _dir)
        if _capture_type == 0:   # none
            _capture = NUM_SQUARES
        elif _capture_type == 1: # approach
            _capture = FanoronaEnv.displace_pos(_to, _dir)
        else:                    # withdraw
            _capture = FanoronaEnv.displace_pos(_from, 8 - _dir)

        # Check that move number is under limit
        if _half_moves >= MOVE_LIMIT:
            return False

        # End turn must be done during a capturing sequence, indicated by last_dir not being Direction.X
        if _end_turn and not self.in_capturing_seq():
            return False

        # Bounds checking on positions
        for pos in (_from, _to):
            if not 0 <= pos < NUM_SQUARES: # pos is within board bounds
                return False
        if not 0 <= _capture <= NUM_SQUARES: # capture may be NUM_SQUARES in case of paika move
            return False 

        # Checking validity of pieces at action positions
        if self.get_piece(_from) != _who_to_play: # from position must contain a piece 
            return False
        if self.get_piece(_to) != Piece.EMPTY: # piece must be played to an empty location
            return False
        if _capture != NUM_SQUARES and self.get_piece(_capture) != self.other_side(): # capturing line must start with opponent color stone
            return False

        # Checking that dir is permitted from given board position
        if dir not in FanoronaEnv.get_valid_dirs(_from):
            return False

        # Check if paika is being played when capturing move exists, which is illegal
        if not self.in_capturing_seq() and self.capture_exists():
            return False
        
        # TODO: Check that capturing piece is not visiting previously visited pos in capturing path

        # TODO: Check that capturing piece is not moving twice in the same direction

    def piece_exists(self, piece):
        """Checks whether a instance of a piece exists on the game board."""
        for pos in range(NUM_SQUARES):
            if self.get_piece(pos) == piece:
                return True
        return False

    def is_done(self):
        """
        Check whether the game is over and return the reward.

        The game is over when -
        a) One side has no pieces left to move (loss/win)
        b) The number of half-moves exceeds the limit (draw)
        """
        _board_state, _who_to_play, _last_dir, _visited_pos, _half_moves = self.state
        if _half_moves >= MOVE_LIMIT:
            return True, Reward.DRAW
        else:
            own_piece_exists = self.piece_exists(_who_to_play)
            other_piece_exists = self.piece_exists(self.other_side())
            if not own_piece_exists: # TODO: actually should be if a move exists, since a piece may exist but not have legal moves
                return True, Reward.LOSS
            if not other_piece_exists:
                return True, Reward.WIN
            else:
                return False, Reward.NONE

    @staticmethod
    def reset_visited_pos(visited_pos):
        """Resets visited_pos of a state to indicate no visited positions."""
        for pos in range(NUM_SQUARES):
            row, col = FanoronaEnv.pos_to_coords(pos)
            visited_pos[row][col] = 0

    def step(self, action):
        """Compute return values based on action taken."""

        if self.is_valid(action):
            _from, _dir, _capture_type, _end_turn = action
            _board_state, _who_to_play, _last_dir, _visited_pos, _half_moves = self.state
            _to = FanoronaEnv.displace_pos(_from, _dir)
            _from_row, _from_col = FanoronaEnv.pos_to_coords(_from)
            _to_row, _to_col = FanoronaEnv.pos_to_coords(_to)

            if capture_type == 0: # paika move
                _board_state[_to_row][_to_col] = self.get_piece(_from)
                _board_state[_from_row][_from_col] = int(Piece.EMPTY)
                _who_to_play = FanoronaEnv.other_side()
                _last_dir = Direction.X
                FanoronaEnv.reset_visited_pos(_visited_pos)
                _half_moves += 1
                self.state = (_board_state, _who_to_play, _last_dir, _visited_pos, _half_moves)

            elif _end_turn: # end turn
                _who_to_play = FanoronaEnv.other_side()
                _last_dir = Direction.X
                FanoronaEnv.reset_visited_pos(_visited_pos)
                _half_moves += 1
                self.state = (_board_state, _who_to_play, _last_dir, _visited_pos, _half_moves)
            
            else: # capture (approach, withdrawal)
                _to = FanoronaEnv.displace_pos(_from, _dir)
                if _capture_type == 1: # approach
                    _capture = FanoronaEnv.displace_pos(_to, _dir)
                    _capture_dir = _dir
                else:                  # withdraw
                    _capture = FanoronaEnv.displace_pos(_from, 8 - _dir)
                    _capture_dir = 8 - _dir
                _capture_row, _capture_col = FanoronaEnv.pos_to_coords(_capture)
                _mod_row, _mod_col = _capture_dir
                while 0 <= _capture_row < BOARD_ROWS and 0 <= _capture_col < BOARD_COLS:
                    if _board_state[_capture_row][_capture_col] == self.other_side():
                        _board_state[_capture_row][_capture_col] = Piece.EMPTY
                        _capture_row += _mod_row
                        _capture_col += _mod_col
                    else:
                        break
                
                _last_dir = _dir
                _visited_pos[_from_row][_from_col] = 1
                _visited_pos[_to_row][_to_col] = 1

        else: # invalid move
            self.state = (_board_state, _who_to_play, _last_dir, _visited_pos, _half_moves)
            obs = self.state
            done, reward = self.is_done()
            reward = Reward.ILLEGAL_MOVE
            info = {}
            return obs, reward, done, info

        self.state = (_board_state, _who_to_play, _last_dir, _visited_pos, _half_moves)
        obs = self.state
        done, reward = self.is_done()
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.state = (
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
        )

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
