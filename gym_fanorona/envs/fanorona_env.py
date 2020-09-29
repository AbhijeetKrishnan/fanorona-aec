from enum import IntEnum
from typing import Dict, List, Tuple

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

BOARD_ROWS  = 5
BOARD_COLS = 9
MOVE_LIMIT = 50

BOARD_SQUARES  = BOARD_ROWS * BOARD_COLS

class Piece(IntEnum):
    def __str__(self):
        return self.name[0] # just the first letter

    WHITE = 0
    BLACK = 1
    EMPTY = 2

class Direction(IntEnum):
    def __str__(self):
        if self.value == 4:
            return '-'
        else:
            return self.name
    SW = 0
    S  = 1
    SE = 2
    W  = 3
    X  = 4 # No direction
    E  = 5
    NW = 6
    N  = 7
    NE = 8 

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

    def __init__(self) -> None:

        super(FanoronaEnv, self).__init__()
        # TODO: would dynamically changing action space be better than current implementation?
        # if no valid moves, turn gets passed e.g. if capturing sequence cannot be continued - should not have to use end_turn action
        self.action_space = spaces.Tuple((
            spaces.Discrete(BOARD_SQUARES),    # from
            spaces.Discrete(len(Direction)), # direction 
            spaces.Discrete(3),              # capture type (none=0, approach=1, withdrawal=2)
            spaces.Discrete(2)               # end turn (0 for no, 1 for yes) 
            # TODO: use capture_type to indicate end_turn action as well (reduces number of states from 2430 to 1620)
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=2, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8), # board state: (9 x 5) x Piece
            spaces.Discrete(2),                                                       # turn to play: (WHITE, BLACK)
            spaces.Discrete(len(Direction)),                                          # last direction used: Direction 
            spaces.Box(low=0, high=1, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8), # positions used: (9 x 5) x (True, False)
            spaces.Discrete(MOVE_LIMIT + 1)                                           # number of half-moves
        ))

        self.state = None

    # TODO: write single translate_coords method to interconvert between human, pos, and coords
    # TODO: create state object with attributes, instead of having to unpack it every time
    @staticmethod
    def convert_human_to_coords(human_coords: str) -> Tuple[int, int]:
        """Converts human-readable board coordinates (e.g. 'A7', 'G1') to integer board coordinates."""
        col_str, row_str = list(human_coords)
        row = int(row_str) - 1
        col = ord(col_str) - ord('A')
        return row, col

    @staticmethod
    def convert_coords_to_human(coords: Tuple) -> str:
        """Converts integer board coordinates to human-readable board coordinates (e.g. 'A7', 'G1')."""
        row, col = coords
        return f'{chr(65 + col)}{row + 1}'

    @staticmethod
    def convert_pos_to_coords(position: int) -> Tuple[int, int]:
        """Converts an integer board coordinate into (row, col) format."""
        # assert 0 <= position < BOARD_SQUARES, f'Invalid position is {position}'
        return (position // BOARD_COLS, position % BOARD_COLS)

    @staticmethod
    def convert_coords_to_pos(coords: Tuple[int, int]) -> int:
        """Converts (row, col) tuple to integer board coordinate."""
        row, col = coords
        # assert 0 <= row < BOARD_ROWS, f'Invalid row is {row}'
        # assert 0 <= col < BOARD_COLS, f'Invalid column is {col}'
        return row * (BOARD_COLS) + col

    @staticmethod
    def displace_pos(pos: int, _dir: Direction) -> int:
        """Adds unit direction vector (given by _dir) to pos."""
        # assert 0 <= pos < BOARD_SQUARES, f'Invalid position is {pos}'
        DIR_VALS = {
            0: (-1, -1), # SW
            1: (-1,  0), # S
            2: (-1,  1), # SE
            3: ( 0, -1), # W
            4: ( 0,  0), # -
            5: ( 0,  1), # E
            6: ( 1, -1), # NW
            7: ( 1,  0), # N
            8: ( 1,  1)  # NE
        }
        res_row, res_col = FanoronaEnv.convert_pos_to_coords(pos)
        mod_row, mod_col = DIR_VALS[_dir]
        res = (res_row + mod_row, res_col + mod_col)
        return FanoronaEnv.convert_coords_to_pos(res)

    def get_piece(self, position: int) -> Piece:
        """Return type of piece at given position (specified in integer coordinates)."""
        assert 0 <= position < BOARD_SQUARES
        _board_state, _, _, _, _ = self.state
        row, col = FanoronaEnv.convert_pos_to_coords(position)
        return Piece(_board_state[row][col])

    def other_side(self) -> Piece:
        """Return the color of the opponent's pieces."""
        _, _who_to_play, _, _, _ = self.state
        if _who_to_play == Piece.WHITE:
            return Piece.BLACK
        else:
            return Piece.WHITE

    @staticmethod
    def get_valid_dirs(pos: int) -> List[Direction]:
        """Get list of valid directions available from a given board position."""
        assert 0 <= pos < BOARD_SQUARES
        row, col = FanoronaEnv.convert_pos_to_coords(pos)
        if row == 0 and col == 0: # bottom-left corner
            dir_list = [Direction.N, Direction.NE, Direction.E]
        elif row == 2 and col == 0: # middle-left
            dir_list = [Direction.S, Direction.SE, Direction.E, Direction.NE, Direction.N]
        elif row == 4 and col == 0: # top-left corner
            dir_list = [Direction.S, Direction.SE, Direction.E]
        elif row == 0 and col == 8: # bottom-right corner
            dir_list = [Direction.W, Direction.NW, Direction.N]
        elif row == 2 and col == 8: # middle-right
            dir_list = [Direction.S, Direction.SW, Direction.W, Direction.NW, Direction.N]
        elif row == 4 and col == 8: # top-right corner
            dir_list = [Direction.S, Direction.SW, Direction.W]
        elif row == 0 and col % 2 == 1: # bottom edge 1
            dir_list = [Direction.W, Direction.N, Direction.E]
        elif row == 0 and col % 2 == 0: # bottom edge 2
            dir_list = [Direction.W, Direction.NW, Direction.N, Direction.NE, Direction.E]
        elif row == 4 and col % 2 == 1: # top edge 1
            dir_list = [Direction.W, Direction.S, Direction.E]
        elif row == 4 and col % 2 == 0: # top edge 2
            dir_list = [Direction.W, Direction.SW, Direction.S, Direction.SE, Direction.E]
        elif col == 0: # left edge
            dir_list = [Direction.S, Direction.E, Direction.N]
        elif col == 8: # right edge
            dir_list = [Direction.S, Direction.W, Direction.N]
        elif (row + col) % 2 == 0: # 8-point
            dir_list = [Direction.S, Direction.SW, Direction.W, Direction.NW, Direction.N, Direction.NE, Direction.E, Direction.SE]
        elif (row + col) % 2 == 1: # 4-point
            dir_list = [Direction.S, Direction.W, Direction.N, Direction.E]
        return dir_list

    def in_capturing_seq(self) -> bool:
        """Returns True if current state is part of a capturing sequence i.e. at least one capture has already been made."""
        _, _, last_dir_used, _, _ = self.state
        return bool(last_dir_used != Direction.X)

    def capture_exists(self) -> bool:
        """
        Returns True if any capturing move exists in the current state.

        A capture exists if -
        1. a piece belonging to the side to play exists
        2. the action of moving the piece in any valid direction in any capture type is also valid 
           (ignoring the no paika when capture exists rule)
        """
        _, _who_to_play, _, _, _ = self.state
        for pos in range(BOARD_SQUARES):
            if self.get_piece(pos) == _who_to_play:
                valid_dirs = self.get_valid_dirs(pos)
                for _dir in valid_dirs:
                    for capture_type in [1, 2]:
                        capture_action = (pos, _dir, capture_type, 0)
                        if self.is_valid(capture_action, skip=['check_no_paika_when_captured']):
                            return True
        return False

    def is_valid(self, action, skip: List[str] = []) -> bool:
        _from, _dir, _capture_type, _end_turn = action
        _board_state, _who_to_play, _last_dir, _visited_pos, _half_moves = self.state

        _to = FanoronaEnv.displace_pos(_from, _dir)
        if _capture_type == 0:   # none
            _capture = BOARD_SQUARES
        elif _capture_type == 1: # approach
            _capture = FanoronaEnv.displace_pos(_to, _dir)
        else:                    # withdraw
            _capture = FanoronaEnv.displace_pos(_from, 8 - _dir)

        def half_move_rule() -> bool:
            """Check that move number is under limit. An action cannot be performed if limit has been reached."""
            return _half_moves < MOVE_LIMIT

        def end_turn_rule():
            """
            End turn must be done during a capturing sequence, indicated by last_dir not being 
            Direction.X. Ignore all other parameters of the action
            """
            if _end_turn:
                return self.in_capturing_seq()
            return True

        def bounds_checking() -> bool:
            """Bounds checking on positions"""
            if _end_turn:
                return True
            for pos in (_from, _to):
                if not 0 <= pos < BOARD_SQUARES: # pos is within board bounds
                    return False
            if not 0 <= _capture <= BOARD_SQUARES: # capture may be BOARD_SQUARES in case of paika move
                return False 
            if _from == _to:
                return False
            return True

        def check_piece_validity() -> bool:
            """Checking validity of pieces at action positions"""
            if _end_turn:
                return True
            if self.get_piece(_from) != _who_to_play: # from position must contain a piece 
                return False
            if self.get_piece(_to) != Piece.EMPTY: # piece must be played to an empty location
                return False
            if _capture != BOARD_SQUARES and self.get_piece(_capture) != self.other_side(): # capturing line must start with opponent color stone
                return False
            return True

        def check_valid_dir() -> bool:
            """Checking that _dir is permitted from given board position"""
            if _end_turn:
                return True
            return _dir in FanoronaEnv.get_valid_dirs(_from)

        def check_no_paika_when_capture() -> bool:
            """Check if paika is being played when capturing move exists, which is illegal"""
            if _end_turn:
                return True
            return not (_capture_type == 0 and self.capture_exists())

        def move_only_capturing_piece() -> bool:
            """
            If in a capturing sequence, check that capturing piece is the one being moved, and 
            not some other piece
            """
            if _end_turn:
                return True
            _from_row, _from_col = FanoronaEnv.convert_pos_to_coords(_from)
            if self.in_capturing_seq() and _visited_pos[_from_row][_from_col] != 1:
                return False
            return True
        
        def check_no_overlap() -> bool:
            """Check that capturing piece is not visiting previously visited pos in capturing path"""
            if _end_turn:
                return True
            _to_row, _to_col = FanoronaEnv.convert_pos_to_coords(_to)
            if _visited_pos[_to_row][_to_col] == 1:
                return False
            return True

        def check_no_same_dir() -> bool:
            """Check that capturing piece is not moving twice in the same direction"""
            if _end_turn:
                return True
            return _dir != _last_dir

        rules = {
            'half_move_rule': half_move_rule,
            'end_turn_rule': end_turn_rule,
            'bounds_checking': bounds_checking,
            'check_piece_validity': check_piece_validity,
            'check_valid_dir': check_valid_dir,
            'check_no_paika_when_capture': check_no_paika_when_capture,
            'move_only_capturing_piece': move_only_capturing_piece, 
            'check_no_overlap': check_no_overlap,
            'check_no_same_dir': check_no_same_dir
        }
        for name, test in rules.items():
            if name in skip:
                continue
            if not test():
                return False
        return True

    # TODO: turn this into a generator
    def get_valid_moves(self) -> List[Tuple[int, Direction, int, int]]:
        """
        Returns a list of all valid moves (in the form of actions).

        Incorporates the rule that a capture must be played if one is available. Works by -
        1. Scan all pieces (of turn to play) and directions for all possible moves + captures in separate lists
        2. If in capturing sequence, add end_turn action (0, 0, 0, 1)
        3. If captures not empty, return captures, else moves
        """
        moves: List[Tuple[int, Direction, int, int]] = []
        captures: List[Tuple[int, Direction, int, int]] = []
        _board_state, _who_to_play, _last_dir, _visited_pos, _half_moves = self.state
        for pos in range(BOARD_SQUARES):
            if self.get_piece(pos) == _who_to_play:
                for _dir in Direction:
                    move_action = (pos, _dir, 0, 0)
                    if self.is_valid(move_action):
                        moves.append(move_action)
                    for capture_type in [1, 2]:
                        capture_action = (pos, _dir, capture_type, 0)
                        if self.is_valid(capture_action):
                            captures.append(capture_action)
        if self.in_capturing_seq():
            captures.append((0, Direction(0), 0, 1))
        if captures:
            return captures
        else:
            return moves

    def piece_exists(self, piece: Piece) -> bool:
        """Checks whether a instance of a piece exists on the game board."""
        for pos in range(BOARD_SQUARES):
            if self.get_piece(pos) == piece:
                return True
        return False

    def is_done(self) -> Tuple[bool, Reward]:
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
            # cannot have a situation where a piece exists but there are no valid moves
            if not own_piece_exists:
                return True, Reward.LOSS
            if not other_piece_exists:
                return True, Reward.WIN
            else:
                return False, Reward.NONE

    def reset_visited_pos(self) -> None:
        """Resets visited_pos of a state to indicate no visited positions."""
        _, _, _, _visited_pos, _ = self.state
        for pos in range(BOARD_SQUARES):
            row, col = FanoronaEnv.convert_pos_to_coords(pos)
            _visited_pos[row][col] = 0

    def step(self, action):
        """Compute return values based on action taken."""

        _from, _dir, _capture_type, _end_turn = action
        _board_state, _who_to_play, _last_dir, _visited_pos, _half_moves = self.state
        _to = FanoronaEnv.displace_pos(_from, _dir)
        _from_row, _from_col = FanoronaEnv.convert_pos_to_coords(_from)
        _to_row, _to_col = FanoronaEnv.convert_pos_to_coords(_to)

        if self.is_valid(action):

            if _end_turn: # end turn
                _who_to_play = self.other_side()
                _last_dir = Direction.X
                self.reset_visited_pos()
                _half_moves += 1
                self.state = (_board_state, _who_to_play, _last_dir, _visited_pos, _half_moves)

            else:
                _board_state[_to_row][_to_col] = self.get_piece(_from)
                _board_state[_from_row][_from_col] = Piece.EMPTY

                if _capture_type == 0: # paika move
                    _who_to_play = self.other_side()
                    _last_dir = Direction.X
                    self.reset_visited_pos()
                    _half_moves += 1
                    self.state = (_board_state, _who_to_play, _last_dir, _visited_pos, _half_moves)
                
                else: # capture (approach, withdrawal)
                    if _capture_type == 1: # approach
                        _capture = FanoronaEnv.displace_pos(_to, _dir)
                        _capture_dir = _dir
                    else: # withdraw
                        _capture = FanoronaEnv.displace_pos(_from, 8 - _dir)
                        _capture_dir = 8 - _dir
                    _capture_row, _capture_col = FanoronaEnv.convert_pos_to_coords(_capture)
                    while 0 <= _capture_row < BOARD_ROWS and 0 <= _capture_col < BOARD_COLS and _board_state[_capture_row][_capture_col] == self.other_side():
                        _board_state[_capture_row][_capture_col] = Piece.EMPTY
                        _capture = FanoronaEnv.displace_pos(_capture, _capture_dir)
                        _capture_row, _capture_col = FanoronaEnv.convert_pos_to_coords(_capture)
                    
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

        # if in capturing sequence, and no valid moves available (other than end turn), then cause turn to end
        if self.in_capturing_seq() and len(self.get_valid_moves()) == 1:
            _who_to_play = self.other_side()
            _last_dir = Direction.X
            self.reset_visited_pos()
            _half_moves += 1
        
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
            0
        )

        return self.state
    
    def set_state_from_board_str(self, board_string: str) -> None:
        """Set the env object state using a board string."""

        def process_board_state_str(board_state_str: str):
            row_strings = board_state_str.split('/')
            board_state_chars = [list(row) for row in row_strings]
            board_state = np.zeros(shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8)
            for row, row_content in enumerate(board_state_chars):
                col_board = 0
                for col, cell in enumerate(row_content):
                    if cell == 'W':
                        board_state[row][col_board] = Piece.WHITE
                    elif cell == 'B':
                        board_state[row][col_board] = Piece.BLACK
                    else:
                        for col_board in range(col_board, col_board + int(cell)):
                            board_state[row][col_board] = Piece.EMPTY
                    col_board += 1
            return board_state

        def process_visited_pos_str(visited_pos_str: str):
            visited = np.zeros(shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8)
            if visited_pos_str != '-':
                visited_pos_list = visited_pos_str.split(',')
                for human_pos in visited_pos_list:
                    row, col = FanoronaEnv.convert_human_to_coords(human_pos)
                    visited[row][col] = True
            return visited
                        
        _board_state_str, _who_to_play_str, _last_dir_str, _visited_pos_str, _half_moves_str = board_string.split()

        _board_state = process_board_state_str(_board_state_str)
        _who_to_play = Piece.WHITE if _who_to_play_str == 'W' else Piece.BLACK
        _last_dir = Direction[_last_dir_str] if _last_dir_str != '-' else Direction.X
        _visited_pos = process_visited_pos_str(_visited_pos_str)
        _half_moves = int(_half_moves_str)

        self.state = (_board_state, _who_to_play, _last_dir, _visited_pos, _half_moves)

    def get_board_str(self) -> str:
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
        _board_state, _who_to_play, _last_dir, _visited_pos, _half_moves = self.state

        board_string = ''
        count = 0
        for row in _board_state:
            for col in row:
                col_str = str(Piece(col))
                if col == Piece.EMPTY:
                    count += 1
                else:
                    if count > 0:
                        board_string += str(count)
                        count = 0
                    board_string += col_str
            if count > 0:
                board_string += str(count)
                count = 0
            board_string += '/'
        board_string = board_string.rstrip('/')
        if count > 0:
                board_string += str(count)

        _who_to_play_str = str(Piece(_who_to_play))
        _last_dir_str = str(Direction(_last_dir))
        
        visited_pos_list = []
        for row_idx, row in enumerate(_visited_pos):
            for col_idx, col in enumerate(row):
                if col:
                    visited_pos_list.append(FanoronaEnv.convert_coords_to_human((row_idx, col_idx)))
        visited_pos_str = ','.join(visited_pos_list)
        if not visited_pos_list:
            visited_pos_str = '-'

        return ' '.join([board_string, _who_to_play_str, _last_dir_str, visited_pos_str, str(_half_moves)])

    def render(self, mode: str = 'human', close: bool = False) -> None:
        _board_state, _, _, _, _ = self.state
        if mode == 'human':
            print(self.get_board_str())
        elif mode == 'svg':
            def convert(coord: Tuple[int, int]) -> Tuple[int, int]:
                row, col = coord
                return 100 + col * 100, 100 + (4 - row) * 100
            svg_w = 1000
            svg_h = 600
            black_piece = '<circle cx="{0[0]!s}" cy="{0[1]!s}" r="30" stroke="black" stroke-width="1.5" fill="black" />'
            white_piece = '<circle cx="{0[0]!s}" cy="{0[1]!s}" r="30" stroke="black" stroke-width="1.5" fill="white" />'
            line = '<line x1="{0[0]!s}" y1="{0[1]!s}" x2="{1[0]!s}" y2="{1[1]!s}" stroke="black" stroke-width="1.5" />'
            board_lines = []
            for row in range(BOARD_ROWS):
                _from, _to = convert((row, 0)), convert((row, 8))
                horizontal = line.format(_from, _to)
                board_lines.append(horizontal)
            for col in range(BOARD_COLS):
                _from, _to = convert((0, col)), convert((4, col))
                vertical = line.format(_from, _to)
                board_lines.append(vertical)
            for diag_forward in range(0, BOARD_COLS, 2):
                _from = [(2, 0), (0, 0), (0, 2), (0, 4), (0, 6)]
                _to = [(4, 2), (4, 4), (4, 6), (4, 8), (2, 8)]
                board_lines.extend([line.format(convert(f), convert(t)) for f, t in zip(_from, _to)])
            for diag_backward in range(0, BOARD_COLS, 2):
                _from = [(2, 0), (4, 0), (4, 2), (4, 4), (4, 6)]
                _to = [(0, 2), (0, 4), (0, 6), (0, 8), (2, 8)]
                board_lines.extend([line.format(convert(f), convert(t)) for f, t in zip(_from, _to)])
            board_pieces = []
            for pos in range(BOARD_SQUARES):
                row, col = FanoronaEnv.convert_pos_to_coords(pos)
                if _board_state[row][col] == Piece.WHITE:
                    board_pieces.append(white_piece.format(convert((row, col))))
                elif _board_state[row][col] == Piece.BLACK:
                    board_pieces.append(black_piece.format(convert((row, col))))
            svg_lines = '\n\t'.join(board_lines + board_pieces)
            svg = f"""
<svg height="{svg_h}" width="{svg_w}">
    {svg_lines}
</svg>
"""
            with open('temp.svg', 'w') as outfile:
                outfile.write(svg)
