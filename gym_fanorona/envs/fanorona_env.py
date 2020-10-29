from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

from .action import FanoronaMove
from .constants import BOARD_COLS, BOARD_ROWS, BOARD_SQUARES, MOVE_LIMIT
from .enums import Direction, Piece, Reward
from .position import Position
from .state import FanoronaState


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

    metadata = {'render.modes': ['human', 'svg']}

    def __init__(self, white_player = None, black_player = None) -> None:

        super(FanoronaEnv, self).__init__()
        self.action_space = spaces.Tuple((
            spaces.Discrete(BOARD_SQUARES),    # from
            spaces.Discrete(len(Direction)), # direction 
            spaces.Discrete(3),              # capture type (none=0, approach=1, withdrawal=2)
            spaces.Discrete(2)               # end turn (0 for no, 1 for yes) 
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=2, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8), # board state: (9 x 5) x Piece
            spaces.Discrete(2),                                                       # turn to play: (WHITE, BLACK)
            spaces.Discrete(len(Direction)),                                          # last direction used: Direction 
            spaces.Box(low=0, high=1, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8), # positions used: (9 x 5) x (True, False)
            spaces.Discrete(MOVE_LIMIT + 1)                                           # number of half-moves
        ))

        self.state: FanoronaState = FanoronaState()
        self.white_player = white_player # agent playing as white
        self.black_player = black_player # agent playing as black

    def get_valid_moves(self) -> List[FanoronaMove]:
        """
        Returns a list of all valid moves (in the form of actions).

        Incorporates the rule that a capture must be played if one is available. Works by -
        1. Scan all pieces (of turn to play) and directions for all possible moves + captures in separate lists
        2. If in capturing sequence, add end_turn action (0, 0, 0, 1)
        3. If captures not empty, return captures, else moves
        """
        moves: List[FanoronaMove] = []
        captures: List[FanoronaMove] = []
        for pos in Position.pos_range():
            if self.state.get_piece(pos) == self.state.turn_to_play:
                for direction in Direction:
                    move_action = FanoronaMove(pos, direction, 0, False)
                    if move_action.is_valid(self.state):
                        moves.append(move_action)
                    for capture_type in [1, 2]:
                        capture_action = FanoronaMove(pos, direction, capture_type, False)
                        if capture_action.is_valid(self.state):
                            captures.append(capture_action)
        if self.state.in_capturing_seq():
            end_turn_action = FanoronaMove(Position((0, 0)), Direction(0), 0, True)
            captures.append(end_turn_action)
        if captures:
            return captures
        else:
            return moves

    def step(self, action: FanoronaMove) -> Tuple[FanoronaState, int, bool, dict]:
        """Compute return values based on action taken."""

        to = action.position.displace(action.direction)
        from_row, from_col = action.position.to_coords()
        to_row, to_col = to.to_coords()

        if action.is_valid(self.state):

            if action.end_turn: # end turn
                self.state.turn_to_play = self.state.other_side()
                self.state.last_dir = Direction.X
                self.state.reset_visited_pos()
                self.state.half_moves += 1
            else:
                self.state.board_state[to_row][to_col] = self.state.get_piece(action.position)
                self.state.board_state[from_row][from_col] = Piece.EMPTY

                if action.capture_type == 0: # paika move
                    self.state.turn_to_play = self.state.other_side()
                    self.state.last_dir = Direction.X
                    self.state.reset_visited_pos()
                    self.state.half_moves += 1
                
                else: # capture (approach, withdrawal)
                    if action.capture_type == 1: # approach
                        capture = to.displace(action.direction)
                        capture_dir = action.direction
                    else: # withdraw
                        capture = action.position.displace(action.direction.opposite())
                        capture_dir = action.direction.opposite()
                    capture_row, capture_col = capture.to_coords()
                    while capture.is_valid() and self.state.board_state[capture_row][capture_col] == self.state.other_side():
                        self.state.board_state[capture_row][capture_col] = Piece.EMPTY
                        capture = capture.displace(capture_dir)
                        capture_row, capture_col = capture.to_coords()
                    
                    self.state.last_dir = action.direction
                    self.state.visited[from_row][from_col] = 1
                    self.state.visited[to_row][to_col] = 1

        else: # invalid move
            obs = self.state
            done = self.state.is_done()
            reward = Reward.ILLEGAL_MOVE
            info: Dict[Any, Any] = {}
            return obs, reward, done, info

        # if in capturing sequence, and no valid moves available (other than end turn), then force turn to end
        if self.state.in_capturing_seq() and len(self.get_valid_moves()) == 1:
            self.state.turn_to_play = self.state.other_side()
            self.state.last_dir = Direction.X
            self.state.reset_visited_pos()
            self.state.half_moves += 1
        
        obs = self.state
        reward = Reward.NONE # reward logic should be refined based on training requirements
        done = self.state.is_done()
        info = {}
        return obs, reward, done, info

    def reset(self) -> None:
        START_STATE_STR = 'WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0'
        self.state = FanoronaState.set_from_board_str(START_STATE_STR)

    def render(self, 
               mode: str = 'human', 
               close: bool = False, 
               filename: str = 'board_000.svg') -> None:
        if mode == 'human':
            print(self.state.get_board_str())
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
                _from_h, _to_h = convert((row, 0)), convert((row, 8))
                horizontal = line.format(_from_h, _to_h)
                board_lines.append(horizontal)
            for col in range(BOARD_COLS):
                _from_v, _to_v = convert((0, col)), convert((4, col))
                vertical = line.format(_from_v, _to_v)
                board_lines.append(vertical)
            for _ in range(0, BOARD_COLS, 2): # diagonal forward lines
                _from_df = [(2, 0), (0, 0), (0, 2), (0, 4), (0, 6)]
                _to_df = [(4, 2), (4, 4), (4, 6), (4, 8), (2, 8)]
                board_lines.extend([line.format(convert(f), convert(t)) for f, t in zip(_from_df, _to_df)])
            for _ in range(0, BOARD_COLS, 2): # diagonal backward lines
                _from_db = [(2, 0), (4, 0), (4, 2), (4, 4), (4, 6)]
                _to_db = [(0, 2), (0, 4), (0, 6), (0, 8), (2, 8)]
                board_lines.extend([line.format(convert(f), convert(t)) for f, t in zip(_from_db, _to_db)])
            board_pieces = []
            for pos in Position.pos_range():
                row, col = pos.to_coords()
                if self.state.board_state[row][col] == Piece.WHITE:
                    board_pieces.append(white_piece.format(convert((row, col))))
                elif self.state.board_state[row][col] == Piece.BLACK:
                        board_pieces.append(black_piece.format(convert((row, col))))
            svg_lines = '\n\t'.join(board_lines + board_pieces)
            svg = f"""
<svg height="{svg_h}" width="{svg_w}">
    {svg_lines}
</svg>
"""
            with open(filename, 'w') as outfile:
                outfile.write(svg)

    def play_game(self) -> List[FanoronaMove]:
        self.reset()
        move_list = []
        done = False
        while not done:
            if not done:
                white_move = self.white_player.move(self)
                move_list.append(white_move)
                obs, reward, done, info = self.step(white_move)
                self.white_player.receive_reward(reward)
            if not done:
                black_move = self.black_player.move(self)
                move_list.append(black_move)
                obs, reward, done, info = self.step(black_move)
                self.black_player.receive_reward(reward)
        return move_list