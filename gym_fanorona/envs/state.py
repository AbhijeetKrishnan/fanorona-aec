from typing import Tuple

import numpy as np

from .constants import MOVE_LIMIT, BOARD_COLS, BOARD_ROWS
from .enums import Direction, Piece, Reward
from .position import Position


class FanoronaState:

    def __init__(self):
        self.board_state: Any = None # TODO: replace with numpy types from numpy.typing
        self.turn_to_play: Piece = Piece.EMPTY
        self.last_dir: Direction = Direction.X
        self.visited: Any = None # TODO: replace with numpy types from numpy.typing OR consider not using numpy at all
        self.half_moves: int = 0

    def get_piece(self, position: Position) -> Piece:
        """Return type of piece at given position (specified in integer coordinates)."""
        assert position.is_valid()
        row, col = position.to_coords()
        return Piece(self.board_state[row][col])

    def other_side(self) -> Piece:
        """Return the color of the opponent's pieces."""
        if self.who_to_play == Piece.WHITE:
            return Piece.BLACK
        else:
            return Piece.WHITE

    def in_capturing_seq(self) -> bool:
        """Returns True if current state is part of a capturing sequence i.e. at least one capture has already been made."""
        return bool(self.last_dir != Direction.X)

    def capture_exists(self) -> bool:
        """
        Returns True if any capturing move exists in the current state.

        A capture exists if -
        1. a piece belonging to the side to play exists
        2. the action of moving the piece in any valid direction in any capture type is also valid 
           (ignoring the no paika when capture exists rule)
        """
        for pos in Position.pos_range():
            if self.get_piece(pos) == self.who_to_play:
                for Direction in pos.get_valid_dirs():
                    for capture_type in [1, 2]:
                        capture_action = FanoronaMove(pos, direction, capture_type, False)
                        if capture_action.is_valid(self, skip=['check_no_paika_when_captured']):
                            return True
        return False
    
    def piece_exists(self, piece: Piece) -> bool:
        """Checks whether a instance of a piece exists on the game board."""
        for pos in pos_range():
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
        if self.half_moves >= MOVE_LIMIT:
            return True, Reward.DRAW
        else:
            own_piece_exists = self.piece_exists(self.who_to_play)
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
        for pos in pos_range():
            row, col = pos.to_coords()
            self.visited[row][col] = 0

    def set_from_board_str(self, board_string: str) -> None:
        """Set the current state using a board string."""

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
                    row, col = Position(human_pos).to_coords()
                    visited[row][col] = True
            return visited
        
        board_state_str, who_to_play_str, last_dir_str, visited_pos_str, half_moves_str = board_string.split()

        self.board_state = process_board_state_str(board_state_str)
        self.who_to_play = Piece.WHITE if who_to_play_str == 'W' else Piece.BLACK
        self.last_dir = Direction[last_dir_str] if last_dir_str != '-' else Direction.X
        self.visited = process_visited_pos_str(visited_pos_str)
        self.half_moves = int(half_moves_str)

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
        board_string = ''
        count = 0
        for row in self.board_state:
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

        who_to_play_str = str(Piece(self.who_to_play))
        last_dir_str = str(Direction(self.last_dir))
        
        visited_pos_list = []
        for row_idx, row in enumerate(self.visited):
            for col_idx, col in enumerate(row):
                if col:
                    visited_pos_list.append(Position((row_idx, col_idx)).to_human())
        visited_pos_str = ','.join(visited_pos_list)
        if not visited_pos_list:
            visited_pos_str = '-'

        return ' '.join([board_string, who_to_play_str, last_dir_str, visited_pos_str, str(self.half_moves)])
