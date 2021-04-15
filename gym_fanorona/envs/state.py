from typing import Tuple, Optional, List

import numpy as np
import numpy.typing as npt

from .utils import MOVE_LIMIT, BOARD_COLS, BOARD_ROWS, Direction, Piece, Position
from .action import FanoronaMove, MoveType


class FanoronaState:
    def __init__(self):
        self.board_state: Optional[npt.ArrayLike] = None
        self.turn_to_play: Piece = Piece.EMPTY
        self.last_capture: Tuple[Position, Direction] = None # TODO: propagate this addition to the rest of the code
        self.visited: Optional[npt.ArrayLike] = None
        self.half_moves: int = 0

    def __repr__(self):
        return f"<FanoronaState: {str(self)}>"

    def __str__(self):
        board_string = ""
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
            board_string += "/"
        board_string = board_string.rstrip("/")
        if count > 0:
            board_string += str(count)

        turn_to_play_str = str(Piece(self.turn_to_play))
        last_dir_str = str(Direction(self.last_dir))

        visited_pos_list = []
        for row_idx, row in enumerate(self.visited):
            for col_idx, col in enumerate(row):
                if col:
                    visited_pos_list.append(Position((row_idx, col_idx)).to_human())
        visited_pos_str = ",".join(visited_pos_list)
        if not visited_pos_list:
            visited_pos_str = "-"

        return " ".join(
            [
                board_string,
                turn_to_play_str,
                last_dir_str,
                visited_pos_str,
                str(self.half_moves),
            ]
        )
    
    def to_svg(self, svg_w: int=1000, svg_h: int=600) -> str: # TODO: adjust output svg size dynamically
        def convert(coord: Tuple[int, int]) -> Tuple[int, int]:
            row, col = coord
            return 100 + col * 100, 100 + (4 - row) * 100

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
        for _ in range(0, BOARD_COLS, 2):  # diagonal forward lines
            _from_df = [(2, 0), (0, 0), (0, 2), (0, 4), (0, 6)]
            _to_df = [(4, 2), (4, 4), (4, 6), (4, 8), (2, 8)]
            board_lines.extend(
                [line.format(convert(f), convert(t)) for f, t in zip(_from_df, _to_df)]
            )
        for _ in range(0, BOARD_COLS, 2):  # diagonal backward lines
            _from_db = [(2, 0), (4, 0), (4, 2), (4, 4), (4, 6)]
            _to_db = [(0, 2), (0, 4), (0, 6), (0, 8), (2, 8)]
            board_lines.extend(
                [line.format(convert(f), convert(t)) for f, t in zip(_from_db, _to_db)]
            )
        board_pieces = []
        for pos in Position.pos_range():
            row, col = pos.to_coords()
            if self.board_state[row][col] == Piece.WHITE:
                board_pieces.append(white_piece.format(convert((row, col))))
            elif self.board_state[row][col] == Piece.BLACK:
                board_pieces.append(black_piece.format(convert((row, col))))
        svg_lines = "\n\t".join(board_lines + board_pieces)
        svg = f"""
<svg height="{svg_h}" width="{svg_w}">
{svg_lines}
</svg>
"""
        return svg

    def get_piece(self, position: Position) -> Piece:
        """Return type of piece at given position (specified in integer coordinates)."""
        return Piece(self.board_state[position.row][position.col])

    def piece_exists(self, piece: Piece) -> bool:
        """Checks whether an instance of a piece exists on the game board."""
        for pos in Position.pos_range():
            if self.get_piece(pos) == piece:
                return True
        return False

    def push(self, move: FanoronaMove):
        """Implement the rules of Fanorona and make the desired move on the board. Returns flags and
        status codes depending on game over or draw
        """
        to = move.position.displace(move.direction)
        from_row, from_col = move.position.to_coords()
        to_row, to_col = to.to_coords()

        # Assume move is valid. Validity check implemented using action mask and TerminateIllegal
        # wrapper

        # Assume that when move.end_turn is True, move.direction is Direction.X
        from_piece = self.get_piece(move.position)
        self.board_state[from_row][from_col] = Piece.EMPTY
        self.board_state[to_row][to_col] = from_piece

        def end_turn():
            self.turn_to_play = self.turn_to_play.other()
            self.last_dir = Direction.X
            self.visited.fill(0) # reset visited ndarray
            self.half_moves += 1   

        if not move.end_turn and move.capture_type != MoveType.PAIKA:
            if move.capture_type == MoveType.APPROACH:  # approach
                capture_pos = to.displace(move.direction)
                capture_dir = move.direction
            else:  # withdraw
                capture_pos = move.position.displace(move.direction.opposite())
                capture_dir = move.direction.opposite()
            capture_row, capture_col = capture_pos.to_coords()
            while (
                capture_pos.is_valid()
                and self.board_state[capture_row][capture_col]
                == self.turn_to_play.other()
            ):
                self.board_state[capture_row][capture_col] = Piece.EMPTY
                capture_pos = capture_pos.displace(capture_dir)
                capture_row, capture_col = capture_pos.to_coords()

            self.last_dir = move.direction
            self.visited[from_row][from_col] = 1
            self.visited[to_row][to_col] = 1

            # if in capturing sequence, and no valid moves available (other than end turn), then
            # force turn to end
            if len(self.legal_moves) == 1:
                end_turn()

        else:  # end turn/paika move
             end_turn()

    def is_game_over(self) -> bool:
        """
        Check whether the game is over (i.e. the current state is a terminal state).

        The game is over when -
        a) One side has no pieces left to move (loss for the side which has no pieces to move)
        b) The number of half-moves exceeds the limit (draw)
        """
        if self.half_moves // 2 >= MOVE_LIMIT:
            return True
        else:
            own_piece_exists = self.piece_exists(self.turn_to_play)
            other_piece_exists = self.piece_exists(self.turn_to_play.other())
            # Conjecture: cannot have a situation in Fanorona where a piece exists but there are no
            # valid moves
            if own_piece_exists and other_piece_exists:
                return False
            else:
                return True

    def get_result(self) -> int:
        """Return result of the current game state. Returns 1 for white win, -1 for black win, and 0
        for draw. Assumes game is done.

        The game is done when -
        a) One side has no pieces left to move (loss for the side which has no pieces to move)
        b) The number of half-moves exceeds the limit (draw)
        """
        assert(self.is_game_over())

        if self.half_moves // 2 >= MOVE_LIMIT:
            return 0
        elif self.piece_exists(Piece.WHITE):
            return 1
        else:
            return -1

    def reset(self) -> None:
        "Reset to the start state"
        START_STATE_STR = "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0"
        self.set_from_board_str(START_STATE_STR)

    def set_from_board_str(self, board_string: str) -> None:
        """Set the state object to a new state represented by a board string.
        """
        def process_board_state_str(self, board_state_str: str):
            row_strings = board_state_str.split("/")
            board_state_chars = [list(row) for row in row_strings]
            self.board_state.fill(0)
            for row, row_content in enumerate(board_state_chars):
                col_board = 0
                for cell in row_content:
                    if cell == "W":
                        self.board_state[row][col_board] = Piece.WHITE
                    elif cell == "B":
                        self.board_state[row][col_board] = Piece.BLACK
                    else:
                        for col_board in range(col_board, col_board + int(cell)):
                            self.board_state[row][col_board] = Piece.EMPTY
                    col_board += 1

        def process_visited_pos_str(self, visited_pos_str: str):
            self.visited.fill(0)
            if visited_pos_str != "-":
                visited_pos_list = visited_pos_str.split(",")
                for human_pos in visited_pos_list:
                    row, col = Position(human_pos).to_coords()
                    self.visited[row][col] = True

        (
            board_state_str,
            turn_to_play_str,
            last_dir_str,
            visited_pos_str,
            half_moves_str,
        ) = board_string.split()

        process_board_state_str(self, board_state_str)
        self.turn_to_play = Piece.WHITE if turn_to_play_str == "W" else Piece.BLACK
        self.last_dir = Direction[last_dir_str] if last_dir_str != "-" else Direction.X
        process_visited_pos_str(self, visited_pos_str)
        self.half_moves = int(half_moves_str)     

    @staticmethod
    def get_observation(self, agent):
        "Return NN-style observation based on the current board state and requesting agent"
        pass

    def is_valid(self, move: FanoronaMove) -> bool: # TODO: fix this; split it according to move type - paika, fresh capture, capture from capturing sequence
        """Check if a given move is valid from the current board state.

        Assumes the following about the input move -
        1. the piece being moved belongs to the colour whose turn it is to play
        2. the square being moved from contains a piece of that colour
        3. the move is not an end turn
        """
        to = move.position.displace(move.direction)
        if move.capture_type == 1:  # approach
            capture = to.displace(move.direction)
        else:  # withdraw
            capture = move.position.displace(move.direction.opposite())

        def bounds_checking() -> bool:
            """Bounds checking on positions"""
            for pos in (move.position, to):
                if not pos.is_valid():  # pos is not within board bounds
                    return False
            if (
                move.capture_type != 0 and not capture.is_valid()
            ):  # capture may be outside in case of paika move
                return False
            if move.position == to:
                return False
            return True
        
        def check_valid_dir() -> bool:
            """Checking that _dir is permitted from given board position"""
            return move.direction in move.position.get_valid_dirs()

        def check_piece_validity() -> bool:
            """Checking validity of pieces at action positions"""
            if (
                self.get_piece(to) != Piece.EMPTY
            ):  # piece must be played to an empty location
                return False
            if (
                move.capture_type != 0
                and capture.is_valid()
                and self.get_piece(capture) != self.turn_to_play.other()
            ):  # capturing line must start with opponent color stone
                return False
            return True

        def move_only_capturing_piece() -> bool:
            """
            If in a capturing sequence, check that capturing piece is the one being moved, and 
            not some other piece
            """
            if move.end_turn:
                return True
            _from_row, _from_col = move.position.to_coords()
            if state.in_capturing_seq() and state.visited[_from_row][_from_col] != 1:
                return False
            return True

        def check_no_overlap() -> bool:
            """Check that capturing piece is not visiting previously visited pos in capturing path"""
            if move.end_turn:
                return True
            _to_row, _to_col = to.to_coords()
            if state.visited[_to_row][_to_col] == 1:
                return False
            return True

        def check_no_same_dir() -> bool:
            """Check that capturing piece is not moving twice in the same direction"""
            if move.end_turn:
                return True
            return move.direction != state.last_dir

        rules = {
            "bounds_checking": bounds_checking,
            "check_piece_validity": check_piece_validity,
            "check_valid_dir": check_valid_dir,
            "move_only_capturing_piece": move_only_capturing_piece,
            "check_no_overlap": check_no_overlap,
            "check_no_same_dir": check_no_same_dir,
        }
        for name, test in rules.items():
            if name in skip:
                continue
            if not test():
                return False
        return True

    @property
    def legal_moves(self) -> List[int]:
        """Return a list of legal actions allowed from the current state. Actions are in their
        integer encoding

        Iterates through all possible actions and validates them, then adds them to the list.
        """
        legal_captures: List[int] = []
        legal_paikas: List[int] = []

        if self.last_dir != Direction.X:
            # TODO: check for captures involving last moved piece only

            # add end turn 
            end_turn_action = FanoronaMove(Position((0, 0)), Direction.X, 0, True)
            legal_captures.append(end_turn_action)

        # check for captures
        for pos in Position.pos_range():
            if self.get_piece(pos) == self.turn_to_play:
                for direction in Direction:
                    for capture_type in [1, 2]:  # approach = 1, withdrawal = 2
                        capture = FanoronaMove(pos, direction, capture_type, False)
                        if self.is_valid(capture):
                            legal_captures.append(capture.to_action())

        # only check for paikas if no captures exist
        if not legal_captures:
            for pos in Position.pos_range():
                if self.get_piece(pos) == self.turn_to_play:
                    for direction in Direction:    
                        paika = FanoronaMove(pos, direction, 0, False)
                        if self.is_valid(paika):
                            legal_paikas.append(paika.to_action())

        if legal_captures: # capture has to be made if available
            return legal_captures
        else:
            return legal_paikas
