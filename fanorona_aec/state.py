from typing import Tuple, Optional, List, cast

import numpy as np

from .utils import MOVE_LIMIT, BOARD_COLS, BOARD_ROWS, Direction, Piece, Position
from .move import FanoronaMove, MoveType, END_TURN


class FanoronaState:
    def __init__(self):
        self.board: Optional[np.ndarray] = None
        self.turn_to_play: Piece = Piece.EMPTY
        self.last_capture: Optional[
            Tuple[Position, Direction]
        ] = None  # TODO: remove requirement to index using 0 and 1 (NamedTuple?)
        self.visited: Optional[np.ndarray] = None
        self.half_moves: int = 0

    def __repr__(self):
        return f"<FanoronaState: {str(self)}>"

    def __str__(self):
        board_string = ""
        count = 0
        for row in self.board:
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
        if self.last_capture:
            last_capture_str = (
                f"{self.last_capture[0].to_human()} {str(self.last_capture[1])}"
            )
        else:
            last_capture_str = f"- -"

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
                last_capture_str,
                visited_pos_str,
                str(self.half_moves),
            ]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FanoronaState):
            return NotImplemented
        return str(self) == str(other)

    def to_svg(self, svg_w: int = 1000, svg_h: int = 600) -> str:
        # TODO: adjust output svg size dynamically
        # TODO: represent other aspects of state on the output svg (turn to play, last capture, visited etc.)
        def convert(coord: Tuple[int, int]) -> Tuple[int, int]:
            row, col = coord
            return 100 + col * 100, 100 + (4 - row) * 100

        if self.board is None or self.visited is None:
            raise Exception('render(mode="svg") called without calling reset()')

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
            if self.board[row][col] == Piece.WHITE:
                board_pieces.append(white_piece.format(convert((row, col))))
            elif self.board[row][col] == Piece.BLACK:
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
        if self.board is not None:
            return Piece(self.board[position.row][position.col])
        else:
            raise Exception("Called get_piece() without calling reset()")

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
        if self.board is None or self.visited is None:
            raise Exception("Called push() without calling reset()")

        # Direction.X is not part of the action space and is an internal implementation detail
        if move.end_turn:
            move.direction = Direction.X
        to = move.position.displace(move.direction)
        from_row, from_col = move.position.to_coords()
        to_row, to_col = to.to_coords()

        # Assume move is valid. Validity check implemented using action mask and TerminateIllegal
        # wrapper

        from_piece = self.get_piece(move.position)
        self.board[from_row][from_col] = Piece.EMPTY
        self.board[to_row][to_col] = from_piece

        def end_turn():
            self.turn_to_play = self.turn_to_play.other()
            self.last_capture = None
            self.visited.fill(0)  # reset visited ndarray
            self.half_moves += 1

        if not move.end_turn and move.move_type != MoveType.PAIKA:
            if move.move_type == MoveType.APPROACH:  # approach
                capture_pos = to.displace(move.direction)
                capture_dir = move.direction
            else:  # withdraw
                capture_pos = move.position.displace(move.direction.opposite())
                capture_dir = move.direction.opposite()
            capture_row, capture_col = capture_pos.to_coords()
            while (
                capture_pos.is_valid()
                and self.board[capture_row][capture_col] == self.turn_to_play.other()
            ):
                self.board[capture_row][capture_col] = Piece.EMPTY
                capture_pos = capture_pos.displace(capture_dir)
                capture_row, capture_col = capture_pos.to_coords()

            self.last_capture = (to, move.direction)
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
        if self.half_moves >= MOVE_LIMIT:
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
        assert self.is_game_over()  # TODO: make done a state property instead?

        if self.half_moves >= MOVE_LIMIT:
            return 0
        elif self.piece_exists(Piece.WHITE):
            return 1
        else:
            return -1

    def reset(self) -> None:
        "Reset to the start state"
        START_STATE_STR = "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - - 0"
        self.set_from_board_str(START_STATE_STR)

    def set_from_board_str(self, board_string: str) -> "FanoronaState":
        """Set the state object to a new state represented by a board string.
        """

        def process_board_state_str(self, board_state_str: str):
            row_strings = board_state_str.split("/")
            board_state_chars = [list(row) for row in row_strings]
            if self.board is not None:
                self.board.fill(0)
            else:
                self.board = np.zeros(shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int32)
            for row, row_content in enumerate(board_state_chars):
                col_board = 0
                for cell in row_content:  # TODO: any way to speed this up?
                    if cell == "W":
                        self.board[row][col_board] = Piece.WHITE
                    elif cell == "B":
                        self.board[row][col_board] = Piece.BLACK
                    else:
                        for col_board in range(col_board, col_board + int(cell)):
                            self.board[row][col_board] = Piece.EMPTY
                    col_board += 1

        def process_visited_pos_str(self, visited_pos_str: str):
            if self.visited is not None:
                self.visited.fill(0)
            else:
                self.visited = np.zeros(shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int32)
            if visited_pos_str != "-":
                visited_pos_list = visited_pos_str.split(",")
                for human_pos in visited_pos_list:  # TODO: any way to speed this up?
                    row, col = Position(human_pos).to_coords()
                    self.visited[row][col] = True

        (
            board_state_str,
            turn_to_play_str,
            last_capture_pos,
            last_capture_dir,
            visited_pos_str,
            half_moves_str,
        ) = board_string.split()

        process_board_state_str(self, board_state_str)
        self.turn_to_play = Piece.WHITE if turn_to_play_str == "W" else Piece.BLACK
        if last_capture_pos != "-" and last_capture_dir != "-":
            self.last_capture = (
                Position(last_capture_pos),
                Direction(last_capture_dir),
            )
        else:
            self.last_capture = None
        process_visited_pos_str(self, visited_pos_str)
        self.half_moves = int(half_moves_str)
        return self

    def get_observation(self, agent):
        """Return NN-style observation based on the current board state and requesting agent. Board
        state is from the perspective of the agent, with their color at the bottom.
        """
        if self.board is None or self.visited is None:
            raise Exception("Called get_observation() without calling reset()")

        obs = np.zeros(shape=(5, 9, 8), dtype=np.int32)
        # TODO: how to handle different observations from different sides? Specifically, how would actions change?

        # channel 1
        obs[:, :, 0] = int(self.turn_to_play)

        # channel 2
        half_moves_pos = Position(self.half_moves)
        assert (
            half_moves_pos.is_valid()
        ), f"{half_moves_pos} is not a valid position. Half-moves = {self.half_moves}"
        obs[half_moves_pos.row, half_moves_pos.col, 1] = 1

        # channel 3
        obs[:, :, 2] = self.visited.copy()

        if self.last_capture is not None:
            # channel 4
            obs[self.last_capture[0].row, self.last_capture[0].col, 3] = 1

            # channel 5
            last_dir_int = (
                int(self.last_capture[1])
                - 1
                - (1 if int(self.last_capture[1].value) >= 4 else 0)
            )
            obs[:, last_dir_int, 4] = 1

        # channel 6
        obs[:, :, 5].fill(1)

        # channel 7
        white_pieces_mask = self.board != int(Piece.WHITE)
        obs[:, :, 6] = self.board.copy()
        obs[white_pieces_mask, 6] = 1
        obs[~white_pieces_mask, 6] = 0

        # channel 8
        black_pieces_mask = self.board != int(Piece.BLACK)
        obs[:, :, 7] = self.board.copy()
        obs[black_pieces_mask, 7] = 1
        obs[~black_pieces_mask, 7] = 0

        return obs

    def is_valid(self, move: FanoronaMove) -> bool:
        """Check if a given move is valid from the current board state.

        Assumes the following about the input move -
        1. the piece being moved belongs to the colour whose turn it is to play
        2. the square being moved from contains a piece of that colour
        3. the move is not an end turn
        """
        if self.board is None or self.visited is None:
            raise Exception(f"Called is_valid({str(move)}) without calling reset()")

        to = move.position.displace(move.direction)
        if move.move_type == MoveType.APPROACH:
            capture = to.displace(move.direction)
        elif move.move_type == MoveType.WITHDRAWAL:
            capture = move.position.displace(move.direction.opposite())
        else:
            capture = Position(
                "A1"
            )  # dummy position to ensure capture.is_valid() is True

        def check_bounds() -> bool:
            """Bounds checking on positions"""
            return all(map(lambda pos: pos.is_valid(), (move.position, to, capture)))

        def check_valid_dir() -> bool:
            """Checking that move direction is permitted from given board position"""
            return move.direction in move.position.get_valid_dirs()

        def check_move_to_empty() -> bool:
            """Checking that piece is being moved to empty location"""
            if self.get_piece(to) != Piece.EMPTY:
                return False
            return True

        def check_opposite_color_capture() -> bool:
            """Checking that piece being captured is of opposite color"""
            if (
                capture.is_valid()
                and self.get_piece(capture) != self.turn_to_play.other()
            ):  # capturing line must start with opponent color stone
                return False
            return True

        def check_move_only_capturing_piece() -> bool:
            """If in a capturing sequence, check that capturing piece is the one being moved, and 
            not some other piece
            """
            assert self.last_capture is not None
            if self.last_capture[0] != move.position:
                return False
            return True

        def check_no_overlap() -> bool:
            """Check that capturing piece is not visiting previously visited pos in capturing path
            """
            assert self.visited is not None
            _to_row, _to_col = to.to_coords()
            if self.visited[_to_row][_to_col] == 1:
                return False
            return True

        def check_no_same_dir() -> bool:
            """Check that capturing piece is not moving twice in the same direction
            """
            assert self.last_capture is not None
            return move.direction != self.last_capture[1]

        if move.move_type == MoveType.PAIKA:
            valid = all(
                test() for test in [check_bounds, check_valid_dir, check_move_to_empty,]
            )
        elif (
            move.move_type != MoveType.PAIKA and self.last_capture is None
        ):  # beginning of capturing sequence
            valid = all(
                test()
                for test in [
                    check_bounds,
                    check_valid_dir,
                    check_move_to_empty,
                    check_opposite_color_capture,
                ]
            )
        else:  # in capturing sequence
            valid = all(
                test()
                for test in [
                    check_bounds,
                    check_valid_dir,
                    check_move_to_empty,
                    check_opposite_color_capture,
                    check_move_only_capturing_piece,
                    check_no_overlap,
                    check_no_same_dir,
                ]
            )
        return valid

    @property
    def legal_moves(self) -> List[int]:
        """Return a list of legal actions allowed from the current state. Actions are in their
        integer encoding
        """
        legal_captures: List[FanoronaMove] = []
        legal_paikas: List[FanoronaMove] = []

        # check for captures involving last moved piece only if in capturing sequence
        if self.last_capture:
            pos = self.last_capture[0]
            for direction in Direction:
                for capture_type in [MoveType.APPROACH, MoveType.WITHDRAWAL]:
                    capture = FanoronaMove(pos, direction, capture_type, False)
                    if self.is_valid(capture):
                        legal_captures.append(capture)

            # add end turn
            legal_captures.append(END_TURN)

        # check for captures
        for pos in Position.pos_range():
            if self.get_piece(pos) == self.turn_to_play:
                for direction in Direction:
                    for capture_type in [MoveType.APPROACH, MoveType.WITHDRAWAL]:
                        capture = FanoronaMove(pos, direction, capture_type, False)
                        if self.is_valid(capture):
                            legal_captures.append(capture)

        # only check for paikas if no captures exist
        if not legal_captures:
            for pos in Position.pos_range():
                if self.get_piece(pos) == self.turn_to_play:
                    for direction in Direction:
                        paika = FanoronaMove(pos, direction, MoveType.PAIKA, False)
                        if self.is_valid(paika):
                            legal_paikas.append(paika)

        if legal_captures:  # capture has to be made if available
            return list(map(lambda move: move.to_action(), legal_captures))
        else:
            return list(map(lambda move: move.to_action(), legal_paikas))
