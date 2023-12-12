from typing import List, Literal, NamedTuple, Tuple, TypeAlias, Union

import numpy as np

from .fanorona_move import END_TURN, ActionType, FanoronaMove, MoveType
from .utils import (
    BOARD_COLS,
    BOARD_ROWS,
    MOVE_LIMIT,
    Direction,
    Piece,
    Position,
)

AgentId: TypeAlias = str


class LastCapture(NamedTuple):
    position: Position
    direction: Direction

    def __repr__(self) -> str:
        return f"<LastCapture: {str(self)}>"

    def __str__(self) -> str:
        return f"{self.position.to_human()} {str(self.direction)}"


class FanoronaState:
    def __init__(self) -> None:
        """
        Initializes the Fanorona state.

        Parameters:
            None

        Returns:
            None
        """

        self.board: np.ndarray[
            Tuple[Literal[5], Literal[9]], np.dtype[np.int8]
        ] | None = None
        self.turn_to_play: Piece = Piece.EMPTY
        self.last_capture: LastCapture | None = None
        self.visited: np.ndarray[
            Tuple[Literal[5], Literal[9]], np.dtype[np.bool_]
        ] | None = None
        self.half_moves: int = 0

    @property
    def visited_pos(self) -> List[Position]:
        if self.visited is not None:
            visited_pos_list = [
                Position(int(visited_pos))
                for visited_pos in np.flatnonzero(self.visited)
            ]
        else:
            visited_pos_list = []
        return visited_pos_list

    def __repr__(self) -> str:
        """
        Returns a string representation of the FanoronaState object.

        Returns:
            str: A string representation of the FanoronaState object.
        """
        return f"<FanoronaState: {str(self)}>"

    def __str__(self) -> str:
        """
        Returns a string representation of the Fanorona game state in a FEN-like
        notation

        Returns:
            str: A string representation of the Fanorona game state.
        """
        if self.board is None:
            return ""

        def row_str(row: np.ndarray[Literal[9], np.dtype[np.int8]]) -> str:
            "String for each row"
            row_ele: List[Union[int, Piece]] = []
            for col in row:
                if (
                    len(row_ele) > 0
                    and not isinstance(row_ele[-1], Piece)
                    and col == Piece.EMPTY
                ):
                    row_ele[-1] += 1
                else:
                    if col == Piece.EMPTY:
                        row_ele.append(1)
                    else:
                        row_ele.append(Piece(col))
            row_str = "".join([str(ele) for ele in row_ele])
            return row_str

        board_pieces_str = "/".join([row_str(row) for row in self.board])

        turn_to_play_str = str(Piece(self.turn_to_play))

        last_capture_str = (
            str(self.last_capture) if self.last_capture else "- -"
        )

        assert self.visited is not None
        visited_pos_list = [
            visited_pos.to_human() for visited_pos in self.visited_pos
        ]
        if len(visited_pos_list) == 0:
            visited_pos_str = "-"
        else:
            visited_pos_str = ",".join(visited_pos_list)

        return f"{board_pieces_str} {turn_to_play_str} {last_capture_str} {visited_pos_str} {str(self.half_moves)}"

    def as_rich_board(self) -> str:
        """
        Returns the current state of the Fanorona game board as a rich board.

        Returns:
            str: The rich board representation of the game board.

        Raises:
            Exception: If the board is None.
        """
        ELE_MAP = {Piece.WHITE: "○", Piece.BLACK: "●", Piece.EMPTY: "."}
        if self.board is None:
            raise Exception(
                'render(mode="human") called without calling reset()'
            )
        rich_board = np.vectorize(ELE_MAP.get)(self.board)
        template = f"""  A B C D E F G H I
{5} {'─'.join(rich_board[4])}
  │╲│╱│╲│╱│╲│╱│╲│╱│
{4} {'─'.join(rich_board[3])}
  │╱│╲│╱│╲│╱│╲│╱│╲│
{3} {'─'.join(rich_board[2])}
  │╲│╱│╲│╱│╲│╱│╲│╱│
{2} {'─'.join(rich_board[1])}
  │╱│╲│╱│╲│╱│╲│╱│╲│
{1} {'─'.join(rich_board[0])}

{self.turn_to_play} to play
Last capture: {str(self.last_capture) if self.last_capture else "- -"}
Visited: {', '.join([pos.to_human()
                     for pos in self.visited_pos])}
Half-moves: {self.half_moves}
"""
        return template

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FanoronaState):
            return NotImplemented
        return str(self) == str(other)

    def to_svg(self, svg_w: int = 1000, svg_h: int = 600) -> str:
        """
        Converts the current state of the Fanorona game board to an SVG format.

        Args:
            svg_w (int): The width of the SVG output (default is 1000).
            svg_h (int): The height of the SVG output (default is 600).

        Returns:
            str: The SVG representation of the game board.

        Raises:
            Exception: If the board or visited state is None.

        TODO:
            - Adjust output SVG size dynamically.
            - Represent other aspects of state on the output SVG (turn to play, last capture, visited, etc.).
        """

        def convert(coord: Tuple[int, int]) -> Tuple[int, int]:
            row, col = coord
            return 100 + col * 100, 100 + (4 - row) * 100

        if self.board is None or self.visited is None:
            raise Exception(
                'render(mode="svg") called without calling reset()'
            )

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
                [
                    line.format(convert(f), convert(t))
                    for f, t in zip(_from_df, _to_df)
                ]
            )
        for _ in range(0, BOARD_COLS, 2):  # diagonal backward lines
            _from_db = [(2, 0), (4, 0), (4, 2), (4, 4), (4, 6)]
            _to_db = [(0, 2), (0, 4), (0, 6), (0, 8), (2, 8)]
            board_lines.extend(
                [
                    line.format(convert(f), convert(t))
                    for f, t in zip(_from_db, _to_db)
                ]
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

    def push(self, move: FanoronaMove) -> None:
        """
        Implement the rules of Fanorona and make the desired move on the board. Returns flags and
        status codes depending on game over or draw.

        Args:
            move (FanoronaMove): The move to be made on the board.

        Returns:
            None

        Raises:
            Exception: If `reset()` method is not called before calling `push()`.

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

        def end_turn() -> None:
            self.turn_to_play = self.turn_to_play.other()
            self.last_capture = None
            assert self.visited is not None
            self.visited.fill(0)  # reset visited ndarray
            self.half_moves += 1

        if move.end_turn or move.move_type == MoveType.PAIKA:
            end_turn()
        else:
            match move.move_type:
                case MoveType.APPROACH:
                    capture_pos = to.displace(move.direction)
                    capture_dir = move.direction
                case MoveType.WITHDRAWAL:
                    capture_pos = move.position.displace(
                        move.direction.opposite()
                    )
                    capture_dir = move.direction.opposite()
                case _:
                    raise ValueError(
                        f"Unexpected move type encountered: \
                                     {move.move_type}"
                    )

            capture_row, capture_col = capture_pos.to_coords()
            while (
                capture_pos.is_valid()
                and self.board[capture_row][capture_col]
                == self.turn_to_play.other()
            ):
                self.board[capture_row][capture_col] = Piece.EMPTY
                capture_pos = capture_pos.displace(capture_dir)
                capture_row, capture_col = capture_pos.to_coords()

            self.last_capture = LastCapture(
                position=to, direction=move.direction
            )
            self.visited[from_row][from_col] = 1
            self.visited[to_row][to_col] = 1

            # if in capturing sequence, and no valid moves available (other than
            # end turn), then force turn to end
            # if len(self.legal_moves) == 1:
            #     end_turn()

    @property
    def done(self) -> bool:
        """
        Check whether the game is over (i.e. the current state is a terminal state).

        The game is over when -
        a) One side has no pieces left to move (loss for the side which has no pieces to move)
        b) The number of half-moves exceeds the limit (draw)

        Returns:
            A tuple containing a boolean value indicating whether the game is over,
            and the piece that has won the game or an empty piece if the game is not over yet.
        """
        if self.half_moves >= MOVE_LIMIT:
            return True
        else:
            own_piece_exists = self.piece_exists(self.turn_to_play)
            other_piece_exists = self.piece_exists(self.turn_to_play.other())
            # Conjecture: cannot have a situation in Fanorona where a piece exists but there are no
            # valid moves
            return not (own_piece_exists and other_piece_exists)

    @property
    def winner(self) -> Piece | None:
        """
        Determines the winner of the game.

        Returns:
            Piece | None: The winning player's piece if there is a winner, None otherwise.
        """

        if self.done:
            if self.half_moves >= MOVE_LIMIT:
                return None  # draw by half-move rule
            else:
                own_piece_exists = self.piece_exists(self.turn_to_play)
                other_piece_exists = self.piece_exists(
                    self.turn_to_play.other()
                )
                if own_piece_exists and other_piece_exists:
                    return None  # game not over
                else:
                    return self.turn_to_play
        else:
            return None  # game not over

    def reset(self) -> None:
        """
        Reset the state of the Fanorona game to the start state.

        This method sets the state of the game board to the initial configuration.
        """
        START_STATE_STR = (
            "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - - 0"
        )
        self.set_from_board_str(START_STATE_STR)

    def set_from_board_str(self, board_string: str) -> "FanoronaState":
        """
        Set the state object to a new state represented by a board string.

        Args:
            board_string (str): The board string representing the new state.

        Returns:
            FanoronaState: The updated state object.
        """

        def process_board_state_str(
            self, board_state_str: str
        ) -> np.ndarray[Tuple[Literal[5], Literal[9]], np.dtype[np.int8]]:
            row_strings = board_state_str.split("/")
            board_state_chars = [list(row) for row in row_strings]
            self.board = np.zeros(
                shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8
            )
            for row, row_content in enumerate(board_state_chars):
                col_board = 0
                for col_content in row_content:
                    match col_content:
                        case "W":
                            self.board[row][col_board] = Piece.WHITE
                            col_board += 1
                        case "B":
                            self.board[row][col_board] = Piece.BLACK
                            col_board += 1
                        case _:
                            self.board[row][
                                col_board : col_board + int(col_content)
                            ] = Piece.EMPTY
                            col_board += int(col_content)
            return self.board

        def process_visited_pos_str(
            self, visited_pos_str: str
        ) -> np.ndarray[Tuple[Literal[5], Literal[9]], np.dtype[np.bool_]]:
            self.visited = np.zeros(
                shape=(BOARD_ROWS, BOARD_COLS), dtype=np.bool_
            )
            if visited_pos_str != "-":
                visited_pos_list = visited_pos_str.split(",")
                human_pos_list = list(
                    map(
                        lambda human_pos: Position(human_pos).to_coords(),
                        visited_pos_list,
                    )
                )
                rows, cols = zip(*human_pos_list)
                self.visited[rows, cols] = True
            return self.visited

        (
            board_state_str,
            turn_to_play_str,
            last_capture_pos,
            last_capture_dir,
            visited_pos_str,
            half_moves_str,
        ) = board_string.split()

        process_board_state_str(self, board_state_str)

        self.turn_to_play = (
            Piece.WHITE if turn_to_play_str == "W" else Piece.BLACK
        )

        if last_capture_pos != "-" and last_capture_dir != "-":
            self.last_capture = LastCapture(
                position=Position(last_capture_pos),
                direction=Direction.from_str(last_capture_dir),
            )
        else:
            self.last_capture = None

        process_visited_pos_str(self, visited_pos_str)

        self.half_moves = int(half_moves_str)

        return self

    def get_observation(
        self, agent: AgentId
    ) -> np.ndarray[
        Tuple[Literal[5], Literal[9], Literal[8]], np.dtype[np.int8]
    ]:
        """Return NN-style observation based on the current board state and requesting agent. Board
        state is from the perspective of the agent, with their color at the bottom.
        """
        if self.board is None or self.visited is None:
            raise Exception("Called get_observation() without calling reset()")

        obs = np.zeros(shape=(5, 9, 8), dtype=np.int8)
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
            raise Exception(
                f"Called is_valid({str(move)}) without calling reset()"
            )

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
            return all(
                map(lambda pos: pos.is_valid(), (move.position, to, capture))
            )

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
            """Check that capturing piece is not visiting previously visited pos in capturing path"""
            assert self.visited is not None
            _to_row, _to_col = to.to_coords()
            if self.visited[_to_row][_to_col] == 1:
                return False
            return True

        def check_no_same_dir() -> bool:
            """Check that capturing piece is not moving twice in the same direction"""
            assert self.last_capture is not None
            return move.direction != self.last_capture[1]

        if move.move_type == MoveType.PAIKA:
            valid = all(
                test()
                for test in [
                    check_bounds,
                    check_valid_dir,
                    check_move_to_empty,
                ]
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
    def legal_moves(self) -> List[ActionType]:
        """Return a list of legal actions allowed from the current state."""
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
                    for capture_type in [
                        MoveType.APPROACH,
                        MoveType.WITHDRAWAL,
                    ]:
                        capture = FanoronaMove(
                            pos, direction, capture_type, False
                        )
                        if self.is_valid(capture):
                            legal_captures.append(capture)

        # only check for paikas if no captures exist
        if not legal_captures:
            for pos in Position.pos_range():
                if self.get_piece(pos) == self.turn_to_play:
                    for direction in Direction:
                        paika = FanoronaMove(
                            pos, direction, MoveType.PAIKA, False
                        )
                        if self.is_valid(paika):
                            legal_paikas.append(paika)

        if legal_captures:  # capture has to be made if available
            legal_moves = legal_captures
        else:
            legal_moves = legal_paikas
        return list(map(lambda move: move.to_action(), legal_moves))
