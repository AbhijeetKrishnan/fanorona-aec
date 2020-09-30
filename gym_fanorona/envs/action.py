from typing import List

from .constants import BOARD_SQUARES, MOVE_LIMIT
from .enums import Direction, Piece
from .position import Position
from .state import FanoronaState


class FanoronaMove:
    def __init__(self, position: Position, direction: Direction, capture_type: int, end_turn: bool):
        self.position = position
        self.direction = direction
        self.capture_type = capture_type
        self.end_turn = end_turn

    def __repr__(self):
        return f'<Action: position={str(self.position.pos)}, direction={str(self.direction)}, capture_type={capture_type}, end_turn={end_turn}>'
    
    def __str__(self):
        return f'{self.position.to_human()}{self.direction.value}{self.capture_type}{int(self.end_turn)}'

    def is_valid(self, state: FanoronaState, skip: List[str] = []) -> bool:

        to = self.position.displace(self.direction)
        if self.capture_type == 0: # none
            capture = Position(BOARD_SQUARES)
        elif self.capture_type == 1: # approach
            capture = to.displace(self.direction)
        else: # withdraw
            capture = self.position.displace(self.direction.opposite())

        def half_move_rule() -> bool:
            """Check that move number is under limit. An action cannot be performed if limit has been reached."""
            return state.half_moves < MOVE_LIMIT

        def end_turn_rule():
            """
            End turn must be done during a capturing sequence, indicated by last_dir not being 
            Direction.X. Ignore all other parameters of the action
            """
            if self.end_turn:
                return state.in_capturing_seq()
            return True

        def bounds_checking() -> bool:
            """Bounds checking on positions"""
            if self.end_turn:
                return True
            for pos in (self.position, to):
                if not pos.is_valid(): # pos is within board bounds
                    return False
            if not 0 <= capture.pos <= BOARD_SQUARES: # capture may be BOARD_SQUARES in case of paika move
                return False 
            if self.position == to:
                return False
            return True

        def check_piece_validity() -> bool:
            """Checking validity of pieces at action positions"""
            if self.end_turn:
                return True
            if state.get_piece(self.position) != state.who_to_play: # from position must contain a piece 
                return False
            if state.get_piece(to) != Piece.EMPTY: # piece must be played to an empty location
                return False
            if capture.pos != BOARD_SQUARES and state.get_piece(capture) != state.other_side(): # capturing line must start with opponent color stone
                return False
            return True

        def check_valid_dir() -> bool:
            """Checking that _dir is permitted from given board position"""
            if self.end_turn:
                return True
            return self.direction in self.position.get_valid_dirs()

        def check_no_paika_when_capture() -> bool:
            """Check if paika is being played when capturing move exists, which is illegal"""
            if self.end_turn:
                return True
            return not (self.capture_type == 0 and state.capture_exists())

        def move_only_capturing_piece() -> bool:
            """
            If in a capturing sequence, check that capturing piece is the one being moved, and 
            not some other piece
            """
            if self.end_turn:
                return True
            _from_row, _from_col = self.position.to_coords()
            if state.in_capturing_seq() and state.visited[_from_row][_from_col] != 1:
                return False
            return True
        
        def check_no_overlap() -> bool:
            """Check that capturing piece is not visiting previously visited pos in capturing path"""
            if self.end_turn:
                return True
            _to_row, _to_col = to.to_coords()
            if state.visited[_to_row][_to_col] == 1:
                return False
            return True

        def check_no_same_dir() -> bool:
            """Check that capturing piece is not moving twice in the same direction"""
            if self.end_turn:
                return True
            return self.direction != state.last_dir

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
