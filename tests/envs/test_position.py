from gym_fanorona.envs import Position
import pytest


def test_convert_coords_to_human():
    """Test that converting to human-readable coords upon initialization using row-col coords returns 
    the correct values for all valid board coords"""
    answer = {
        (0, 0): "A1",
        (0, 1): "B1",
        (0, 2): "C1",
        (0, 3): "D1",
        (0, 4): "E1",
        (0, 5): "F1",
        (0, 6): "G1",
        (0, 7): "H1",
        (0, 8): "I1",
        (1, 0): "A2",
        (1, 1): "B2",
        (1, 2): "C2",
        (1, 3): "D2",
        (1, 4): "E2",
        (1, 5): "F2",
        (1, 6): "G2",
        (1, 7): "H2",
        (1, 8): "I2",
        (2, 0): "A3",
        (2, 1): "B3",
        (2, 2): "C3",
        (2, 3): "D3",
        (2, 4): "E3",
        (2, 5): "F3",
        (2, 6): "G3",
        (2, 7): "H3",
        (2, 8): "I3",
        (3, 0): "A4",
        (3, 1): "B4",
        (3, 2): "C4",
        (3, 3): "D4",
        (3, 4): "E4",
        (3, 5): "F4",
        (3, 6): "G4",
        (3, 7): "H4",
        (3, 8): "I4",
        (4, 0): "A5",
        (4, 1): "B5",
        (4, 2): "C5",
        (4, 3): "D5",
        (4, 4): "E5",
        (4, 5): "F5",
        (4, 6): "G5",
        (4, 7): "H5",
        (4, 8): "I5",
    }
    for row, col in Position.coord_range():
        assert Position((row, col)).to_human() == answer[(row, col)]


def test_convert_human_to_coords():
    """
    Test that converting to coords upon initialization using human-reaable coords returns the 
    correct values for all valid board coords
    """
    answer = {
        "A1": (0, 0),
        "B1": (0, 1),
        "C1": (0, 2),
        "D1": (0, 3),
        "E1": (0, 4),
        "F1": (0, 5),
        "G1": (0, 6),
        "H1": (0, 7),
        "I1": (0, 8),
        "A2": (1, 0),
        "B2": (1, 1),
        "C2": (1, 2),
        "D2": (1, 3),
        "E2": (1, 4),
        "F2": (1, 5),
        "G2": (1, 6),
        "H2": (1, 7),
        "I2": (1, 8),
        "A3": (2, 0),
        "B3": (2, 1),
        "C3": (2, 2),
        "D3": (2, 3),
        "E3": (2, 4),
        "F3": (2, 5),
        "G3": (2, 6),
        "H3": (2, 7),
        "I3": (2, 8),
        "A4": (3, 0),
        "B4": (3, 1),
        "C4": (3, 2),
        "D4": (3, 3),
        "E4": (3, 4),
        "F4": (3, 5),
        "G4": (3, 6),
        "H4": (3, 7),
        "I4": (3, 8),
        "A5": (4, 0),
        "B5": (4, 1),
        "C5": (4, 2),
        "D5": (4, 3),
        "E5": (4, 4),
        "F5": (4, 5),
        "G5": (4, 6),
        "H5": (4, 7),
        "I5": (4, 8),
    }
    for human in Position.human_range():
        assert Position(human).to_coords() == answer[human]


def test_convert_coords_to_pos():
    "Test that initializing using coords returns the correct values for all valid board coords"
    answer = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (0, 4): 4,
        (0, 5): 5,
        (0, 6): 6,
        (0, 7): 7,
        (0, 8): 8,
        (1, 0): 9,
        (1, 1): 10,
        (1, 2): 11,
        (1, 3): 12,
        (1, 4): 13,
        (1, 5): 14,
        (1, 6): 15,
        (1, 7): 16,
        (1, 8): 17,
        (2, 0): 18,
        (2, 1): 19,
        (2, 2): 20,
        (2, 3): 21,
        (2, 4): 22,
        (2, 5): 23,
        (2, 6): 24,
        (2, 7): 25,
        (2, 8): 26,
        (3, 0): 27,
        (3, 1): 28,
        (3, 2): 29,
        (3, 3): 30,
        (3, 4): 31,
        (3, 5): 32,
        (3, 6): 33,
        (3, 7): 34,
        (3, 8): 35,
        (4, 0): 36,
        (4, 1): 37,
        (4, 2): 38,
        (4, 3): 39,
        (4, 4): 40,
        (4, 5): 41,
        (4, 6): 42,
        (4, 7): 43,
        (4, 8): 44,
    }
    for row, col in Position.coord_range():
        assert Position((row, col)).to_pos() == answer[(row, col)]


def test_convert_pos_to_coords():
    """
    Test that converting to coords upon initialization using pos returns the correct values for 
    all valid board positions
    """
    answer = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (0, 4): 4,
        (0, 5): 5,
        (0, 6): 6,
        (0, 7): 7,
        (0, 8): 8,
        (1, 0): 9,
        (1, 1): 10,
        (1, 2): 11,
        (1, 3): 12,
        (1, 4): 13,
        (1, 5): 14,
        (1, 6): 15,
        (1, 7): 16,
        (1, 8): 17,
        (2, 0): 18,
        (2, 1): 19,
        (2, 2): 20,
        (2, 3): 21,
        (2, 4): 22,
        (2, 5): 23,
        (2, 6): 24,
        (2, 7): 25,
        (2, 8): 26,
        (3, 0): 27,
        (3, 1): 28,
        (3, 2): 29,
        (3, 3): 30,
        (3, 4): 31,
        (3, 5): 32,
        (3, 6): 33,
        (3, 7): 34,
        (3, 8): 35,
        (4, 0): 36,
        (4, 1): 37,
        (4, 2): 38,
        (4, 3): 39,
        (4, 4): 40,
        (4, 5): 41,
        (4, 6): 42,
        (4, 7): 43,
        (4, 8): 44,
    }
    answer = {
        value: key for key, value in answer.items()
    }  # inverting the dictionary created in the previous test
    for pos in Position.pos_range():
        assert pos.to_coords() == answer[pos.to_pos()]


@pytest.mark.skip(reason="Not implemented")
def test_get_valid_dirs():
    "TODO: "
    pass


@pytest.mark.skip(reason="Not implemented")
def test_displace_pos():
    "TODO: "
    pass
