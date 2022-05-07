import numpy as np
import pytest

import os
import sys
import pyastar2d

from pyastar2d import Heuristic


def test_small():
    weights = np.array([[1, 3, 3, 3, 3],
                        [2, 1, 3, 3, 3],
                        [2, 2, 1, 3, 3],
                        [2, 2, 2, 1, 3],
                        [2, 2, 2, 2, 1]], dtype=np.float32)
    # Run down the diagonal.
    path = pyastar2d.astar_path(weights, (0, 0), (4, 4), allow_diagonal=True)
    expected = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])

    assert np.all(path == expected)

    # Down, right, down, right, etc.
    path = pyastar2d.astar_path(weights, (0, 0), (4, 4), allow_diagonal=False)
    expected = np.array([[0, 0], [1, 0], [1, 1], [2, 1],
                         [2, 2], [3, 2], [3, 3], [4, 3], [4, 4]])

    assert np.all(path == expected)


def test_no_solution():
    # Vertical wall.
    weights = np.ones((5, 5), dtype=np.float32)
    weights[:, 2] = np.inf

    path = pyastar2d.astar_path(weights, (0, 0), (4, 4), allow_diagonal=True)
    assert not path

    # Horizontal wall.
    weights = np.ones((5, 5), dtype=np.float32)
    weights[2, :] = np.inf

    path = pyastar2d.astar_path(weights, (0, 0), (4, 4), allow_diagonal=True)
    assert not path


def test_match_reverse():
    # Might fail if there are multiple paths, but this should be rare.
    h, w = 25, 25
    weights = (1. + 5. * np.random.random((h, w))).astype(np.float32)

    fwd = pyastar2d.astar_path(weights, (0, 0), (h - 1, w - 1))
    rev = pyastar2d.astar_path(weights, (h - 1, w - 1), (0, 0))

    assert np.all(fwd[::-1] == rev)

    fwd = pyastar2d.astar_path(weights, (0, 0), (h - 1, w - 1),
                             allow_diagonal=True)
    rev = pyastar2d.astar_path(weights, (h - 1, w - 1), (0, 0),
                             allow_diagonal=True)

    assert np.all(fwd[::-1] == rev)


def test_narrow():
    # Column weights.
    weights = np.ones((5, 1), dtype=np.float32)
    path = pyastar2d.astar_path(weights, (0, 0), (4, 0))

    expected = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])

    assert np.all(path == expected)

    # Row weights.
    weights = np.ones((1, 5), dtype=np.float32)
    path = pyastar2d.astar_path(weights, (0, 0), (0, 4))

    expected = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]])

    assert np.all(path == expected)


def test_bad_heuristic():
    # For valid heuristics, the cost to move must be at least 1.
    weights = (1. + 5. * np.random.random((10, 10))).astype(np.float32)
    # An element smaller than 1 should raise a ValueError.
    bad_cost = np.random.random() / 2.
    weights[4, 4] = bad_cost

    with pytest.raises(ValueError) as exc:
        pyastar2d.astar_path(weights, (0, 0), (9, 9))
        assert '.f' % bad_cost in exc.value.args[0]


def test_invalid_start_and_goal():
    weights = (1. + 5. * np.random.random((10, 10))).astype(np.float32)
    # Test bad start indices.
    with pytest.raises(ValueError) as exc:
        pyastar2d.astar_path(weights, (-1, 0), (9, 9))
        assert '-1' in exc.value.args[0]
    with pytest.raises(ValueError) as exc:
        pyastar2d.astar_path(weights, (10, 0), (9, 9))
        assert '10' in exc.value.args[0]
    with pytest.raises(ValueError) as exc:
        pyastar2d.astar_path(weights, (0, -1), (9, 9))
        assert '-1' in exc.value.args[0]
    with pytest.raises(ValueError) as exc:
        pyastar2d.astar_path(weights, (0, 10), (9, 9))
        assert '10' in exc.value.args[0]
    # Test bad goal indices.
    with pytest.raises(ValueError) as exc:
        pyastar2d.astar_path(weights, (0, 0), (-1, 9))
        assert '-1' in exc.value.args[0]
    with pytest.raises(ValueError) as exc:
        pyastar2d.astar_path(weights, (0, 0), (10, 9))
        assert '10' in exc.value.args[0]
    with pytest.raises(ValueError) as exc:
        pyastar2d.astar_path(weights, (0, 0), (0, -1))
        assert '-1' in exc.value.args[0]
    with pytest.raises(ValueError) as exc:
        pyastar2d.astar_path(weights, (0, 0), (0, 10))
        assert '10' in exc.value.args[0]


def test_bad_weights_dtype():
    weights = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=np.float64)
    with pytest.raises(AssertionError) as exc:
        pyastar2d.astar_path(weights, (0, 0), (2, 2))
    assert "float64" in exc.value.args[0]


def test_orthogonal_x():
    weights = np.ones((5, 5), dtype=np.float32)
    path = pyastar2d.astar_path(weights, (0, 0), (4, 4), allow_diagonal=False, heuristic_override=Heuristic.ORTHOGONAL_X)
    expected = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [3, 4], [4, 4]])

    assert np.all(path == expected)
    
    
def test_orthogonal_y():
    weights = np.ones((5, 5), dtype=np.float32)
    path = pyastar2d.astar_path(weights, (0, 0), (4, 4), allow_diagonal=False, heuristic_override=Heuristic.ORTHOGONAL_Y)
    expected = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4]])

    assert np.all(path == expected)
