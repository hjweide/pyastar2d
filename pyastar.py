import ctypes
import numpy as np

import inspect
from os.path import abspath, dirname, join

fname = abspath(inspect.getfile(inspect.currentframe()))
lib = ctypes.cdll.LoadLibrary(join(dirname(fname), 'astar.so'))

astar = lib.astar
ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
ndmat_i_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
astar.restype = ctypes.c_bool
astar.argtypes = [ndmat_f_type, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int, ctypes.c_bool,
                  ndmat_i_type]


def astar_path(weights, start, goal, allow_diagonal=False):
    # For the heuristic to be valid, each move must cost at least 1.
    if weights.min(axis=None) < 1.:
        raise ValueError('Minimum cost to move must be 1, but got %f' % (
            weights.min(axis=None)))
    # Ensure start is within bounds.
    if (start[0] < 0 or start[0] >= weights.shape[0] or
            start[1] < 0 or start[1] >= weights.shape[1]):
        raise ValueError('Start of (%d, %d) lies outside grid.' % (start))
    # Ensure goal is within bounds.
    if (goal[0] < 0 or goal[0] >= weights.shape[0] or
            goal[1] < 0 or goal[1] >= weights.shape[1]):
        raise ValueError('Goal of (%d, %d) lies outside grid.' % (goal))

    height, width = weights.shape
    start_idx = np.ravel_multi_index(start, (height, width))
    goal_idx = np.ravel_multi_index(goal, (height, width))

    # The C++ code writes the solution to the paths array
    paths = np.full(height * width, -1, dtype=np.int32)
    success = astar(
        weights.flatten(), height, width, start_idx, goal_idx, allow_diagonal,
        paths  # output parameter
    )
    if not success:
        return np.array([])

    coordinates = []
    path_idx = goal_idx
    while path_idx != start_idx:
        pi, pj = np.unravel_index(path_idx, (height, width))
        coordinates.append((pi, pj))

        path_idx = paths[path_idx]

    if coordinates:
        coordinates.append(np.unravel_index(start_idx, (height, width)))
        return np.vstack(coordinates[::-1])
    else:
        return np.array([])
