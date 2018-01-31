import cv2
import numpy as np

import pyastar

from time import time
from os.path import basename, join, splitext

# input/output files
MAZE_FPATH = join('mazes', 'maze_small.png')
#MAZE_FPATH = join('mazes', 'maze_large.png')
OUTP_FPATH = join('solns', '%s_soln.png' % splitext(basename(MAZE_FPATH))[0])


def main():
    maze = cv2.imread(MAZE_FPATH)
    if maze is None:
        print('no file found: %s' % (MAZE_FPATH))
        return
    else:
        print('loaded maze of shape %r' % (maze.shape[0:2],))

    grid = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grid[grid == 0] = np.inf
    grid[grid == 255] = 1

    assert grid.min() == 1, 'cost of moving must be at least 1'

    # start is the first white block in the top row
    start_j, = np.where(grid[0, :] == 1)
    start = np.array([0, start_j[0]])

    # end is the first white block in the final column
    end_i, = np.where(grid[:, -1] == 1)
    end = np.array([end_i[0], grid.shape[0] - 1])

    t0 = time()
    # set allow_diagonal=True to enable 8-connectivity
    path = pyastar.astar_path(grid, start, end, allow_diagonal=False)
    dur = time() - t0

    if path.shape[0] > 0:
        print('found path of length %d in %.6fs' % (path.shape[0], dur))
        maze[path[:, 0], path[:, 1]] = (0, 0, 255)

        print('plotting path to %s' % (OUTP_FPATH))
        cv2.imwrite(OUTP_FPATH, maze)
    else:
        print('no path found')

    print('done')


if __name__ == '__main__':
    main()
