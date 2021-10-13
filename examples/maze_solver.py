import argparse
import numpy as np
import imageio
import time

import pyastar2d

from os.path import basename, join


def parse_args():
    parser = argparse.ArgumentParser(
        "An example of using pyastar2d to find the solution to a maze"
    )
    parser.add_argument(
        "--input", type=str, default="mazes/maze_small.png",
        help="Path to the black-and-white image to be used as input.",
    )
    parser.add_argument(
        "--output", type=str, help="Path to where the output will be written",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = join("solns", basename(args.input))

    return args


def main():
    args = parse_args()
    maze = imageio.imread(args.input)

    if maze is None:
        print(f"No file found: {args.input}")
        return
    else:
        print(f"Loaded maze of shape {maze.shape} from {args.input}")

    grid = maze.astype(np.float32)
    grid[grid == 0] = np.inf
    grid[grid == 255] = 1

    assert grid.min() == 1, "cost of moving must be at least 1"

    # start is the first white block in the top row
    start_j, = np.where(grid[0, :] == 1)
    start = np.array([0, start_j[0]])

    # end is the first white block in the final column
    end_i, = np.where(grid[:, -1] == 1)
    end = np.array([end_i[0], grid.shape[0] - 1])

    t0 = time.time()
    # set allow_diagonal=True to enable 8-connectivity
    path = pyastar2d.astar_path(grid, start, end, allow_diagonal=False)
    dur = time.time() - t0

    if path.shape[0] > 0:
        print(f"Found path of length {path.shape[0]} in {dur:.6f}s")
        maze = np.stack((maze, maze, maze), axis=2)
        maze[path[:, 0], path[:, 1]] = (255, 0, 0)

        print(f"Plotting path to {args.output}")
        imageio.imwrite(args.output, maze)
    else:
        print("No path found")

    print("Done")


if __name__ == "__main__":
    main()
