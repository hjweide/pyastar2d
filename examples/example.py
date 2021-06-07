import numpy as np
import pyastar2d


# The start and goal coordinates are in matrix coordinates (i, j).
start = (0, 0)
goal = (4, 4)

# The minimum cost must be 1 for the heuristic to be valid.
weights = np.array([[1, 3, 3, 3, 3],
                    [2, 1, 3, 3, 3],
                    [2, 2, 1, 3, 3],
                    [2, 2, 2, 1, 3],
                    [2, 2, 2, 2, 1]], dtype=np.float32)
print("Cost matrix:")
print(weights)
path = pyastar2d.astar_path(weights, start, goal, allow_diagonal=True)

# The path is returned as a numpy array of (i, j) coordinates.
print(f"Shortest path from {start} to {goal} found:")
print(path)
