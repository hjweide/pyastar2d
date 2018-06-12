# A\*
This is a very simple C++ implementation of the A\* algorithm for pathfinding
on a two-dimensional grid.  The compiled ```astar.so``` file is callable from Python.  See ```pyastar.py``` for the Python wrapper and
```examples.py``` for example usage.  Uses 4-connectivity by default, set ```allow_diagonal=True``` for 8-connectivity.

## Motivation
I recently needed an implementation of the A* algorithm in Python.
Normally I would simply use [networkx](https://networkx.github.io/),
but for graphs with millions of nodes the overhead incurred to
construct the graph can be expensive.  Considering that my use case was
so simple, I decided to implement it myself.

## Usage
Run ```make``` to build the shared object file ```astar.so```.
``` python
import numpy as np
import pyastar
# The minimum cost must be 1 for the heuristic to be valid.
weights = np.array([[1, 3, 3, 3, 3],
                    [2, 1, 3, 3, 3],
                    [2, 2, 1, 3, 3],
                    [2, 2, 2, 1, 3],
                    [2, 2, 2, 2, 1]], dtype=np.float32)
# The start and goal coordinates are in matrix coordinates (i, j).
path = pyastar.astar_path(weights, (0, 0), (4, 4), allow_diagonal=True)
print(path)
# The path is returned as a numpy array of (i, j) coordinates.
array([[0, 0],
       [1, 1],
       [2, 2],
       [3, 3],
       [4, 4]])
```

## Example Results
To test the implementation, I grabbed two nasty mazes from Wikipedia.  They are
included in the ```mazes``` directory, but are originally from here:
[Small](https://upload.wikimedia.org/wikipedia/commons/c/cf/MAZE.png) and
[Large](https://upload.wikimedia.org/wikipedia/commons/3/32/MAZE_2000x2000_DFS.png).
I load the ```.png``` files as grayscale images, and set the white pixels to 1
(open space) and the black pixels to ```INF``` (walls).

To run the examples:
1. Run ```make``` to build the shared object file ```astar.so```.
2. Set the ```MAZE_FPATH``` and ```OUTP_FPATH``` as desired in ```examples.py```.
3. Run ```python examples.py```.

Output for the small maze:
```
time python examples.py
loaded maze of shape (1802, 1802)
found path of length 10032 in 0.258270s
plotting path to solns/maze_small_soln.png
done

real  0m2.319s
user  0m0.403s
sys 0m1.691s
```

The solution is visualized below:
<img src="solns/maze_small_soln.png" alt="Maze Small Solution" style="width: 100%"/>

Output for the large maze:
```
time python examples.py
loaded maze of shape (4002, 4002)
found path of length 783737 in 3.886067s
plotting path to solns/maze_large_soln.png
done

real  0m6.495s
user  0m4.007s
sys 0m2.273s
```

The solution is visualized below:
<img src="solns/maze_large_soln.png" alt="Maze Large Solution" style="width: 100%"/>

## Tests
To run the tests, simply run ```py.test``` in the ```tests``` directory.
```
cd tests
py.test
```
The tests are fairly basic but cover some of the more common pitfalls.  Pull
requests for more extensive tests are welcome.

## References
1. [A\* search algorithm on Wikipedia](https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode)
2. [Pathfinding with A* on Red Blob Games](http://www.redblobgames.com/pathfinding/a-star/introduction.html)

