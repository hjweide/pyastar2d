[![Build Status](https://travis-ci.com/hjweide/pyastar2d.svg?branch=master)](https://travis-ci.com/hjweide/pyastar2d)
[![Coverage Status](https://coveralls.io/repos/github/hjweide/pyastar2d/badge.svg?branch=master)](https://coveralls.io/github/hjweide/pyastar2d?branch=master)
[![PyPI version](https://badge.fury.io/py/pyastar2d.svg)](https://badge.fury.io/py/pyastar2d)
# PyAstar2D
This is a very simple C++ implementation of the A\* algorithm for pathfinding
on a two-dimensional grid.  The solver itself is implemented in C++, but is
callable from Python.  This combines the speed of C++ with the convenience of
Python.

I have not done any formal benchmarking, but the solver finds the solution to a
1802 by 1802 maze in 0.29s and a 4008 by 4008 maze in 0.83s when running on my
nine-year-old Intel(R) Core(TM) i7-2630QM CPU @ 2.00GHz.  See [Example
Results](#example-results) for more details.

See `src/cpp/astar.cpp` for the core C++ implementation of the A\* shortest
path search algorithm, `src/pyastar2d/astar_wrapper.py` for the Python wrapper
and `examples/example.py` for example usage.

When determining legal moves, 4-connectivity is the default, but it is possible
to set `allow_diagonal=True` for 8-connectivity.

## Installation
Instructions for installing `pyastar2d` are given below.

### From PyPI
The easiest way to install `pyastar2d` is directly from the Python package index:
```
pip install pyastar2d
```

### From source
You can also install `pyastar2d` by cloning this repository and building it
yourself.  If running on Linux or MacOS, simply run
```bash
pip install .
````
from the root directory.  If you are using Windows you may have to install Cython manually first:
```bash
pip install Cython
pip install .
```
To check that everything worked, run the example:
```bash
python examples/example.py
```

### As a dependency
Include `pyastar2d` in your `requirements.txt` to install from `pypi`, or add
this line to `requirements.txt`:
```
pyastar2d @ git+git://github.com/hjweide/pyastar2d.git@master#egg=pyastar2d
```

## Usage
A simple example is given below:
```python
import numpy as np
import pyastar2d
# The minimum cost must be 1 for the heuristic to be valid.
# The weights array must have np.float32 dtype to be compatible with the C++ code.
weights = np.array([[1, 3, 3, 3, 3],
                    [2, 1, 3, 3, 3],
                    [2, 2, 1, 3, 3],
                    [2, 2, 2, 1, 3],
                    [2, 2, 2, 2, 1]], dtype=np.float32)
# The start and goal coordinates are in matrix coordinates (i, j).
path = pyastar2d.astar_path(weights, (0, 0), (4, 4), allow_diagonal=True)
print(path)
# The path is returned as a numpy array of (i, j) coordinates.
array([[0, 0],
       [1, 1],
       [2, 2],
       [3, 3],
       [4, 4]])
```
Note that all grid points are represented as `(i, j)` coordinates.  An example
of using `pyastar2d` to solve a maze is given in `examples/maze_solver.py`.

## Example Results
<a name="example-results"></a>
To test the implementation, I grabbed two nasty mazes from Wikipedia.  They are
included in the ```mazes``` directory, but are originally from here:
[Small](https://upload.wikimedia.org/wikipedia/commons/c/cf/MAZE.png) and
[Large](https://upload.wikimedia.org/wikipedia/commons/3/32/MAZE_2000x2000_DFS.png).
I load the ```.png``` files as grayscale images, and set the white pixels to 1
(open space) and the black pixels to `INF` (walls).

To run the examples specify the input and output files using the `--input` and
`--output` flags.  For example, the following commands will solve the small and
large mazes:
```
python examples/maze_solver.py --input mazes/maze_small.png --output solns/maze_small.png
python examples/maze_solver.py --input mazes/maze_large.png --output solns/maze_large.png
```

### Small Maze (1802 x 1802): 
```bash
time python examples/maze_solver.py --input mazes/maze_small.png --output solns/maze_small.png
Loaded maze of shape (1802, 1802) from mazes/maze_small.png
Found path of length 10032 in 0.292794s
Plotting path to solns/maze_small.png
Done

real	0m1.214s
user	0m1.526s
sys	0m0.606s
```
The solution found for the small maze is shown below:
<img src="https://github.com/hjweide/pyastar2d/raw/master/solns/maze_small_soln.png" alt="Maze Small Solution" style="width: 100%"/>

### Large Maze (4002 x 4002): 
```bash
time python examples/maze_solver.py --input mazes/maze_large.png --output solns/maze_large.png
Loaded maze of shape (4002, 4002) from mazes/maze_large.png
Found path of length 783737 in 0.829181s
Plotting path to solns/maze_large.png
Done

real	0m29.385s
user	0m29.563s
sys	0m0.728s
```
The solution found for the large maze is shown below:
<img src="https://github.com/hjweide/pyastar2d/raw/master/solns/maze_large_soln.png" alt="Maze Large Solution" style="width: 100%"/>

## Motivation
I recently needed an implementation of the A* algorithm in Python to find the
shortest path between two points in a cost matrix representing an image.
Normally I would simply use [networkx](https://networkx.github.io/), but for
graphs with millions of nodes the overhead incurred to construct the graph can
be expensive.  Considering that I was only interested in graphs that may be
represented as two-dimensional grids, I decided to implement it myself using
this special structure of the graph to make various optimizations.
Specifically, the graph is represented as a one-dimensional array because there
is no need to store the neighbors.  Additionally, the lookup tables for
previously-explored nodes (their costs and paths) are also stored as
one-dimensional arrays.  The implication of this is that checking the lookup
table can be done in O(1), at the cost of using O(n) memory.  Alternatively, we
could store only the nodes we traverse in a hash table to reduce the memory
usage.  Empirically I found that replacing the one-dimensional array with a
hash table (`std::unordered_map`) was about five times slower.

## Tests
The default installation does not include the dependencies necessary to run the
tests.  To install these, first run
```bash
pip install -r requirements-dev.txt
```
before running
```bash
py.test
```
The tests are fairly basic but cover some of the
more common pitfalls.  Pull requests for more extensive tests are welcome.

## References
1. [A\* search algorithm on Wikipedia](https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode)
2. [Pathfinding with A* on Red Blob Games](http://www.redblobgames.com/pathfinding/a-star/introduction.html)

