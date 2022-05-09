// Please note below heuristics are experimental and only for pretty lines.
// They may not take the shortest path and require additional cpu cycles.

#ifndef EXPERIMENTAL_HEURISTICS_H_
#define EXPERIMENTAL_HEURISTICS_H_


enum Heuristic { DEFAULT, ORTHOGONAL_X, ORTHOGONAL_Y };

typedef float (*heuristic_ptr)(int, int, int, int, int, int);

heuristic_ptr select_heuristic(int);

// Orthogonal x (moves by x first, then half way by y)
float orthogonal_x(int, int, int, int, int, int);

// Orthogonal y (moves by y first, then half way by x)
float orthogonal_y(int, int, int, int, int, int);

#endif
