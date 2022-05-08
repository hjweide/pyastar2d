// Please note below heuristics are experimental and only for pretty lines.
// They may not take the shortest path and require additional cpu cycles.

#include <cmath>
#include <cstddef>
#include <experimental_heuristics.h>


heuristic_ptr select_heuristic(int h) {
  switch (h) {
    case ORTHOGONAL_X:
      return orthogonal_x;
    case ORTHOGONAL_Y:
      return orthogonal_y;
    default:
      return NULL;
  }
}

// Orthogonal x (moves by x first, then half way by y)
float orthogonal_x(int i0, int j0, int i1, int j1, int i2, int j2) {
  int di = std::abs(i0 - i1);
  int dim = std::abs(i1 - i2);
  int djm = std::abs(j1 - j2);
  if (di > (dim * 0.5)) {
    return di + djm;
  } else {
    return std::abs(j0 - j1);
  }
}

// Orthogonal y (moves by y first, then half way by x)
float orthogonal_y(int i0, int j0, int i1, int j1, int i2, int j2) {
  int dj = std::abs(j0 - j1);
  int djm = std::abs(j1 - j2);
  int dim = std::abs(i1 - i2);
  if (dj > (djm * 0.5)) {
    return dj + dim;
  } else {
    return std::abs(i0 - i1);
  }
}
