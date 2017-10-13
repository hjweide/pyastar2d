#include <queue>
#include <limits>
#include <cmath>

// represents a single pixel
class Node {
  public:
    int idx;     // index in the flattened grid
    float cost;  // cost of traversing this pixel

    Node(int i, float c) : idx(i),cost(c) {}
};

// the top of the priority queue is the greatest element by default,
// but we want the smallest, so flip the sign
bool operator<(const Node &n1, const Node &n2) {
  return n1.cost > n2.cost;
}

bool operator==(const Node &n1, const Node &n2) {
  return n1.idx == n2.idx;
}

// manhattan distance: requires each move to cost >= 1
float heuristic(int i0, int j0, int i1, int j1) {
  return std::abs(i0 - i1) + std::abs(j0 - j1);
}

// weights:        flattened h x w grid of costs
// start, goal:    index of start/goal in flattened grid
// paths (output): for each node, stores previous node in path
extern "C" bool astar(
      const float* weights, const int height, const int width,
      const int start, const int goal,
      int* paths) {

  const float INF = std::numeric_limits<float>::infinity();

  Node start_node(start, 0.);
  Node goal_node(goal, 0.);

  float* costs = new float[height * width];
  for (int i = 0; i < height * width; ++i)
    costs[i] = INF;
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);

  int* nbrs = new int[4];

  bool solution_found = false;
  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

    if (cur == goal_node) {
      solution_found = true;
      break;
    }

    nodes_to_visit.pop();

    // check bounds and find up to four neighbors
    nbrs[0] = (cur.idx / width > 0) ? (cur.idx - width) : -1;
    nbrs[1] = (cur.idx % width > 0) ? (cur.idx - 1) : -1;
    nbrs[2] = (cur.idx / width + 1 < height) ? (cur.idx + width) : -1;
    nbrs[3] = (cur.idx % width + 1 < width) ? (cur.idx + 1) : -1;
    for (int i = 0; i < 4; ++i) {
      if (nbrs[i] >= 0) {
        // the sum of the cost so far and the cost of this move
        float new_cost = costs[cur.idx] + weights[nbrs[i]];
        if (new_cost < costs[nbrs[i]]) {
          costs[nbrs[i]] = new_cost;
          float priority = new_cost + heuristic(nbrs[i] / width,
                                                nbrs[i] % width,
                                                goal / width,
                                                goal % width);
          // paths with lower expected cost are explored first
          nodes_to_visit.push(Node(nbrs[i], priority));
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }

  delete[] costs;
  delete[] nbrs;

  return solution_found;
}
