#pragma once

#include "kangaroo/runmeta.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kangaroo {

// Immutable bounding-volume tree. Storage scales with boxes, not cells, and
// point and box queries prune disjoint subtrees. Safe to share among tasks.
class BoxSpatialIndex {
 public:
  explicit BoxSpatialIndex(std::span<const BlockBox> boxes) {
    std::vector<BlockBox> sorted(boxes.begin(), boxes.end());
    nodes_.reserve(sorted.size() * 2);
    if (!sorted.empty()) {
      build(sorted, 0, sorted.size());
    }
  }

  bool intersects(const int32_t lo[3], const int32_t hi[3]) const {
    return !nodes_.empty() && intersects_node(0, lo, hi);
  }

  bool contains(const int32_t cell[3]) const { return intersects(cell, cell); }

  // Null for an empty level.
  const BlockBox *bounds() const {
    return nodes_.empty() ? nullptr : &nodes_.front().bounds;
  }

 private:
  struct Node {
    BlockBox bounds;
    std::size_t left = 0;
    std::size_t right = 0;  // zero marks a leaf
  };

  static int32_t coordinate(const Int3 &p, int axis) {
    return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
  }

  std::size_t build(std::vector<BlockBox> &boxes, std::size_t begin, std::size_t end) {
    BlockBox bounds = boxes[begin];
    for (std::size_t i = begin + 1; i < end; ++i) {
      bounds.lo.x = std::min(bounds.lo.x, boxes[i].lo.x);
      bounds.lo.y = std::min(bounds.lo.y, boxes[i].lo.y);
      bounds.lo.z = std::min(bounds.lo.z, boxes[i].lo.z);
      bounds.hi.x = std::max(bounds.hi.x, boxes[i].hi.x);
      bounds.hi.y = std::max(bounds.hi.y, boxes[i].hi.y);
      bounds.hi.z = std::max(bounds.hi.z, boxes[i].hi.z);
    }
    const std::size_t node = nodes_.size();
    nodes_.push_back({bounds});
    if (end - begin > 1) {
      int axis = 0;
      for (int a = 1; a < 3; ++a) {
        if (static_cast<int64_t>(coordinate(bounds.hi, a)) - coordinate(bounds.lo, a) >
            static_cast<int64_t>(coordinate(bounds.hi, axis)) - coordinate(bounds.lo, axis)) {
          axis = a;
        }
      }
      const std::size_t mid = begin + (end - begin) / 2;
      std::nth_element(boxes.begin() + begin, boxes.begin() + mid, boxes.begin() + end,
                       [axis](const BlockBox &a, const BlockBox &b) {
        return static_cast<int64_t>(coordinate(a.lo, axis)) + coordinate(a.hi, axis) <
               static_cast<int64_t>(coordinate(b.lo, axis)) + coordinate(b.hi, axis);
      });
      const auto left = build(boxes, begin, mid);
      const auto right = build(boxes, mid, end);
      nodes_[node].left = left;
      nodes_[node].right = right;
    }
    return node;
  }

  bool intersects_node(std::size_t index, const int32_t lo[3], const int32_t hi[3]) const {
    const auto &node = nodes_[index];
    const auto &b = node.bounds;
    if (b.lo.x > hi[0] || b.hi.x < lo[0] || b.lo.y > hi[1] || b.hi.y < lo[1] ||
        b.lo.z > hi[2] || b.hi.z < lo[2]) {
      return false;
    }
    return node.right == 0 || intersects_node(node.left, lo, hi) ||
           intersects_node(node.right, lo, hi);
  }

  std::vector<Node> nodes_;
};

}  // namespace kangaroo
