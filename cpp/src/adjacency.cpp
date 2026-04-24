#include "kangaroo/adjacency.hpp"

#include <algorithm>

namespace kangaroo {

namespace {

bool overlaps_1d(int32_t a0, int32_t a1, int32_t b0, int32_t b1) {
  return !(a1 < b0 || b1 < a0);
}

}  // namespace

AdjacencyServiceLocal::AdjacencyServiceLocal(const RunMeta& meta) : meta_(meta) {
  neighbors_.reserve(meta_.steps.size());
  for (const auto& step_meta : meta_.steps) {
    neighbors_.push_back(build_step_neighbors(step_meta));
  }
}

NeighborSpan AdjacencyServiceLocal::neighbors(int32_t step, int16_t level, int32_t block, Face face) {
  if (step < 0 || static_cast<std::size_t>(step) >= neighbors_.size()) {
    return NeighborSpan{};
  }
  if (level < 0 || static_cast<std::size_t>(level) >= neighbors_[step].size()) {
    return NeighborSpan{};
  }
  const auto& level_neighbors = neighbors_[step][level];
  if (block < 0 || static_cast<std::size_t>(block) >= level_neighbors.size()) {
    return NeighborSpan{};
  }
  const auto& block_neighbors = level_neighbors[block][static_cast<std::size_t>(face)];
  return NeighborSpan{block_neighbors.data(), static_cast<int32_t>(block_neighbors.size())};
}

std::vector<AdjacencyServiceLocal::LevelNeighbors> AdjacencyServiceLocal::build_step_neighbors(
    const StepMeta& step_meta) const {
  std::vector<LevelNeighbors> step_neighbors;
  step_neighbors.reserve(step_meta.levels.size());
  for (const auto& level_meta : step_meta.levels) {
    LevelNeighbors level_neighbors(level_meta.boxes.size());
    for (int32_t block = 0; block < static_cast<int32_t>(level_meta.boxes.size()); ++block) {
      for (std::size_t face_idx = 0; face_idx < level_neighbors[block].size(); ++face_idx) {
        level_neighbors[block][face_idx] =
            compute_neighbors(level_meta, block, static_cast<Face>(face_idx));
      }
    }
    step_neighbors.push_back(std::move(level_neighbors));
  }
  return step_neighbors;
}

std::vector<int32_t> AdjacencyServiceLocal::compute_neighbors(const LevelMeta& level_meta,
                                                              int32_t block,
                                                              Face face) const {
  const auto& boxes = level_meta.boxes;
  const auto& self = boxes.at(block);

  std::vector<int32_t> neighbors;
  neighbors.reserve(8);

  for (int32_t i = 0; i < static_cast<int32_t>(boxes.size()); ++i) {
    if (i == block) {
      continue;
    }
    const auto& other = boxes.at(i);

    bool adjacent = false;
    switch (face) {
      case Face::Xm:
        adjacent = (other.hi.x + 1 == self.lo.x) &&
                   overlaps_1d(other.lo.y, other.hi.y, self.lo.y, self.hi.y) &&
                   overlaps_1d(other.lo.z, other.hi.z, self.lo.z, self.hi.z);
        break;
      case Face::Xp:
        adjacent = (other.lo.x == self.hi.x + 1) &&
                   overlaps_1d(other.lo.y, other.hi.y, self.lo.y, self.hi.y) &&
                   overlaps_1d(other.lo.z, other.hi.z, self.lo.z, self.hi.z);
        break;
      case Face::Ym:
        adjacent = (other.hi.y + 1 == self.lo.y) &&
                   overlaps_1d(other.lo.x, other.hi.x, self.lo.x, self.hi.x) &&
                   overlaps_1d(other.lo.z, other.hi.z, self.lo.z, self.hi.z);
        break;
      case Face::Yp:
        adjacent = (other.lo.y == self.hi.y + 1) &&
                   overlaps_1d(other.lo.x, other.hi.x, self.lo.x, self.hi.x) &&
                   overlaps_1d(other.lo.z, other.hi.z, self.lo.z, self.hi.z);
        break;
      case Face::Zm:
        adjacent = (other.hi.z + 1 == self.lo.z) &&
                   overlaps_1d(other.lo.x, other.hi.x, self.lo.x, self.hi.x) &&
                   overlaps_1d(other.lo.y, other.hi.y, self.lo.y, self.hi.y);
        break;
      case Face::Zp:
        adjacent = (other.lo.z == self.hi.z + 1) &&
                   overlaps_1d(other.lo.x, other.hi.x, self.lo.x, self.hi.x) &&
                   overlaps_1d(other.lo.y, other.hi.y, self.lo.y, self.hi.y);
        break;
    }

    if (adjacent) {
      neighbors.push_back(i);
    }
  }

  return neighbors;
}

}  // namespace kangaroo
