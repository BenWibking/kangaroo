#include "kangaroo/adjacency.hpp"

#include <algorithm>

namespace kangaroo {

namespace {

bool overlaps_1d(int32_t a0, int32_t a1, int32_t b0, int32_t b1) {
  return !(a1 < b0 || b1 < a0);
}

}  // namespace

AdjacencyServiceLocal::AdjacencyServiceLocal(const RunMeta& meta) : meta_(meta) {}

bool AdjacencyServiceLocal::CacheKey::operator==(const CacheKey& other) const {
  return step == other.step && level == other.level && block == other.block && face == other.face;
}

std::size_t AdjacencyServiceLocal::CacheKeyHash::operator()(const CacheKey& key) const {
  std::size_t h = 1469598103934665603ull;
  auto mix = [&](auto v) {
    h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  };
  mix(key.step);
  mix(key.level);
  mix(key.block);
  mix(static_cast<int>(key.face));
  return h;
}

NeighborSpan AdjacencyServiceLocal::neighbors(int32_t step, int16_t level, int32_t block, Face face) {
  CacheKey key{step, level, block, face};
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    return NeighborSpan{it->second.data(), static_cast<int32_t>(it->second.size())};
  }
  NeighborSpan span = compute_neighbors(step, level, block, face);
  return span;
}

NeighborSpan AdjacencyServiceLocal::compute_neighbors(int32_t step, int16_t level, int32_t block, Face face) {
  const auto& level_meta = meta_.steps.at(step).levels.at(level);
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

  CacheKey key{step, level, block, face};
  auto it = cache_.emplace(key, std::move(neighbors)).first;
  return NeighborSpan{it->second.data(), static_cast<int32_t>(it->second.size())};
}

}  // namespace kangaroo
