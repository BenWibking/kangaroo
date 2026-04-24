#pragma once

#include "kangaroo/runmeta.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace kangaroo {

enum class Face : uint8_t { Xm = 0, Xp = 1, Ym = 2, Yp = 3, Zm = 4, Zp = 5 };

struct NeighborSpan {
  const int32_t* ptr = nullptr;
  int32_t n = 0;
};

class AdjacencyService {
 public:
  virtual ~AdjacencyService() = default;
  virtual NeighborSpan neighbors(int32_t step, int16_t level, int32_t block, Face face) = 0;
};

class AdjacencyServiceLocal final : public AdjacencyService {
 public:
 explicit AdjacencyServiceLocal(const RunMeta& meta);

  NeighborSpan neighbors(int32_t step, int16_t level, int32_t block, Face face) override;

 private:
  using FaceNeighbors = std::array<std::vector<int32_t>, 6>;
  using LevelNeighbors = std::vector<FaceNeighbors>;
  std::vector<LevelNeighbors> build_step_neighbors(const StepMeta& step_meta) const;
  std::vector<int32_t> compute_neighbors(const LevelMeta& level_meta, int32_t block, Face face) const;

  const RunMeta& meta_;
  std::vector<std::vector<LevelNeighbors>> neighbors_;
};

}  // namespace kangaroo
