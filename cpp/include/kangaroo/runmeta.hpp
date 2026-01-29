#pragma once

#include <cstdint>
#include <vector>

namespace kangaroo {

struct Int3 {
  int32_t x = 0;
  int32_t y = 0;
  int32_t z = 0;
};

struct BlockBox {
  Int3 lo;
  Int3 hi;
};

struct LevelGeom {
  double dx[3] = {0.0, 0.0, 0.0};
  double x0[3] = {0.0, 0.0, 0.0};
  int ref_ratio = 1;
};

struct LevelMeta {
  LevelGeom geom;
  std::vector<BlockBox> boxes;
};

struct StepMeta {
  int32_t step = 0;
  std::vector<LevelMeta> levels;
};

struct RunMeta {
  std::vector<StepMeta> steps;
};

}  // namespace kangaroo
