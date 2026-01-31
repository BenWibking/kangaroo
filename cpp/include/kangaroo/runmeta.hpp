#pragma once

#include <cstdint>
#include <vector>

namespace kangaroo {

struct Int3 {
  int32_t x = 0;
  int32_t y = 0;
  int32_t z = 0;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& x& y& z;
  }
};

struct BlockBox {
  Int3 lo;
  Int3 hi;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& lo& hi;
  }
};

struct LevelGeom {
  double dx[3] = {0.0, 0.0, 0.0};
  double x0[3] = {0.0, 0.0, 0.0};
  int32_t index_origin[3] = {0, 0, 0};
  int ref_ratio = 1;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    for (auto& v : dx) {
      ar& v;
    }
    for (auto& v : x0) {
      ar& v;
    }
    for (auto& v : index_origin) {
      ar& v;
    }
    ar& ref_ratio;
  }
};

struct LevelMeta {
  LevelGeom geom;
  std::vector<BlockBox> boxes;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& geom& boxes;
  }
};

struct StepMeta {
  int32_t step = 0;
  std::vector<LevelMeta> levels;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& step& levels;
  }
};

struct RunMeta {
  std::vector<StepMeta> steps;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& steps;
  }
};

}  // namespace kangaroo
