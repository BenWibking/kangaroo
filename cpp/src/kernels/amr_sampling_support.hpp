#pragma once

#include "kernel_buffer_support.hpp"

#include "kangaroo/runmeta.hpp"

#include <cstdint>
#include <optional>
#include <span>

namespace kangaroo {

struct SamplePatch {
  int16_t level = 0;
  IndexBox3 box;
  LevelGeom geom;
  ChunkBuffer view;
  RealGridAccessor values;
};

struct SamplePoint {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double value = 0.0;
};

double cell_edge(const LevelGeom &geom, int axis, int idx);
double cell_center(const LevelGeom &geom, int axis, int idx);
int32_t coord_to_index(const LevelGeom &geom, int axis, double x);
bool solve_3x3(double a[3][3], double b[3], double x[3]);
std::optional<SamplePoint>
composite_sample_at(std::span<const SamplePatch> patches, int finest_level,
                    const int32_t domain_lo[3], const int32_t domain_hi[3],
                    const bool is_periodic[3], const double domain_lo_edge[3],
                    const double domain_hi_edge[3], double x, double y,
                    double z);

} // namespace kangaroo
