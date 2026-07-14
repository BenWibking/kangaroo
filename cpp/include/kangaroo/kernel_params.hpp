#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <hpx/serialization/array.hpp>
#include <hpx/serialization/string.hpp>
#include <hpx/serialization/variant.hpp>
#include <hpx/serialization/vector.hpp>

namespace kangaroo {

struct CoveredBoxIR {
  std::array<int32_t, 3> lo{0, 0, 0};
  std::array<int32_t, 3> hi{0, 0, 0};

  template <typename Archive> void serialize(Archive &ar, unsigned) {
    ar & lo & hi;
  }
};

using CoveredBoxListIR = std::vector<CoveredBoxIR>;

struct NoKernelParamsIR {
  template <typename Archive> void serialize(Archive &, unsigned) {}
};

#define KANGAROO_SERIALIZE_FIELDS(...)                                         \
  template <typename Archive> void serialize(Archive &ar, unsigned) {          \
    ar & __VA_ARGS__;                                                          \
  }

struct AmrSubboxPackParams {
  int32_t input_field = -1;
  int32_t input_version = 0;
  int32_t input_step = 0;
  int16_t input_level = 0;
  int32_t halo_cells = 1;
  KANGAROO_SERIALIZE_FIELDS(input_field & input_version & input_step &
                            input_level & halo_cells)
};

struct GradStencilParams {
  int32_t input_field = -1;
  int32_t input_version = 0;
  int32_t input_step = 0;
  int16_t input_level = 0;
  int32_t stencil_radius = 1;
  KANGAROO_SERIALIZE_FIELDS(input_field & input_version & input_step &
                            input_level & stencil_radius)
};

struct PlotfileLoadParams {
  std::string plotfile;
  int32_t level = 0;
  int32_t comp = 0;
  KANGAROO_SERIALIZE_FIELDS(plotfile & level & comp)
};

struct FluxSurfaceParams {
  std::vector<double> radii;
  std::vector<int32_t> radius_indices;
  std::vector<double> temperature_bins;
  std::size_t num_radii = 0;
  double gamma = 5.0 / 3.0;
  std::shared_ptr<const CoveredBoxListIR> covered_boxes;
  KANGAROO_SERIALIZE_FIELDS(radii & radius_indices & temperature_bins &
                            num_radii & gamma)
};

struct CylindricalFluxParams {
  double radius = 0.0;
  std::vector<double> heights;
  std::vector<int32_t> height_indices;
  std::vector<double> temperature_bins;
  std::size_t num_heights = 0;
  double gamma = 5.0 / 3.0;
  std::shared_ptr<const CoveredBoxListIR> covered_boxes;
  KANGAROO_SERIALIZE_FIELDS(radius & heights & height_indices &
                            temperature_bins & num_heights & gamma)
};

struct ToomreProfileParams {
  std::array<double, 2> radial_range{0.0, 1.0};
  int32_t bins = 1;
  std::array<double, 2> z_bounds{-1.0, 1.0};
  std::array<double, 3> center{0.0, 0.0, 0.0};
  std::shared_ptr<const CoveredBoxListIR> covered_boxes;
  KANGAROO_SERIALIZE_FIELDS(radial_range & bins & z_bounds & center)
};

struct UniformSliceCellParams {
  int32_t axis = 2;
  double coord = 0.0;
  int32_t plane_index = 0;
  bool has_plane_index = false;
  std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
  std::array<int32_t, 2> resolution{1, 1};
  std::shared_ptr<const CoveredBoxListIR> covered_boxes;
  KANGAROO_SERIALIZE_FIELDS(axis & coord & plane_index & has_plane_index &
                            rect & resolution)
};

struct UniformProjectionParams {
  int32_t axis = 2;
  std::array<double, 2> axis_bounds{0.0, 1.0};
  std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
  std::array<int32_t, 2> resolution{1, 1};
  std::shared_ptr<const CoveredBoxListIR> covered_boxes;
  KANGAROO_SERIALIZE_FIELDS(axis & axis_bounds & rect & resolution)
};

struct FieldExprParams {
  std::string expression;
  std::vector<std::string> variables;
  KANGAROO_SERIALIZE_FIELDS(expression &variables)
};

struct UniformSliceParams {
  int32_t axis = 2;
  double coord = 0.0;
  std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
  std::array<int32_t, 2> resolution{1, 1};
  KANGAROO_SERIALIZE_FIELDS(axis & coord & rect & resolution)
};

struct Histogram1DParams {
  std::array<double, 2> range{0.0, 1.0};
  int32_t bins = 1;
  std::shared_ptr<const CoveredBoxListIR> covered_boxes;
  KANGAROO_SERIALIZE_FIELDS(range &bins)
};

struct Histogram2DParams {
  std::array<double, 2> x_range{0.0, 1.0};
  std::array<double, 2> y_range{0.0, 1.0};
  std::array<int32_t, 2> bins{1, 1};
  std::string weight_mode{"input"};
  std::shared_ptr<const CoveredBoxListIR> covered_boxes;
  KANGAROO_SERIALIZE_FIELDS(x_range & y_range & bins & weight_mode)
};

struct ParticleFieldParams {
  std::string particle_type;
  std::string field_name;
  KANGAROO_SERIALIZE_FIELDS(particle_type &field_name)
};

struct ParticleCicGridParams {
  std::string particle_type;
  int32_t level_index = -1;
  int32_t axis = 2;
  std::array<double, 2> axis_bounds{0.0, 0.0};
  double mass_max = std::numeric_limits<double>::quiet_NaN();
  std::shared_ptr<const CoveredBoxListIR> covered_boxes;
  KANGAROO_SERIALIZE_FIELDS(particle_type & level_index & axis & axis_bounds &
                            mass_max)
};

struct ParticleCicProjectionParams {
  std::string particle_type;
  int32_t level_index = -1;
  int32_t axis = 2;
  std::array<double, 2> axis_bounds{0.0, 0.0};
  std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
  std::array<int32_t, 2> resolution{1, 1};
  double mass_max = std::numeric_limits<double>::quiet_NaN();
  std::shared_ptr<const CoveredBoxListIR> covered_boxes;
  KANGAROO_SERIALIZE_FIELDS(particle_type & level_index & axis & axis_bounds &
                            rect & resolution & mass_max)
};

struct ScalarParams {
  double scalar = 0.0;
  KANGAROO_SERIALIZE_FIELDS(scalar)
};

struct ValuesParams {
  std::vector<double> values;
  KANGAROO_SERIALIZE_FIELDS(values)
};

struct FiniteOnlyParams {
  bool finite_only = true;
  KANGAROO_SERIALIZE_FIELDS(finite_only)
};

struct ParticleHistogramParams {
  std::vector<double> edges;
  bool density = false;
  KANGAROO_SERIALIZE_FIELDS(edges &density)
};

struct TopKModesParams {
  int64_t k = 0;
  KANGAROO_SERIALIZE_FIELDS(k)
};

struct SliceFinalizeParams {
  double pixel_area = 1.0;
  KANGAROO_SERIALIZE_FIELDS(pixel_area)
};

#undef KANGAROO_SERIALIZE_FIELDS

using KernelParamsIR =
    std::variant<NoKernelParamsIR, AmrSubboxPackParams, GradStencilParams,
                 PlotfileLoadParams, FluxSurfaceParams, CylindricalFluxParams,
                 ToomreProfileParams,
                 UniformSliceCellParams, UniformProjectionParams,
                 FieldExprParams, UniformSliceParams, Histogram1DParams,
                 Histogram2DParams, ParticleFieldParams, ParticleCicGridParams,
                 ParticleCicProjectionParams, ScalarParams, ValuesParams,
                 FiniteOnlyParams, ParticleHistogramParams, TopKModesParams,
                 SliceFinalizeParams>;

template <typename Params>
const Params &require_kernel_params(const KernelParamsIR &params,
                                    const char *kernel) {
  if (const auto *typed = std::get_if<Params>(&params))
    return *typed;
  throw std::runtime_error(std::string("kernel ") + kernel +
                           " received incompatible typed parameters");
}

} // namespace kangaroo
