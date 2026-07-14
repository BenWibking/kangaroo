#pragma once

#include "kangaroo/default_kernels.hpp"

#include "kangaroo/amr_patch_codec.hpp"
#include "kangaroo/buffer_resolution.hpp"
#include "kangaroo/param_decode.hpp"
#include "kangaroo/plotfile_reader.hpp"
#include "kangaroo/runtime.hpp"

#include "amrexpr.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <msgpack.hpp>

namespace kangaroo {

namespace {

bool debug_projection_enabled() {
  static const bool enabled =
      std::getenv("KANGAROO_DEBUG_PROJECTION") != nullptr;
  return enabled;
}

bool plotfile_zero_copy_reads_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("KANGAROO_PLOTFILE_ZERO_COPY_READS");
    if (env == nullptr || *env == '\0') {
      return true;
    }
    const std::string value(env);
    return value != "0" && value != "false" && value != "FALSE" &&
           value != "off" && value != "OFF";
  }();
  return enabled;
}

void log_projection_kernel_summary(const char *kernel, int32_t level_index,
                                   int32_t block,
                                   std::size_t covered_boxes_count,
                                   std::size_t candidates,
                                   std::size_t covered_skips,
                                   std::size_t bounds_skips,
                                   std::size_t deposited, double out_sum) {
  if (!debug_projection_enabled()) {
    return;
  }
  std::cout << "[kangaroo][projection] kernel=" << kernel
            << " locality=" << hpx::get_locality_id()
            << " level=" << level_index << " block=" << block
            << " covered_boxes=" << covered_boxes_count
            << " candidates=" << candidates
            << " covered_skips=" << covered_skips
            << " bounds_skips=" << bounds_skips << " deposited=" << deposited
            << " output_sum=" << out_sum << std::endl;
}

template <typename T>
T load_buffer_scalar(const std::uint8_t *data, std::size_t index);

template <typename T>
void store_buffer_scalar(std::uint8_t *data, std::size_t index, T value);

void append_particle_values_as_f64(const ParticleFieldChunk &data,
                                   const std::string &name,
                                   const std::string &context,
                                   std::vector<double> &out_vals) {
  const std::size_t n =
      static_cast<std::size_t>(std::max<int64_t>(0, data.count));
  const std::size_t start = out_vals.size();
  out_vals.resize(start + n, 0.0);
  if (data.dtype == "float64") {
    if (data.bytes.size() < n * sizeof(double)) {
      throw std::runtime_error(context + ": short float64 payload for " + name);
    }
    for (std::size_t i = 0; i < n; ++i) {
      out_vals[start + i] = load_buffer_scalar<double>(data.bytes.data(), i);
    }
  } else if (data.dtype == "float32") {
    if (data.bytes.size() < n * sizeof(float)) {
      throw std::runtime_error(context + ": short float32 payload for " + name);
    }
    for (std::size_t i = 0; i < n; ++i) {
      out_vals[start + i] =
          static_cast<double>(load_buffer_scalar<float>(data.bytes.data(), i));
    }
  } else if (data.dtype == "int64") {
    if (data.bytes.size() < n * sizeof(int64_t)) {
      throw std::runtime_error(context + ": short int64 payload for " + name);
    }
    for (std::size_t i = 0; i < n; ++i) {
      out_vals[start + i] = static_cast<double>(
          load_buffer_scalar<int64_t>(data.bytes.data(), i));
    }
  } else {
    throw std::runtime_error(context + ": unsupported dtype '" + data.dtype +
                             "' for " + name);
  }
}

std::unordered_map<double, int64_t>
decode_particle_value_counts(std::span<const std::uint8_t> bytes) {
  std::unordered_map<double, int64_t> counts;
  if (bytes.size() < sizeof(uint64_t)) {
    return counts;
  }
  uint64_t n = 0;
  std::memcpy(&n, bytes.data(), sizeof(uint64_t));
  const std::size_t expected =
      sizeof(uint64_t) +
      static_cast<std::size_t>(n) * (sizeof(double) + sizeof(int64_t));
  if (bytes.size() < expected) {
    return counts;
  }
  counts.reserve(static_cast<std::size_t>(n));
  const auto *ptr = bytes.data() + sizeof(uint64_t);
  for (uint64_t i = 0; i < n; ++i) {
    double value = 0.0;
    int64_t count = 0;
    std::memcpy(&value, ptr, sizeof(double));
    ptr += sizeof(double);
    std::memcpy(&count, ptr, sizeof(int64_t));
    ptr += sizeof(int64_t);
    counts[value] += count;
  }
  return counts;
}

template <typename T>
T load_buffer_scalar(const std::uint8_t *data, std::size_t index) {
  T value;
  std::memcpy(&value, data + index * sizeof(T), sizeof(T));
  return value;
}

template <typename T>
void store_buffer_scalar(std::uint8_t *data, std::size_t index, T value) {
  std::memcpy(data + index * sizeof(T), &value, sizeof(T));
}

bool descriptors_match(const BufferDesc &lhs, const BufferDesc &rhs) {
  if (lhs.scalar != rhs.scalar || lhs.rank != rhs.rank)
    return false;
  for (std::size_t axis = 0; axis < lhs.rank; ++axis) {
    if (lhs.extents[axis] != rhs.extents[axis] ||
        lhs.strides_bytes[axis] != rhs.strides_bytes[axis])
      return false;
  }
  return true;
}

template <typename T>
void reduce_matching_buffers(std::span<const ChunkBuffer> inputs,
                             ChunkBuffer &output) {
  const auto count = static_cast<std::size_t>(output.desc().element_count());
  auto out = output.mutable_byte_view();
  std::fill(out.begin(), out.end(), std::uint8_t{0});
  for (const auto &input : inputs) {
    if (!descriptors_match(input.desc(), output.desc())) {
      throw BufferContractError(
          BufferContractReason::kDescriptorStorageMismatch,
          "generic reduction requires identical descriptors");
    }
    const auto in = input.byte_view();
    for (std::size_t index = 0; index < count; ++index) {
      const T sum = load_buffer_scalar<T>(out.data(), index) +
                    load_buffer_scalar<T>(in.data(), index);
      store_buffer_scalar(out.data(), index, sum);
    }
  }
}

void reduce_matching_real_buffers(std::span<const ChunkBuffer> inputs,
                                  ChunkBuffer &output) {
  if (output.desc().scalar == ScalarType::kF32) {
    reduce_matching_buffers<float>(inputs, output);
  } else if (output.desc().scalar == ScalarType::kF64) {
    reduce_matching_buffers<double>(inputs, output);
  } else {
    throw BufferContractError(
        BufferContractReason::kScalarMismatch,
        "generic real reduction requires f32 or f64 buffers");
  }
}

struct RealGridAccessor {
  const std::uint8_t *data = nullptr;
  std::array<std::int64_t, 3> strides{};
  double (*load)(const RealGridAccessor &, int, int, int) = nullptr;

  double operator()(int i, int j, int k) const { return load(*this, i, j, k); }
};

template <typename T>
RealGridAccessor make_real_grid_accessor(const TensorView<const T, 3> &grid) {
  RealGridAccessor accessor;
  accessor.data = grid.byte_data();
  accessor.strides = grid.strides_bytes();
  accessor.load = [](const RealGridAccessor &self, int i, int j, int k) {
    const auto offset = static_cast<std::uint64_t>(i) * self.strides[0] +
                        static_cast<std::uint64_t>(j) * self.strides[1] +
                        static_cast<std::uint64_t>(k) * self.strides[2];
    T value;
    std::memcpy(&value, self.data + offset, sizeof(T));
    return static_cast<double>(value);
  };
  return accessor;
}

RealGridAccessor make_real_grid_accessor(const ChunkBuffer &buffer) {
  RealGridAccessor accessor;
  visit_real_buffers_exact<1>(
      std::span<const ChunkBuffer>(&buffer, 1), [&](auto view) {
        accessor = make_real_grid_accessor(view.template tensor<3>());
      });
  return accessor;
}

void encode_particle_value_counts(
    const std::unordered_map<double, int64_t> &counts,
    std::vector<std::uint8_t> &out) {
  const uint64_t n = static_cast<uint64_t>(counts.size());
  out.resize(sizeof(uint64_t) +
             static_cast<std::size_t>(n) * (sizeof(double) + sizeof(int64_t)));
  auto *ptr = out.data();
  std::memcpy(ptr, &n, sizeof(uint64_t));
  ptr += sizeof(uint64_t);
  for (const auto &[value, count] : counts) {
    std::memcpy(ptr, &value, sizeof(double));
    ptr += sizeof(double);
    std::memcpy(ptr, &count, sizeof(int64_t));
    ptr += sizeof(int64_t);
  }
}

std::shared_ptr<plotfile::PlotfileReader>
get_plotfile_reader(const std::string &path) {
  static std::mutex reader_mutex;
  static std::unordered_map<std::string,
                            std::weak_ptr<plotfile::PlotfileReader>>
      readers;

  std::lock_guard<std::mutex> lock(reader_mutex);
  auto it = readers.find(path);
  if (it != readers.end()) {
    if (auto shared = it->second.lock()) {
      return shared;
    }
  }
  auto shared = std::make_shared<plotfile::PlotfileReader>(path);
  readers[path] = shared;
  return shared;
}

struct SamplePatch {
  int16_t level = 0;
  IndexBox3 box;
  LevelGeom geom;
  ChunkBuffer view;
  RealGridAccessor values;
};

double cell_edge(const LevelGeom &geom, int axis, int idx) {
  return geom.x0[axis] + (idx - geom.index_origin[axis]) * geom.dx[axis];
}

double cell_center(const LevelGeom &geom, int axis, int idx) {
  return cell_edge(geom, axis, idx) + 0.5 * geom.dx[axis];
}

int32_t coord_to_index(const LevelGeom &geom, int axis, double x) {
  return static_cast<int32_t>(std::floor((x - geom.x0[axis]) / geom.dx[axis])) +
         geom.index_origin[axis];
}

double wrap_coord(double x, double lo, double hi) {
  const double period = hi - lo;
  if (!(period > 0.0)) {
    return x;
  }
  double y = std::fmod(x - lo, period);
  if (y < 0.0) {
    y += period;
  }
  return lo + y;
}

bool solve_3x3(double a[3][3], double b[3], double x[3]) {
  double m[3][4] = {
      {a[0][0], a[0][1], a[0][2], b[0]},
      {a[1][0], a[1][1], a[1][2], b[1]},
      {a[2][0], a[2][1], a[2][2], b[2]},
  };

  for (int col = 0; col < 3; ++col) {
    int piv = col;
    double best = std::abs(m[col][col]);
    for (int r = col + 1; r < 3; ++r) {
      const double cand = std::abs(m[r][col]);
      if (cand > best) {
        best = cand;
        piv = r;
      }
    }
    if (best < 1e-30) {
      return false;
    }
    if (piv != col) {
      for (int c = col; c < 4; ++c) {
        std::swap(m[col][c], m[piv][c]);
      }
    }
    const double inv = 1.0 / m[col][col];
    for (int c = col; c < 4; ++c) {
      m[col][c] *= inv;
    }
    for (int r = 0; r < 3; ++r) {
      if (r == col) {
        continue;
      }
      const double f = m[r][col];
      if (f == 0.0) {
        continue;
      }
      for (int c = col; c < 4; ++c) {
        m[r][c] -= f * m[col][c];
      }
    }
  }

  x[0] = m[0][3];
  x[1] = m[1][3];
  x[2] = m[2][3];
  return true;
}

std::optional<double> patch_value_at(const SamplePatch &p, int32_t i, int32_t j,
                                     int32_t k) {
  if (i < p.box.lo[0] || i > p.box.hi[0] || j < p.box.lo[1] ||
      j > p.box.hi[1] || k < p.box.lo[2] || k > p.box.hi[2]) {
    return std::nullopt;
  }
  const int32_t li = i - p.box.lo[0];
  const int32_t lj = j - p.box.lo[1];
  const int32_t lk = k - p.box.lo[2];
  return p.values(li, lj, lk);
}

struct SamplePoint {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double value = 0.0;
};

std::optional<SamplePoint>
composite_sample_at(std::span<const SamplePatch> patches, int finest_level,
                    const int32_t domain_lo[3], const int32_t domain_hi[3],
                    const bool is_periodic[3], const double domain_lo_edge[3],
                    const double domain_hi_edge[3], double x, double y,
                    double z) {
  double xyz[3] = {x, y, z};
  for (int ax = 0; ax < 3; ++ax) {
    if (is_periodic[ax]) {
      xyz[ax] = wrap_coord(xyz[ax], domain_lo_edge[ax], domain_hi_edge[ax]);
    } else if (xyz[ax] < domain_lo_edge[ax] || xyz[ax] >= domain_hi_edge[ax]) {
      return std::nullopt;
    }
  }

  for (int lev = finest_level; lev >= 0; --lev) {
    for (const auto &p : patches) {
      if (p.level != lev) {
        continue;
      }
      int32_t idx[3];
      for (int ax = 0; ax < 3; ++ax) {
        idx[ax] = coord_to_index(p.geom, ax, xyz[ax]);
      }
      if (idx[0] < p.box.lo[0] || idx[0] > p.box.hi[0] ||
          idx[1] < p.box.lo[1] || idx[1] > p.box.hi[1] ||
          idx[2] < p.box.lo[2] || idx[2] > p.box.hi[2]) {
        continue;
      }
      auto value = patch_value_at(p, idx[0], idx[1], idx[2]);
      if (!value.has_value()) {
        continue;
      }
      SamplePoint out;
      out.x = cell_center(p.geom, 0, idx[0]);
      out.y = cell_center(p.geom, 1, idx[1]);
      out.z = cell_center(p.geom, 2, idx[2]);
      out.value = *value;
      return out;
    }
  }
  return std::nullopt;
}

template <typename Params, typename DecodeFn>
KernelRegistry::KernelParamsPrepareFn
make_kernel_params_preparer(DecodeFn decode_fn) {
  return
      [decode_fn = std::move(decode_fn)](
          const KernelParamContext &context) -> KernelRegistry::PreparedParams {
        if (context.params_msgpack.empty()) {
          return {};
        }
        auto decoded = decode_fn(cached_params_root(context.params_msgpack));
        auto prepared = std::make_shared<const Params>(std::move(decoded));
        return KernelRegistry::PreparedParams{std::type_index(typeid(Params)),
                                              std::move(prepared)};
      };
}

template <typename Params, typename DecodeFn>
KernelRegistry::KernelParamsPrepareFn
make_covered_box_params_preparer(DecodeFn decode_fn) {
  return
      [decode_fn = std::move(decode_fn)](
          const KernelParamContext &context) -> KernelRegistry::PreparedParams {
        if (context.params_msgpack.empty() && !context.covered_boxes) {
          return {};
        }
        Params decoded;
        if (!context.params_msgpack.empty()) {
          decoded = decode_fn(cached_params_root(context.params_msgpack));
        }
        if (context.covered_boxes) {
          decoded.covered_boxes = context.covered_boxes;
        }
        auto prepared = std::make_shared<const Params>(std::move(decoded));
        return KernelRegistry::PreparedParams{std::type_index(typeid(Params)),
                                              std::move(prepared)};
      };
}

std::shared_ptr<const CoveredBoxListIR>
parse_covered_boxes_param(const msgpack::object &root) {
  const auto *boxes = find_msgpack_map_value(root, "covered_boxes");
  if (boxes == nullptr || boxes->type != msgpack::type::ARRAY) {
    return {};
  }

  auto parsed = std::make_shared<CoveredBoxListIR>();
  parsed->reserve(boxes->via.array.size);
  for (uint32_t i = 0; i < boxes->via.array.size; ++i) {
    const auto &entry = boxes->via.array.ptr[i];
    if (entry.type != msgpack::type::ARRAY || entry.via.array.size != 2) {
      continue;
    }
    const auto &lo = entry.via.array.ptr[0];
    const auto &hi = entry.via.array.ptr[1];
    if (lo.type != msgpack::type::ARRAY || hi.type != msgpack::type::ARRAY ||
        lo.via.array.size != 3 || hi.via.array.size != 3) {
      continue;
    }
    CoveredBoxIR box;
    for (uint32_t d = 0; d < 3; ++d) {
      box.lo[d] = lo.via.array.ptr[d].as<int32_t>();
      box.hi[d] = hi.via.array.ptr[d].as<int32_t>();
    }
    parsed->push_back(box);
  }
  return parsed;
}

std::size_t
covered_box_count(const std::shared_ptr<const CoveredBoxListIR> &boxes) {
  return boxes ? boxes->size() : 0;
}

bool covered_box_contains(const CoveredBoxIR &box, int i, int j, int k) {
  return i >= box.lo[0] && i <= box.hi[0] && j >= box.lo[1] && j <= box.hi[1] &&
         k >= box.lo[2] && k <= box.hi[2];
}

double min_dist_sq_to_interval(double a0, double a1) {
  if (a1 < 0.0) {
    return a1 * a1;
  }
  if (a0 > 0.0) {
    return a0 * a0;
  }
  return 0.0;
}

double max_dist_sq_to_interval(double a0, double a1) {
  const double aa0 = std::abs(a0);
  const double aa1 = std::abs(a1);
  const double amax = (aa0 > aa1) ? aa0 : aa1;
  return amax * amax;
}

bool sphere_may_intersect_cell(double radius2, double x0, double x1, double y0,
                               double y1, double z0, double z1) {
  const double r2_min = min_dist_sq_to_interval(x0, x1) +
                        min_dist_sq_to_interval(y0, y1) +
                        min_dist_sq_to_interval(z0, z1);
  const double r2_max = max_dist_sq_to_interval(x0, x1) +
                        max_dist_sq_to_interval(y0, y1) +
                        max_dist_sq_to_interval(z0, z1);
  return radius2 >= r2_min && radius2 <= r2_max;
}

bool cylinder_may_intersect_cell(double radius2, double height, double x0,
                                 double x1, double y0, double y1, double z0,
                                 double z1) {
  const double r2_min =
      min_dist_sq_to_interval(x0, x1) + min_dist_sq_to_interval(y0, y1);
  const double r2_max =
      max_dist_sq_to_interval(x0, x1) + max_dist_sq_to_interval(y0, y1);
  return radius2 >= r2_min && radius2 <= r2_max && z1 >= -height &&
         z0 <= height;
}

struct CellIndexRange {
  int first = 0;
  int last = -1;

  bool empty() const { return first > last; }
};

CellIndexRange cells_intersecting_axis_band(double band_lo, double band_hi,
                                            double x0, double dx,
                                            int32_t index_origin,
                                            int32_t block_lo,
                                            int32_t block_hi) {
  if (!std::isfinite(band_lo) || !std::isfinite(band_hi) || dx <= 0.0 ||
      band_hi < band_lo || block_hi < block_lo) {
    return {};
  }

  const double scaled_first =
      (band_lo - x0) / dx + static_cast<double>(index_origin) - 1.0;
  const double scaled_last =
      (band_hi - x0) / dx + static_cast<double>(index_origin);
  constexpr double pad = 1.0e-12;
  const auto raw_first =
      static_cast<int64_t>(std::ceil(scaled_first - pad)) - 1;
  const auto raw_last = static_cast<int64_t>(std::floor(scaled_last + pad)) + 1;
  const int64_t clamped_first =
      std::max<int64_t>(static_cast<int64_t>(block_lo), raw_first);
  const int64_t clamped_last =
      std::min<int64_t>(static_cast<int64_t>(block_hi), raw_last);
  if (clamped_first > clamped_last) {
    return {};
  }
  return CellIndexRange{static_cast<int>(clamped_first - block_lo),
                        static_cast<int>(clamped_last - block_lo)};
}

struct FluxPoint {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

void add_point_unique(std::array<FluxPoint, 16> &pts, int &npts, double x,
                      double y, double z, double tol) {
  const double tol2 = tol * tol;
  for (int i = 0; i < npts; ++i) {
    const double dx = pts[static_cast<std::size_t>(i)].x - x;
    const double dy = pts[static_cast<std::size_t>(i)].y - y;
    const double dz = pts[static_cast<std::size_t>(i)].z - z;
    if ((dx * dx + dy * dy + dz * dz) <= tol2) {
      return;
    }
  }
  if (npts < static_cast<int>(pts.size())) {
    pts[static_cast<std::size_t>(npts)] = FluxPoint{x, y, z};
    ++npts;
  }
}

double plane_box_section_area(double x0, double x1, double y0, double y1,
                              double z0, double z1, double nx, double ny,
                              double nz, double d) {
  const double scale = std::abs(x0) + std::abs(x1) + std::abs(y0) +
                       std::abs(y1) + std::abs(z0) + std::abs(z1) +
                       std::abs(d) + 1.0;
  const double tol = 1.0e-12 * scale;

  const std::array<FluxPoint, 8> verts{
      FluxPoint{x0, y0, z0}, FluxPoint{x1, y0, z0}, FluxPoint{x0, y1, z0},
      FluxPoint{x1, y1, z0}, FluxPoint{x0, y0, z1}, FluxPoint{x1, y0, z1},
      FluxPoint{x0, y1, z1}, FluxPoint{x1, y1, z1}};
  const std::array<std::array<int, 2>, 12> edges{
      std::array<int, 2>{0, 1}, std::array<int, 2>{2, 3},
      std::array<int, 2>{4, 5}, std::array<int, 2>{6, 7},
      std::array<int, 2>{0, 2}, std::array<int, 2>{1, 3},
      std::array<int, 2>{4, 6}, std::array<int, 2>{5, 7},
      std::array<int, 2>{0, 4}, std::array<int, 2>{1, 5},
      std::array<int, 2>{2, 6}, std::array<int, 2>{3, 7}};

  std::array<FluxPoint, 16> pts{};
  int npts = 0;

  for (const auto &edge : edges) {
    const int i0 = edge[0];
    const int i1 = edge[1];
    const auto &p0 = verts[static_cast<std::size_t>(i0)];
    const auto &p1 = verts[static_cast<std::size_t>(i1)];

    double f0 = nx * p0.x + ny * p0.y + nz * p0.z - d;
    double f1 = nx * p1.x + ny * p1.y + nz * p1.z - d;
    if (std::abs(f0) <= tol) {
      f0 = 0.0;
    }
    if (std::abs(f1) <= tol) {
      f1 = 0.0;
    }

    if (f0 == 0.0 && f1 == 0.0) {
      add_point_unique(pts, npts, p0.x, p0.y, p0.z, tol);
      add_point_unique(pts, npts, p1.x, p1.y, p1.z, tol);
      continue;
    }
    if (f0 == 0.0) {
      add_point_unique(pts, npts, p0.x, p0.y, p0.z, tol);
      continue;
    }
    if (f1 == 0.0) {
      add_point_unique(pts, npts, p1.x, p1.y, p1.z, tol);
      continue;
    }
    if ((f0 < 0.0 && f1 > 0.0) || (f0 > 0.0 && f1 < 0.0)) {
      const double t = f0 / (f0 - f1);
      const double x = p0.x + t * (p1.x - p0.x);
      const double y = p0.y + t * (p1.y - p0.y);
      const double z = p0.z + t * (p1.z - p0.z);
      add_point_unique(pts, npts, x, y, z, tol);
    }
  }

  if (npts < 3) {
    return 0.0;
  }

  double cx = 0.0;
  double cy = 0.0;
  double cz = 0.0;
  for (int i = 0; i < npts; ++i) {
    cx += pts[static_cast<std::size_t>(i)].x;
    cy += pts[static_cast<std::size_t>(i)].y;
    cz += pts[static_cast<std::size_t>(i)].z;
  }
  cx /= static_cast<double>(npts);
  cy /= static_cast<double>(npts);
  cz /= static_cast<double>(npts);

  double ax = 1.0;
  double ay = 0.0;
  double az = 0.0;
  if (std::abs(nx) > 0.9) {
    ax = 0.0;
    ay = 1.0;
    az = 0.0;
  }
  double e1x = ny * az - nz * ay;
  double e1y = nz * ax - nx * az;
  double e1z = nx * ay - ny * ax;
  const double e1norm = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
  if (e1norm <= 0.0) {
    return 0.0;
  }
  e1x /= e1norm;
  e1y /= e1norm;
  e1z /= e1norm;

  const double e2x = ny * e1z - nz * e1y;
  const double e2y = nz * e1x - nx * e1z;
  const double e2z = nx * e1y - ny * e1x;

  std::array<double, 16> u{};
  std::array<double, 16> v{};
  std::array<double, 16> ang{};
  for (int i = 0; i < npts; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const double rx = pts[idx].x - cx;
    const double ry = pts[idx].y - cy;
    const double rz = pts[idx].z - cz;
    u[idx] = rx * e1x + ry * e1y + rz * e1z;
    v[idx] = rx * e2x + ry * e2y + rz * e2z;
    ang[idx] = std::atan2(v[idx], u[idx]);
  }

  for (int i = 1; i < npts; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const double key_ang = ang[idx];
    const double key_u = u[idx];
    const double key_v = v[idx];
    int j = i - 1;
    while (j >= 0 && ang[static_cast<std::size_t>(j)] > key_ang) {
      const std::size_t dst = static_cast<std::size_t>(j + 1);
      const std::size_t src = static_cast<std::size_t>(j);
      ang[dst] = ang[src];
      u[dst] = u[src];
      v[dst] = v[src];
      --j;
    }
    const std::size_t dst = static_cast<std::size_t>(j + 1);
    ang[dst] = key_ang;
    u[dst] = key_u;
    v[dst] = key_v;
  }

  double area2 = 0.0;
  for (int i = 0; i < npts; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const std::size_t next =
        static_cast<std::size_t>((i + 1 < npts) ? (i + 1) : 0);
    area2 += u[idx] * v[next] - v[idx] * u[next];
  }
  return 0.5 * std::abs(area2);
}

double spherical_section_area_in_intersecting_cell(double radius, double x0,
                                                   double x1, double y0,
                                                   double y1, double z0,
                                                   double z1) {
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const double dz = z1 - z0;
  const double vol = dx * dy * dz;
  if (vol <= 0.0) {
    return 0.0;
  }

  const double xc = 0.5 * (x0 + x1);
  const double yc = 0.5 * (y0 + y1);
  const double zc = 0.5 * (z0 + z1);
  const double rc = std::sqrt(xc * xc + yc * yc + zc * zc);
  if (rc <= 0.0) {
    return 0.0;
  }

  return plane_box_section_area(x0, x1, y0, y1, z0, z1, xc / rc, yc / rc,
                                zc / rc, radius);
}

double spherical_section_area_in_cell(double radius, double x0, double x1,
                                      double y0, double y1, double z0,
                                      double z1) {
  const double radius2 = radius * radius;
  if (!sphere_may_intersect_cell(radius2, x0, x1, y0, y1, z0, z1)) {
    return 0.0;
  }
  return spherical_section_area_in_intersecting_cell(radius, x0, x1, y0, y1, z0,
                                                     z1);
}

double cylindrical_section_area_in_intersecting_cell(double radius,
                                                     double height, double x0,
                                                     double x1, double y0,
                                                     double y1, double z0,
                                                     double z1) {
  const double zlo = std::max(z0, -height);
  const double zhi = std::min(z1, height);
  if (zhi <= zlo) {
    return 0.0;
  }

  const double xc = 0.5 * (x0 + x1);
  const double yc = 0.5 * (y0 + y1);
  const double rc = std::sqrt(xc * xc + yc * yc);
  if (rc <= 0.0) {
    return 0.0;
  }

  return plane_box_section_area(x0, x1, y0, y1, zlo, zhi, xc / rc, yc / rc, 0.0,
                                radius);
}

} // namespace

} // namespace kangaroo
