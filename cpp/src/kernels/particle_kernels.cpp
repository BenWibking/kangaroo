#include "default_kernel_families.hpp"

#include "kernel_buffer_support.hpp"
#include "kernel_param_support.hpp"
#include "particle_kernel_support.hpp"
#include "projection_kernel_support.hpp"

#include "kangaroo/runtime.hpp"
#include <span>
#include <limits>
#include <cmath>
#include <array>
#include <algorithm>

namespace kangaroo {

namespace {

// Conservative cloud-in-cell deposition on the AMR hierarchy as a whole.
//
// A block's task used to deposit into an array of its own cells only, so
// CIC weight that landed in a neighboring block, in a cell a finer level
// covers, or outside the domain was dropped: mass was lost at block,
// coarse-fine and domain boundaries. Here each share is resolved against
// every level's boxes before it is projected:
//
// - a cell held by any grid of its level receives the share there;
// - a cell a finer level covers shares it among the finer cells under it, in
//   proportion to their image-plane area, in the fine layer along the line of
//   sight that holds the particle, recursing where those are covered;
// - a cell no grid of its level holds passes it to the coarser cell that
//   contains it;
// - periodic axes wrap; on the others, a share leaving the domain is folded
//   back into the boundary cell.
//
// The receiving cell's footprint is then spread over the image's pixels by
// overlap area. Each block's task still writes a partial image, so depositing
// into cells of other blocks is safe: the partial images are summed.
class HierarchyDeposit {
 public:
  HierarchyDeposit(const std::vector<LevelMeta> &levels,
                   const std::vector<BoxSpatialIndex> &box_indices, int axis,
                   const double rect[4], int out_nx, int out_ny,
                   std::span<std::uint8_t> out)
      : levels_(levels), box_indices_(box_indices), axis_(axis), out_nx_(out_nx), out_ny_(out_ny), out_(out) {
    u_axis_ = (axis == 0) ? 1 : 0;
    v_axis_ = (axis == 2) ? 1 : 2;
    rect_u_lo_ = std::min(rect[0], rect[2]);
    rect_u_hi_ = std::max(rect[0], rect[2]);
    rect_v_lo_ = std::min(rect[1], rect[3]);
    rect_v_hi_ = std::max(rect[1], rect[3]);
    out_du_ = (rect_u_hi_ - rect_u_lo_) / static_cast<double>(out_nx);
    out_dv_ = (rect_v_hi_ - rect_v_lo_) / static_cast<double>(out_ny);

    // Each level's domain, from the level-0 boxes' extent refined down.
    domain_lo_.resize(levels.size());
    domain_hi_.resize(levels.size());
    int32_t lo0[3] = {std::numeric_limits<int32_t>::max(),
                      std::numeric_limits<int32_t>::max(),
                      std::numeric_limits<int32_t>::max()};
    int32_t hi0[3] = {std::numeric_limits<int32_t>::min(),
                      std::numeric_limits<int32_t>::min(),
                      std::numeric_limits<int32_t>::min()};
    if (!box_indices.empty()) {
      if (const auto *b = box_indices.front().bounds()) {
        lo0[0] = b->lo.x;
        lo0[1] = b->lo.y;
        lo0[2] = b->lo.z;
        hi0[0] = b->hi.x;
        hi0[1] = b->hi.y;
        hi0[2] = b->hi.z;
      }
    }
    int64_t ratio = 1;
    for (std::size_t lev = 0; lev < levels.size(); ++lev) {
      if (lev > 0) {
        ratio *= std::max(1, levels[lev - 1].geom.ref_ratio);
      }
      const auto &g = levels[lev].geom;
      const auto &g0 = levels.front().geom;
      for (int ax = 0; ax < 3; ++ax) {
        domain_lo_[lev][ax] = static_cast<int32_t>(
            (static_cast<int64_t>(lo0[ax]) - g0.index_origin[ax]) * ratio + g.index_origin[ax]);
        domain_hi_[lev][ax] = static_cast<int32_t>(
            (static_cast<int64_t>(hi0[ax]) + 1 - g0.index_origin[ax]) * ratio - 1 +
            g.index_origin[ax]);
      }
    }
  }

  // The particle's four CIC shares at `level`.
  void deposit_particle(int level, double u, double v, double a, double mass) {
    const auto &g = levels_[static_cast<std::size_t>(level)].geom;
    const double du = g.dx[u_axis_];
    const double dv = g.dx[v_axis_];
    const double u_center0 = g.x0[u_axis_] + (0.5 - static_cast<double>(g.index_origin[u_axis_])) * du;
    const double v_center0 = g.x0[v_axis_] + (0.5 - static_cast<double>(g.index_origin[v_axis_])) * dv;
    const double su = (u - u_center0) / du;
    const double sv = (v - v_center0) / dv;
    const int iu0 = static_cast<int>(std::floor(su));
    const int iv0 = static_cast<int>(std::floor(sv));
    const double tu = su - static_cast<double>(iu0);
    const double tv = sv - static_cast<double>(iv0);
    const int ia = static_cast<int>(std::floor((a - g.x0[axis_]) / g.dx[axis_])) +
                   g.index_origin[axis_];
    const double shares[4][3] = {{0.0, 0.0, (1.0 - tu) * (1.0 - tv)},
                                 {1.0, 0.0, tu * (1.0 - tv)},
                                 {0.0, 1.0, (1.0 - tu) * tv},
                                 {1.0, 1.0, tu * tv}};
    for (const auto &share : shares) {
      if (share[2] == 0.0) {
        continue;
      }
      int32_t cell[3];
      cell[u_axis_] = iu0 + static_cast<int>(share[0]);
      cell[v_axis_] = iv0 + static_cast<int>(share[1]);
      cell[axis_] = ia;
      resolve(level, cell, mass * share[2], a, Direction::kAny);
    }
  }

  std::size_t deposited_cells() const { return deposited_cells_; }

 private:
  enum class Direction { kAny, kNoRefine, kNoCoarsen };

  static int32_t floor_divide(int64_t value, int64_t divisor) {
    int64_t q = value / divisor;
    if ((value % divisor != 0) && (value < 0)) {
      --q;
    }
    return static_cast<int32_t>(q);
  }

  void fold(int level, int32_t cell[3]) const {
    const auto &g = levels_[static_cast<std::size_t>(level)].geom;
    for (int ax = 0; ax < 3; ++ax) {
      const int32_t lo = domain_lo_[static_cast<std::size_t>(level)][ax];
      const int32_t hi = domain_hi_[static_cast<std::size_t>(level)][ax];
      if (g.is_periodic[ax]) {
        const int64_t period = static_cast<int64_t>(hi) - lo + 1;
        int64_t offset = (static_cast<int64_t>(cell[ax]) - lo) % period;
        if (offset < 0) {
          offset += period;
        }
        cell[ax] = static_cast<int32_t>(lo + offset);
      } else {
        cell[ax] = std::clamp(cell[ax], lo, hi);
      }
    }
  }

  bool in_grid(int level, const int32_t cell[3]) const {
    return box_indices_[static_cast<std::size_t>(level)].contains(cell);
  }

  bool any_grid_meets(int level, const int32_t lo[3], const int32_t hi[3]) const {
    return box_indices_[static_cast<std::size_t>(level)].intersects(lo, hi);
  }

  bool resolve(int level, int32_t cell_in[3], double weight, double a, Direction direction) {
    int32_t cell[3] = {cell_in[0], cell_in[1], cell_in[2]};
    fold(level, cell);
    const auto &g = levels_[static_cast<std::size_t>(level)].geom;
    if (!in_grid(level, cell)) {
      if (direction == Direction::kNoCoarsen) {
        return false;
      }
      if (level == 0) {
        // Level 0 covers the domain in a valid hierarchy; keep the weight.
        project(level, cell, weight);
        return true;
      }
      const auto &coarse = levels_[static_cast<std::size_t>(level - 1)].geom;
      const int ratio = std::max(1, coarse.ref_ratio);
      int32_t parent[3];
      for (int ax = 0; ax < 3; ++ax) {
        parent[ax] = floor_divide(static_cast<int64_t>(cell[ax]) - g.index_origin[ax], ratio) +
                     coarse.index_origin[ax];
      }
      return resolve(level - 1, parent, weight, a, Direction::kNoRefine);
    }
    const int finer = level + 1;
    if (direction != Direction::kNoRefine &&
        static_cast<std::size_t>(finer) < levels_.size()) {
      const auto &fg = levels_[static_cast<std::size_t>(finer)].geom;
      const int ratio = std::max(1, g.ref_ratio);
      int32_t lo[3];
      int32_t hi[3];
      for (int ax = 0; ax < 3; ++ax) {
        lo[ax] = static_cast<int32_t>((static_cast<int64_t>(cell[ax]) - g.index_origin[ax]) * ratio +
                                      fg.index_origin[ax]);
        hi[ax] = lo[ax] + ratio - 1;
      }
      if (any_grid_meets(finer, lo, hi)) {
        int32_t layer = static_cast<int32_t>(std::floor((a - fg.x0[axis_]) / fg.dx[axis_])) +
                        fg.index_origin[axis_];
        layer = std::clamp(layer, lo[axis_], hi[axis_]);
        const double share = weight / (static_cast<double>(ratio) * static_cast<double>(ratio));
        double kept = 0.0;
        for (int32_t cv = lo[v_axis_]; cv <= hi[v_axis_]; ++cv) {
          for (int32_t cu = lo[u_axis_]; cu <= hi[u_axis_]; ++cu) {
            int32_t child[3];
            child[u_axis_] = cu;
            child[v_axis_] = cv;
            child[axis_] = layer;
            if (!resolve(finer, child, share, a, Direction::kNoCoarsen)) {
              kept += share;
            }
          }
        }
        if (kept > 0.0) {
          project(level, cell, kept);
        }
        return true;
      }
    }
    project(level, cell, weight);
    return true;
  }

  // Spread a cell's weight over the pixels its image-plane footprint meets.
  void project(int level, const int32_t cell[3], double weight) {
    ++deposited_cells_;
    const auto &g = levels_[static_cast<std::size_t>(level)].geom;
    const double du = g.dx[u_axis_];
    const double dv = g.dx[v_axis_];
    const double u_lo = g.x0[u_axis_] + static_cast<double>(cell[u_axis_] - g.index_origin[u_axis_]) * du;
    const double u_hi = u_lo + du;
    const double v_lo = g.x0[v_axis_] + static_cast<double>(cell[v_axis_] - g.index_origin[v_axis_]) * dv;
    const double v_hi = v_lo + dv;
    if (u_hi <= rect_u_lo_ || u_lo >= rect_u_hi_ || v_hi <= rect_v_lo_ || v_lo >= rect_v_hi_) {
      return;
    }
    int ix_lo = static_cast<int>(std::floor((u_lo - rect_u_lo_) / out_du_));
    int ix_hi = static_cast<int>(std::floor((u_hi - rect_u_lo_) / out_du_));
    int iy_lo = static_cast<int>(std::floor((v_lo - rect_v_lo_) / out_dv_));
    int iy_hi = static_cast<int>(std::floor((v_hi - rect_v_lo_) / out_dv_));
    ix_lo = std::max(0, ix_lo);
    iy_lo = std::max(0, iy_lo);
    ix_hi = std::min(out_nx_ - 1, ix_hi);
    iy_hi = std::min(out_ny_ - 1, iy_hi);
    const double cell_area = du * dv;
    for (int iy = iy_lo; iy <= iy_hi; ++iy) {
      const double pv_lo = rect_v_lo_ + static_cast<double>(iy) * out_dv_;
      const double ov = std::max(0.0, std::min(v_hi, pv_lo + out_dv_) - std::max(v_lo, pv_lo));
      if (ov <= 0.0) {
        continue;
      }
      for (int ix = ix_lo; ix <= ix_hi; ++ix) {
        const double pu_lo = rect_u_lo_ + static_cast<double>(ix) * out_du_;
        const double ou = std::max(0.0, std::min(u_hi, pu_lo + out_du_) - std::max(u_lo, pu_lo));
        if (ou <= 0.0) {
          continue;
        }
        const auto index = static_cast<std::size_t>(iy) * static_cast<std::size_t>(out_nx_) +
                           static_cast<std::size_t>(ix);
        store_buffer_scalar<double>(
            out_.data(), index,
            load_buffer_scalar<double>(out_.data(), index) + weight * (ou * ov) / cell_area);
      }
    }
  }

  const std::vector<LevelMeta> &levels_;
  const std::vector<BoxSpatialIndex> &box_indices_;
  int axis_ = 2;
  int u_axis_ = 0;
  int v_axis_ = 1;
  int out_nx_ = 0;
  int out_ny_ = 0;
  std::span<std::uint8_t> out_;
  double rect_u_lo_ = 0.0;
  double rect_u_hi_ = 0.0;
  double rect_v_lo_ = 0.0;
  double rect_v_hi_ = 0.0;
  double out_du_ = 0.0;
  double out_dv_ = 0.0;
  std::vector<std::array<int32_t, 3>> domain_lo_;
  std::vector<std::array<int32_t, 3>> domain_hi_;
  std::size_t deposited_cells_ = 0;
};

// The step whose level list holds `level`: the executor hands kernels a
// reference into the run's metadata.
std::size_t step_containing(const RunMeta &meta, const LevelMeta &level) {
  for (const auto &step : meta.steps) {
    for (const auto &candidate : step.levels) {
      if (&candidate == &level) {
        return static_cast<std::size_t>(&step - meta.steps.data());
      }
    }
  }
  if (meta.steps.size() == 1) {
    return 0;
  }
  throw std::runtime_error("particle_cic_projection_accumulate: cannot find the block's step");
}

} // namespace

void register_particle_kernels(KernelRegistry &registry) {
  {
    using Params = ParticleFieldParams;

    /**
     * @brief Loads one particle field chunk and converts its values to double.
     * @par Chunk inputs None; the chunk is read from the dataset backend.
     * @par Typed parameters `particle_type` and `field_name` identify the
     * particle field.
     * @par Chunk outputs `outputs[0]` is a dynamically sized f64 particle
     * array.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_load_field_chunk_f64",
                   .n_inputs = 0,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t block, std::span<const ChunkBuffer>,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");
          if (params.particle_type.empty() || params.field_name.empty()) {
            throw std::runtime_error("particle_load_field_chunk_f64 requires "
                                     "particle_type and field_name");
          }

          const auto &dataset = current_dataset();
          if (!dataset.backend) {
            throw std::runtime_error(
                "particle_load_field_chunk_f64: missing dataset backend");
          }
          auto data = dataset.backend->read_particle_field_chunk(
              params.particle_type, params.field_name, block);
          const std::size_t n =
              static_cast<std::size_t>(std::max<int64_t>(0, data.count));
          auto out_values = outputs[0].mutable_dynamic_array<double>();
          auto *out = out_values.byte_data();

          if (data.dtype == "float64") {
            if (data.bytes.size() < n * sizeof(double)) {
              throw std::runtime_error(
                  "particle_load_field_chunk_f64: short float64 payload");
            }
            std::memcpy(out, data.bytes.data(), n * sizeof(double));
          } else if (data.dtype == "float32") {
            if (data.bytes.size() < n * sizeof(float)) {
              throw std::runtime_error(
                  "particle_load_field_chunk_f64: short float32 payload");
            }
            for (std::size_t i = 0; i < n; ++i) {
              store_buffer_scalar<double>(
                  out, i,
                  static_cast<double>(
                      load_buffer_scalar<float>(data.bytes.data(), i)));
            }
          } else if (data.dtype == "int64") {
            if (data.bytes.size() < n * sizeof(int64_t)) {
              throw std::runtime_error(
                  "particle_load_field_chunk_f64: short int64 payload");
            }
            for (std::size_t i = 0; i < n; ++i) {
              store_buffer_scalar<double>(
                  out, i,
                  static_cast<double>(
                      load_buffer_scalar<int64_t>(data.bytes.data(), i)));
            }
          } else {
            throw std::runtime_error(
                "particle_load_field_chunk_f64: unsupported particle dtype '" +
                data.dtype + "'");
          }
          outputs[0].commit_dynamic_extent(n);
          return hpx::make_ready_future();
        },
        [](const DynamicOutputBoundContext &context)
            -> std::optional<std::uint64_t> {
          const auto &params = context.params<Params>();
          if (params.particle_type.empty())
            return std::nullopt;
          return context.data.estimate_particle_chunk_records(
              params.particle_type, context.block);
        });
  }
  {
    using Params = ParticleCicGridParams;

    /**
     * @brief Deposits particle mass onto an AMR grid with cloud-in-cell
     * weighting.
     * @par Chunk inputs None; particle position and mass grids are read from
     * the dataset backend.
     * @par Typed parameters `particle_type`, `level_index`, `axis`,
     * `axis_bounds`, optional `mass_max`, and `covered_boxes` select particles
     * and excluded AMR cells.
     * @par Chunk outputs `outputs[0]` is an f64 block grid of deposited mass
     * density.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_cic_grid_accumulate",
                   .n_inputs = 0,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block, std::span<const ChunkBuffer>,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");

          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          if (params.particle_type.empty()) {
            throw std::runtime_error(
                "particle_cic_grid_accumulate: missing particle_type");
          }
          if (params.axis < 0 || params.axis > 2) {
            throw std::runtime_error(
                "particle_cic_grid_accumulate: axis must be 0, 1, or 2");
          }
          if (params.level_index < 0) {
            throw std::runtime_error(
                "particle_cic_grid_accumulate: missing level_index");
          }
          if (block < 0 ||
              static_cast<std::size_t>(block) >= level.boxes.size()) {
            throw std::runtime_error(
                "particle_cic_grid_accumulate: block index out of range");
          }

          const auto &dataset = current_dataset();
          if (!dataset.backend) {
            throw std::runtime_error(
                "particle_cic_grid_accumulate: missing dataset backend");
          }
          auto px = dataset.backend->read_particle_field_grid(
              params.particle_type, "x", params.level_index, block);
          auto py = dataset.backend->read_particle_field_grid(
              params.particle_type, "y", params.level_index, block);
          auto pz = dataset.backend->read_particle_field_grid(
              params.particle_type, "z", params.level_index, block);
          auto pm = dataset.backend->read_particle_field_grid(
              params.particle_type, "mass", params.level_index, block);

          std::vector<double> px_vals;
          std::vector<double> py_vals;
          std::vector<double> pz_vals;
          std::vector<double> pm_vals;
          append_particle_values_as_f64(px, "x", "particle_cic_grid_accumulate",
                                        px_vals);
          append_particle_values_as_f64(py, "y", "particle_cic_grid_accumulate",
                                        py_vals);
          append_particle_values_as_f64(pz, "z", "particle_cic_grid_accumulate",
                                        pz_vals);
          append_particle_values_as_f64(
              pm, "mass", "particle_cic_grid_accumulate", pm_vals);

          const std::size_t n =
              std::min(std::min(px_vals.size(), py_vals.size()),
                       std::min(pz_vals.size(), pm_vals.size()));
          px_vals.resize(n);
          py_vals.resize(n);
          pz_vals.resize(n);
          pm_vals.resize(n);

          const int axis = params.axis;
          const int u_axis = (axis == 0) ? 1 : 0;
          const int v_axis = (axis == 2) ? 1 : 2;

          const auto &box = level.boxes[static_cast<std::size_t>(block)];
          const int box_lo[3] = {box.lo.x, box.lo.y, box.lo.z};
          const int box_hi[3] = {box.hi.x, box.hi.y, box.hi.z};
          const int nx = box_hi[0] - box_lo[0] + 1;
          const int ny = box_hi[1] - box_lo[1] + 1;
          const int nz = box_hi[2] - box_lo[2] + 1;
          if (nx <= 0 || ny <= 0 || nz <= 0) {
            return hpx::make_ready_future();
          }

          const double x0[3] = {level.geom.x0[0], level.geom.x0[1],
                                level.geom.x0[2]};
          const double dx[3] = {level.geom.dx[0], level.geom.dx[1],
                                level.geom.dx[2]};
          const int origin[3] = {level.geom.index_origin[0],
                                 level.geom.index_origin[1],
                                 level.geom.index_origin[2]};
          if (dx[0] == 0.0 || dx[1] == 0.0 || dx[2] == 0.0) {
            return hpx::make_ready_future();
          }

          const std::size_t out_elems = static_cast<std::size_t>(nx) *
                                        static_cast<std::size_t>(ny) *
                                        static_cast<std::size_t>(nz);
          auto out = outputs[0].mutable_byte_view();
          std::fill(out.begin(), out.end(), std::uint8_t{0});

          const double cell_volume = dx[0] * dx[1] * dx[2];
          if (!(cell_volume > 0.0)) {
            return hpx::make_ready_future();
          }

          const double a_lo =
              std::min(params.axis_bounds[0], params.axis_bounds[1]);
          const double a_hi =
              std::max(params.axis_bounds[0], params.axis_bounds[1]);

          auto covered = [&](int i, int j, int k) -> bool {
            if (!params.covered_boxes) {
              return false;
            }
            for (const auto &b : *params.covered_boxes) {
              if (covered_box_contains(b, i, j, k)) {
                return true;
              }
            }
            return false;
          };

          auto out_index = [ny, nz](int i, int j, int k) -> std::size_t {
            return static_cast<std::size_t>((i * ny + j) * nz + k);
          };

          const double *coord[3] = {px_vals.data(), py_vals.data(),
                                    pz_vals.data()};
          const double du = dx[u_axis];
          const double dv = dx[v_axis];
          const double da = dx[axis];

          const double u_center0 =
              x0[u_axis] + (0.5 - static_cast<double>(origin[u_axis])) * du;
          const double v_center0 =
              x0[v_axis] + (0.5 - static_cast<double>(origin[v_axis])) * dv;
          const double a_cell_lo =
              x0[axis] +
              (static_cast<double>(box_lo[axis] - origin[axis])) * da;
          const double a_cell_hi =
              x0[axis] +
              (static_cast<double>(box_hi[axis] + 1 - origin[axis])) * da;

          auto add_density = [&](int iu, int iv, int ia, double wmass) {
            int ii = 0;
            int jj = 0;
            int kk = 0;
            if (axis == 0) {
              ii = ia;
              jj = iu;
              kk = iv;
            } else if (axis == 1) {
              ii = iu;
              jj = ia;
              kk = iv;
            } else {
              ii = iu;
              jj = iv;
              kk = ia;
            }
            if (ii < box_lo[0] || ii > box_hi[0] || jj < box_lo[1] ||
                jj > box_hi[1] || kk < box_lo[2] || kk > box_hi[2] ||
                covered(ii, jj, kk)) {
              return;
            }
            const int i_local = ii - box_lo[0];
            const int j_local = jj - box_lo[1];
            const int k_local = kk - box_lo[2];
            if (i_local < 0 || i_local >= nx || j_local < 0 || j_local >= ny ||
                k_local < 0 || k_local >= nz) {
              return;
            }
            const auto index = out_index(i_local, j_local, k_local);
            store_buffer_scalar<double>(
                out.data(), index,
                load_buffer_scalar<double>(out.data(), index) +
                    wmass / cell_volume);
          };

          for (std::size_t p = 0; p < n; ++p) {
            const double u = coord[u_axis][p];
            const double v = coord[v_axis][p];
            const double a = coord[axis][p];
            const double m = pm_vals[p];
            if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(a) ||
                !std::isfinite(m)) {
              continue;
            }
            if (m <= 0.0) {
              continue;
            }
            if (std::isfinite(params.mass_max) && m > params.mass_max) {
              continue;
            }
            if (a < a_lo || a > a_hi) {
              continue;
            }
            if (a < a_cell_lo || a >= a_cell_hi) {
              continue;
            }

            const double su = (u - u_center0) / du;
            const double sv = (v - v_center0) / dv;
            const int iu0 = static_cast<int>(std::floor(su));
            const int iv0 = static_cast<int>(std::floor(sv));
            const double tu = su - static_cast<double>(iu0);
            const double tv = sv - static_cast<double>(iv0);
            const int ia = static_cast<int>(std::floor((a - x0[axis]) / da)) +
                           origin[axis];

            add_density(iu0, iv0, ia, m * (1.0 - tu) * (1.0 - tv));
            add_density(iu0 + 1, iv0, ia, m * tu * (1.0 - tv));
            add_density(iu0, iv0 + 1, ia, m * (1.0 - tu) * tv);
            add_density(iu0 + 1, iv0 + 1, ia, m * tu * tv);
          }

          return hpx::make_ready_future();
        });
  }
  {
    using Params = ParticleCicProjectionParams;

    /**
     * @brief Projects particles onto an image plane with cloud-in-cell
     * weighting.
     * @par Chunk inputs None; particle position and mass grids are read from
     * the dataset backend.
     * @par Typed parameters `particle_type`, `level_index`, `axis`,
     * `axis_bounds`, `rect`, `resolution`, optional `mass_max`, and
     * `covered_boxes` define the selected particles, image, and excluded AMR
     * cells.
     * @par Chunk outputs `outputs[0]` is an f64 image of deposited particle
     * mass.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_cic_projection_accumulate",
                   .n_inputs = 0,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block, std::span<const ChunkBuffer>,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");

          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          const int out_nx = params.resolution[0];
          const int out_ny = params.resolution[1];
          if (out_nx <= 0 || out_ny <= 0) {
            throw std::runtime_error(
                "particle_cic_projection_accumulate: invalid resolution");
          }
          auto out = outputs[0].mutable_byte_view();
          std::fill(out.begin(), out.end(), std::uint8_t{0});

          if (params.particle_type.empty()) {
            throw std::runtime_error(
                "particle_cic_projection_accumulate: missing particle_type");
          }
          if (params.axis < 0 || params.axis > 2) {
            throw std::runtime_error(
                "particle_cic_projection_accumulate: axis must be 0, 1, or 2");
          }
          if (params.level_index < 0) {
            throw std::runtime_error(
                "particle_cic_projection_accumulate: missing level_index");
          }
          if (block < 0 ||
              static_cast<std::size_t>(block) >= level.boxes.size()) {
            throw std::runtime_error(
                "particle_cic_projection_accumulate: block index out of range");
          }

          const auto &dataset = current_dataset();
          if (!dataset.backend) {
            throw std::runtime_error(
                "particle_cic_projection_accumulate: missing dataset backend");
          }
          auto px = dataset.backend->read_particle_field_grid(
              params.particle_type, "x", params.level_index, block);
          auto py = dataset.backend->read_particle_field_grid(
              params.particle_type, "y", params.level_index, block);
          auto pz = dataset.backend->read_particle_field_grid(
              params.particle_type, "z", params.level_index, block);
          auto pm = dataset.backend->read_particle_field_grid(
              params.particle_type, "mass", params.level_index, block);

          std::vector<double> px_vals;
          std::vector<double> py_vals;
          std::vector<double> pz_vals;
          std::vector<double> pm_vals;
          append_particle_values_as_f64(
              px, "x", "particle_cic_projection_accumulate", px_vals);
          append_particle_values_as_f64(
              py, "y", "particle_cic_projection_accumulate", py_vals);
          append_particle_values_as_f64(
              pz, "z", "particle_cic_projection_accumulate", pz_vals);
          append_particle_values_as_f64(
              pm, "mass", "particle_cic_projection_accumulate", pm_vals);

          const std::size_t n =
              std::min(std::min(px_vals.size(), py_vals.size()),
                       std::min(pz_vals.size(), pm_vals.size()));
          px_vals.resize(n);
          py_vals.resize(n);
          pz_vals.resize(n);
          pm_vals.resize(n);

          const int axis = params.axis;
          const int u_axis = (axis == 0) ? 1 : 0;
          const int v_axis = (axis == 2) ? 1 : 2;

          if (level.geom.dx[0] == 0.0 || level.geom.dx[1] == 0.0 ||
              level.geom.dx[2] == 0.0) {
            return hpx::make_ready_future();
          }

          const double a_lo =
              std::min(params.axis_bounds[0], params.axis_bounds[1]);
          const double a_hi =
              std::max(params.axis_bounds[0], params.axis_bounds[1]);
          const double rect[4] = {params.rect[0], params.rect[1], params.rect[2],
                                  params.rect[3]};
          if (!(std::abs(rect[2] - rect[0]) > 0.0) || !(std::abs(rect[3] - rect[1]) > 0.0)) {
            return hpx::make_ready_future();
          }
          const auto &ctx = execution_context(current_execution_context_id());
          const auto step = step_containing(ctx.meta, level);
          HierarchyDeposit deposit(ctx.meta.steps[step].levels, ctx.particle_box_indices.at(step),
                                   axis, rect, out_nx, out_ny, out);

          const double *coord[3] = {px_vals.data(), py_vals.data(),
                                    pz_vals.data()};
          std::size_t candidates = 0;
          std::size_t bounds_skips = 0;
          const std::size_t covered_skips = 0;
          for (std::size_t p = 0; p < n; ++p) {
            ++candidates;
            const double u = coord[u_axis][p];
            const double v = coord[v_axis][p];
            const double a = coord[axis][p];
            const double m = pm_vals[p];
            if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(a) ||
                !std::isfinite(m)) {
              ++bounds_skips;
              continue;
            }
            if (m <= 0.0) {
              ++bounds_skips;
              continue;
            }
            if (std::isfinite(params.mass_max) && m > params.mass_max) {
              ++bounds_skips;
              continue;
            }
            if (a < a_lo || a > a_hi) {
              ++bounds_skips;
              continue;
            }
            deposit.deposit_particle(params.level_index, u, v, a, m);
          }
          const std::size_t deposited_cells = deposit.deposited_cells();
          double total_sum = 0.0;
          for (std::size_t idx = 0; idx < static_cast<std::size_t>(out_nx) *
                                              static_cast<std::size_t>(out_ny);
               ++idx) {
            total_sum += load_buffer_scalar<double>(out.data(), idx);
          }
          log_projection_kernel_summary(
              "particle_cic_projection_accumulate", params.level_index, block,
              covered_box_count(params.covered_boxes), candidates,
              covered_skips, bounds_skips, deposited_cells, total_sum);
          return hpx::make_ready_future();
        });
  }
  {
    using Params = ScalarParams;

    /**
     * @brief Marks particle values exactly equal to a scalar.
     * @par Chunk inputs `inputs[0]` is an f64 particle array.
     * @par Typed parameters `scalar` is the comparison value.
     * @par Chunk outputs `outputs[0]` is a same-length u8 mask.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_eq_mask",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto in = inputs[0].array<double>();
          auto out = outputs[0].mutable_array<std::uint8_t>();
          const std::size_t n = in.extent(0);
          for (std::size_t i = 0; i < n; ++i) {
            out(i) = (in(i) == params.scalar) ? 1 : 0;
          }
          return hpx::make_ready_future();
        });

    /**
     * @brief Marks particle values whose absolute value is below a scalar.
     * @par Chunk inputs `inputs[0]` is an f64 particle array.
     * @par Typed parameters `scalar` is the strict absolute-value
     * threshold.
     * @par Chunk outputs `outputs[0]` is a same-length u8 mask.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_abs_lt_mask",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto in = inputs[0].array<double>();
          auto out = outputs[0].mutable_array<std::uint8_t>();
          const std::size_t n = in.extent(0);
          for (std::size_t i = 0; i < n; ++i) {
            out(i) = (std::abs(in(i)) < params.scalar) ? 1 : 0;
          }
          return hpx::make_ready_future();
        });

    /**
     * @brief Marks particle values less than or equal to a scalar.
     * @par Chunk inputs `inputs[0]` is an f64 particle array.
     * @par Typed parameters `scalar` is the inclusive upper bound.
     * @par Chunk outputs `outputs[0]` is a same-length u8 mask.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_le_mask",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto in = inputs[0].array<double>();
          auto out = outputs[0].mutable_array<std::uint8_t>();
          const std::size_t n = in.extent(0);
          for (std::size_t i = 0; i < n; ++i) {
            out(i) = (in(i) <= params.scalar) ? 1 : 0;
          }
          return hpx::make_ready_future();
        });

    /**
     * @brief Marks particle values greater than a scalar.
     * @par Chunk inputs `inputs[0]` is an f64 particle array.
     * @par Typed parameters `scalar` is the strict lower bound.
     * @par Chunk outputs `outputs[0]` is a same-length u8 mask.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_gt_mask",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto in = inputs[0].array<double>();
          auto out = outputs[0].mutable_array<std::uint8_t>();
          const std::size_t n = in.extent(0);
          for (std::size_t i = 0; i < n; ++i) {
            out(i) = (in(i) > params.scalar) ? 1 : 0;
          }
          return hpx::make_ready_future();
        });

    auto register_scalar_comparison = [&registry](const std::string &name,
                                                  auto comparison) {
      registry.register_typed_kernel<Params>(
          KernelDesc{.name = name,
                     .n_inputs = 1,
                     .n_outputs = 1,
                     .needs_neighbors = false},
          [comparison, name](
              const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
              const NeighborViews &, std::span<ChunkBuffer> outputs,
              const KernelParamsIR &kernel_params) {
            const auto &params =
                require_kernel_params<Params>(kernel_params, name.c_str());
            if (inputs.empty() || outputs.empty()) {
              return hpx::make_ready_future();
            }
            const auto in = inputs[0].array<double>();
            auto out = outputs[0].mutable_array<std::uint8_t>();
            for (std::size_t i = 0; i < in.extent(0); ++i) {
              out(i) = comparison(in(i), params.scalar) ? 1 : 0;
            }
            return hpx::make_ready_future();
          });
    };
    register_scalar_comparison("particle_lt_mask",
                               [](double a, double b) { return a < b; });
    register_scalar_comparison("particle_ge_mask",
                               [](double a, double b) { return a >= b; });
    register_scalar_comparison("particle_ne_mask",
                               [](double a, double b) { return a != b; });
  }
  {
    using Params = ValuesParams;

    /**
     * @brief Marks particle values contained in a configured set.
     * @par Chunk inputs `inputs[0]` is an f64 particle array.
     * @par Typed parameters `values` is the set of exact matches.
     * @par Chunk outputs `outputs[0]` is a same-length u8 mask.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_isin_mask",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto in = inputs[0].array<double>();
          auto out = outputs[0].mutable_array<std::uint8_t>();
          const std::size_t n = in.extent(0);
          for (std::size_t i = 0; i < n; ++i) {
            bool found = false;
            for (double x : params.values) {
              if (in(i) == x) {
                found = true;
                break;
              }
            }
            out(i) = found ? 1 : 0;
          }
          return hpx::make_ready_future();
        });
  }
  {
    using Params = FiniteOnlyParams;

    /**
     * @brief Finds the minimum particle value, optionally ignoring non-finite
     * values.
     * @par Chunk inputs `inputs[0]` is an f64 particle array.
     * @par Typed parameters `finite_only` controls whether non-finite
     * values are skipped.
     * @par Chunk outputs `outputs[0]` is one f64 minimum value, or positive
     * infinity when no eligible value exists.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_min",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto in = inputs[0].array<double>();
          const std::size_t n = in.extent(0);
          double out_v = std::numeric_limits<double>::infinity();
          bool any = false;
          for (std::size_t i = 0; i < n; ++i) {
            const double v = in(i);
            if (params.finite_only && !std::isfinite(v)) {
              continue;
            }
            if (!any || v < out_v) {
              out_v = v;
              any = true;
            }
          }
          outputs[0].mutable_array<double>()(0) =
              any ? out_v : std::numeric_limits<double>::infinity();
          return hpx::make_ready_future();
        });

    /**
     * @brief Finds the maximum particle value, optionally ignoring non-finite
     * values.
     * @par Chunk inputs `inputs[0]` is an f64 particle array.
     * @par Typed parameters `finite_only` controls whether non-finite
     * values are skipped.
     * @par Chunk outputs `outputs[0]` is one f64 maximum value, or negative
     * infinity when no eligible value exists.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_max",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle_max");
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto in = inputs[0].array<double>();
          const std::size_t n = in.extent(0);
          double out_v = -std::numeric_limits<double>::infinity();
          bool any = false;
          for (std::size_t i = 0; i < n; ++i) {
            const double v = in(i);
            if (params.finite_only && !std::isfinite(v)) {
              continue;
            }
            if (!any || v > out_v) {
              out_v = v;
              any = true;
            }
          }
          outputs[0].mutable_array<double>()(0) =
              any ? out_v : -std::numeric_limits<double>::infinity();
          return hpx::make_ready_future();
        });
  }
  {
    using Params = ParticleHistogramParams;

    auto make_histogram_kernel = [](bool weighted) -> KernelFn {
      return [weighted](const LevelMeta &, int32_t,
                        std::span<const ChunkBuffer> inputs,
                        const NeighborViews &, std::span<ChunkBuffer> outputs,
                        const KernelParamsIR &kernel_params) {
        const auto &params =
            require_kernel_params<Params>(kernel_params, "particle kernel");
        if (outputs.empty() || params.edges.size() < 2) {
          return hpx::make_ready_future();
        }
        const std::size_t expected_inputs = weighted ? 2 : 1;
        if (inputs.size() != expected_inputs) {
          throw BufferContractError(
              BufferContractReason::kInvalidExtent,
              weighted
                  ? "particle_histogram1d_weighted requires values and weights"
                  : "particle_histogram1d requires exactly one input");
        }
        const std::size_t bins = params.edges.size() - 1;
        auto out = outputs[0].mutable_array<double>();
        for (std::size_t i = 0; i < bins; ++i)
          out(i) = 0.0;
        const auto values = inputs[0].array<double>();
        const std::size_t n = values.extent(0);
        std::optional<ArrayView<const double>> weights;
        if (weighted) {
          weights = inputs[1].array<double>();
          if (weights->extent(0) != n) {
            throw BufferContractError(BufferContractReason::kInvalidExtent,
                                      "particle_histogram1d values and weights "
                                      "must have matching extents");
          }
        }
        for (std::size_t i = 0; i < n; ++i) {
          const double x = values(i);
          if (!std::isfinite(x) || x < params.edges.front() ||
              x > params.edges.back()) {
            continue;
          }
          std::size_t idx = bins - 1;
          if (x != params.edges.back()) {
            auto it =
                std::upper_bound(params.edges.begin(), params.edges.end(), x);
            idx = static_cast<std::size_t>(
                std::distance(params.edges.begin(), it) - 1);
          }
          if (idx >= bins) {
            continue;
          }
          double w = 1.0;
          if (weights) {
            w = (*weights)(i);
            if (!std::isfinite(w)) {
              continue;
            }
          }
          out(idx) = static_cast<double>(out(idx)) + w;
        }
        if (params.density) {
          double total = 0.0;
          for (std::size_t i = 0; i < bins; ++i) {
            total += out(i);
          }
          if (total > 0.0) {
            for (std::size_t i = 0; i < bins; ++i) {
              const double width = params.edges[i + 1] - params.edges[i];
              if (width > 0.0) {
                out(i) = static_cast<double>(out(i)) / (total * width);
              }
            }
          }
        }
        return hpx::make_ready_future();
      };
    };

    /**
     * @brief Bins particle values into a one-dimensional histogram.
     * @par Chunk inputs `inputs[0]` is an f64 particle-value array.
     * @par Typed parameters `edges` defines the bin edges and `density`
     * requests probability-density normalization.
     * @par Chunk outputs `outputs[0]` is an f64 array with `edges.size() - 1`
     * bins.
     */
    registry.register_typed_kernel<Params>(KernelDesc{.name = "particle_histogram1d",
                                        .n_inputs = 1,
                                        .n_outputs = 1,
                                        .needs_neighbors = false},
                             make_histogram_kernel(false));
    /**
     * @brief Bins particle values into a weighted one-dimensional histogram.
     * @par Chunk inputs `inputs[0]` contains f64 values and matching
     * `inputs[1]` contains f64 weights.
     * @par Typed parameters `edges` defines the bin edges and `density`
     * requests weighted probability-density normalization.
     * @par Chunk outputs `outputs[0]` is an f64 array with `edges.size() - 1`
     * bins.
     */
    registry.register_typed_kernel<Params>(KernelDesc{.name = "particle_histogram1d_weighted",
                                        .n_inputs = 2,
                                        .n_outputs = 1,
                                        .needs_neighbors = false},
                             make_histogram_kernel(true));
  }
  {
    using Params = ParticleFieldParams;

    /**
     * @brief Encodes per-chunk occurrence counts for finite particle values.
     * @par Chunk inputs None; the particle field is read from the dataset
     * backend.
     * @par Typed parameters `particle_type` and `field_name` identify the
     * particle field.
     * @par Chunk outputs `outputs[0]` is a dynamically sized opaque map from
     * f64 values to signed 64-bit occurrence counts.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_topk_modes_map",
                   .n_inputs = 0,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t block, std::span<const ChunkBuffer>,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");
          if (params.particle_type.empty() || params.field_name.empty()) {
            outputs[0].commit_dynamic_extent(0);
            return hpx::make_ready_future();
          }

          const auto &dataset = current_dataset();
          if (!dataset.backend) {
            throw std::runtime_error(
                "particle_topk_modes_map: missing dataset backend");
          }
          std::unordered_map<double, int64_t> counts;
          const auto data = dataset.backend->read_particle_field_chunk(
              params.particle_type, params.field_name, block);
          std::vector<double> values;
          append_particle_values_as_f64(data, params.field_name,
                                        "particle_topk_modes_map", values);
          for (double v : values) {
            if (!std::isfinite(v)) {
              continue;
            }
            counts[v] += 1;
          }
          std::vector<std::uint8_t> encoded;
          encode_particle_value_counts(counts, encoded);
          outputs[0].assign_dynamic_bytes(encoded);
          return hpx::make_ready_future();
        },
        [](const DynamicOutputBoundContext &context)
            -> std::optional<std::uint64_t> {
          const auto &params = context.params<Params>();
          if (params.particle_type.empty())
            return std::nullopt;
          const auto records = context.data.estimate_particle_chunk_records(
              params.particle_type, context.block);
          if (!records.has_value())
            return std::nullopt;
          return checked_add(
              sizeof(std::uint64_t),
              checked_multiply(*records,
                               sizeof(double) + sizeof(std::int64_t)));
        });
  }
  {
    using Params = TopKModesParams;

    /**
     * @brief Selects the most frequent particle values from merged occurrence
     * counts.
     * @par Chunk inputs `inputs[0]` is an opaque encoded value-count map.
     * @par Typed parameters `k` is the number of modes to return.
     * @par Chunk outputs `outputs[0]` is an f64 array of length `2 * k`: the
     * first half contains values and the second half contains their counts.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "particle_topk_modes_finalize",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto &params =
              require_kernel_params<Params>(kernel_params, "particle kernel");
          if (inputs.empty() || params.k <= 0) {
            return hpx::make_ready_future();
          }
          auto counts = decode_particle_value_counts(inputs[0].byte_view());
          std::vector<std::pair<double, int64_t>> modes;
          modes.reserve(counts.size());
          for (const auto &it : counts) {
            modes.emplace_back(it.first, it.second);
          }
          std::sort(modes.begin(), modes.end(),
                    [](const auto &a, const auto &b) {
                      if (a.second != b.second) {
                        return a.second > b.second;
                      }
                      return a.first > b.first;
                    });

          const std::size_t out_len = static_cast<std::size_t>(params.k);
          auto out = outputs[0].mutable_array<double>();
          for (std::size_t i = 0; i < out_len; ++i) {
            if (i < modes.size()) {
              out(i) = modes[i].first;
              out(out_len + i) = static_cast<double>(modes[i].second);
            } else {
              out(i) = std::numeric_limits<double>::quiet_NaN();
              out(out_len + i) = 0.0;
            }
          }
          return hpx::make_ready_future();
        });
  }
  /**
   * @brief Marks finite particle values.
   * @par Chunk inputs `inputs[0]` is an f64 particle array.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is a same-length u8 mask.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_isfinite_mask",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (inputs.empty() || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto in = inputs[0].array<double>();
        auto out = outputs[0].mutable_array<std::uint8_t>();
        const std::size_t n = in.extent(0);
        for (std::size_t i = 0; i < n; ++i) {
          out(i) = std::isfinite(in(i)) ? 1 : 0;
        }
        return hpx::make_ready_future();
      });
  /**
   * @brief Computes the elementwise logical conjunction of two particle masks.
   * @par Chunk inputs `inputs[0]` and `inputs[1]` are u8 masks.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is a dynamically sized u8 mask whose length
   * is the shorter input length.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_and_mask",
                 .n_inputs = 2,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (inputs.size() < 2 || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto a = inputs[0].array<std::uint8_t>();
        const auto b = inputs[1].array<std::uint8_t>();
        const std::size_t n = std::min(a.extent(0), b.extent(0));
        auto out = outputs[0].mutable_dynamic_array<std::uint8_t>();
        for (std::size_t i = 0; i < n; ++i) {
          out(i) = (a(i) != 0 && b(i) != 0) ? std::uint8_t{1} : std::uint8_t{0};
        }
        outputs[0].commit_dynamic_extent(n);
        return hpx::make_ready_future();
      });
  /**
   * @brief Compacts particle values selected by a mask.
   * @par Chunk inputs `inputs[0]` is an f64 value array and `inputs[1]` is a u8
   * mask.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is a dynamically sized f64 array containing
   * values whose corresponding mask entry is nonzero.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_filter",
                 .n_inputs = 2,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (inputs.size() < 2 || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto values = inputs[0].array<double>();
        const auto mask = inputs[1].array<std::uint8_t>();
        const std::size_t n = std::min(values.extent(0), mask.extent(0));
        std::size_t count = 0;
        for (std::size_t i = 0; i < n; ++i) {
          if (mask(i) != 0) {
            ++count;
          }
        }
        auto out = outputs[0].mutable_dynamic_array<double>();
        std::size_t out_idx = 0;
        for (std::size_t i = 0; i < n; ++i) {
          if (mask(i) != 0) {
            out(out_idx++) = values(i);
          }
        }
        outputs[0].commit_dynamic_extent(count);
        return hpx::make_ready_future();
      });
  auto register_particle_binary = [&registry](const std::string &name,
                                              auto operation) {
    registry.register_kernel(
        KernelDesc{.name = name,
                   .n_inputs = 2,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [operation](const LevelMeta &, int32_t,
                    std::span<const ChunkBuffer> inputs, const NeighborViews &,
                    std::span<ChunkBuffer> outputs, const KernelParamsIR &) {
          if (inputs.size() < 2 || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto a = inputs[0].array<double>();
          const auto b = inputs[1].array<double>();
          const std::size_t n = std::min(a.extent(0), b.extent(0));
          auto out = outputs[0].mutable_dynamic_array<double>();
          for (std::size_t i = 0; i < n; ++i) {
            out(i) = operation(a(i), b(i));
          }
          outputs[0].commit_dynamic_extent(n);
          return hpx::make_ready_future();
        });
  };
  register_particle_binary("particle_add",
                           [](double a, double b) { return a + b; });
  register_particle_binary("particle_multiply",
                           [](double a, double b) { return a * b; });
  register_particle_binary("particle_divide",
                           [](double a, double b) { return a / b; });
  register_particle_binary("particle_power",
                           [](double a, double b) { return std::pow(a, b); });

  auto register_particle_scalar = [&registry](const std::string &name,
                                              auto operation) {
    registry.register_typed_kernel<ScalarParams>(
        KernelDesc{.name = name,
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [operation, name](const LevelMeta &, int32_t,
                          std::span<const ChunkBuffer> inputs,
                          const NeighborViews &, std::span<ChunkBuffer> outputs,
                          const KernelParamsIR &kernel_params) {
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto &params =
              require_kernel_params<ScalarParams>(kernel_params, name.c_str());
          const auto in = inputs[0].array<double>();
          const std::size_t n = in.extent(0);
          auto out = outputs[0].mutable_dynamic_array<double>();
          for (std::size_t i = 0; i < n; ++i) {
            out(i) = operation(in(i), params.scalar);
          }
          outputs[0].commit_dynamic_extent(n);
          return hpx::make_ready_future();
        });
  };
  register_particle_scalar("particle_add_scalar",
                           [](double a, double b) { return a + b; });
  register_particle_scalar("particle_subtract_scalar",
                           [](double a, double b) { return a - b; });
  register_particle_scalar("particle_rsubtract_scalar",
                           [](double a, double b) { return b - a; });
  register_particle_scalar("particle_multiply_scalar",
                           [](double a, double b) { return a * b; });
  register_particle_scalar("particle_divide_scalar",
                           [](double a, double b) { return a / b; });
  register_particle_scalar("particle_rdivide_scalar",
                           [](double a, double b) { return b / a; });
  register_particle_scalar("particle_power_scalar",
                           [](double a, double b) { return std::pow(a, b); });
  register_particle_scalar("particle_rpower_scalar",
                           [](double a, double b) { return std::pow(b, a); });
  /**
   * @brief Subtracts two particle arrays elementwise.
   * @par Chunk inputs `inputs[0]` and `inputs[1]` are f64 particle arrays.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is their dynamically sized elementwise
   * difference, truncated to the shorter input length.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_subtract",
                 .n_inputs = 2,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (inputs.size() < 2 || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto a = inputs[0].array<double>();
        const auto b = inputs[1].array<double>();
        const std::size_t n = std::min(a.extent(0), b.extent(0));
        auto out = outputs[0].mutable_dynamic_array<double>();
        for (std::size_t i = 0; i < n; ++i) {
          out(i) = a(i) - b(i);
        }
        outputs[0].commit_dynamic_extent(n);
        return hpx::make_ready_future();
      });
  /**
   * @brief Computes elementwise Euclidean distances between two 3D point
   * arrays.
   * @par Chunk inputs Six f64 arrays ordered as first x/y/z then second x/y/z.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is a dynamically sized f64 distance array,
   * truncated to the shortest input length.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_distance3",
                 .n_inputs = 6,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (inputs.size() < 6 || outputs.empty()) {
          return hpx::make_ready_future();
        }
        std::array<ArrayView<const double>, 6> values{
            inputs[0].array<double>(), inputs[1].array<double>(),
            inputs[2].array<double>(), inputs[3].array<double>(),
            inputs[4].array<double>(), inputs[5].array<double>()};
        const std::size_t n = std::min(
            {values[0].extent(0), values[1].extent(0), values[2].extent(0),
             values[3].extent(0), values[4].extent(0), values[5].extent(0)});
        auto out = outputs[0].mutable_dynamic_array<double>();
        for (std::size_t i = 0; i < n; ++i) {
          const double dx = values[0](i) - values[3](i);
          const double dy = values[1](i) - values[4](i);
          const double dz = values[2](i) - values[5](i);
          out(i) = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        outputs[0].commit_dynamic_extent(n);
        return hpx::make_ready_future();
      });
  /**
   * @brief Sums all values in a particle array.
   * @par Chunk inputs `inputs[0]` is an f64 particle array.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is one f64 sum.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_sum",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (inputs.empty() || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto in = inputs[0].array<double>();
        const std::size_t n = in.extent(0);
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          sum += in(i);
        }
        outputs[0].mutable_array<double>()(0) = sum;
        return hpx::make_ready_future();
      });
  /**
   * @brief Counts nonzero entries in a particle mask.
   * @par Chunk inputs `inputs[0]` is a u8 particle mask.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is one signed 64-bit count.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_count",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (inputs.empty() || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto in = inputs[0].array<std::uint8_t>();
        const std::size_t n = in.extent(0);
        int64_t count = 0;
        for (std::size_t i = 0; i < n; ++i) {
          if (in(i) != 0) {
            ++count;
          }
        }
        outputs[0].mutable_array<std::int64_t>()(0) = count;
        return hpx::make_ready_future();
      });
  /**
   * @brief Returns the number of values in a double-precision particle array.
   * @par Chunk inputs `inputs[0]` is an f64 particle array.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is one signed 64-bit length.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_len_f64",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (inputs.empty() || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto n =
            static_cast<int64_t>(inputs[0].array<double>().extent(0));
        outputs[0].mutable_array<std::int64_t>()(0) = n;
        return hpx::make_ready_future();
      });
  /**
   * @brief Merges encoded particle-value occurrence counts.
   * @par Chunk inputs `inputs[0..N)` are opaque encoded value-count maps.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is the dynamically sized opaque merged map.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_value_counts_reduce",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }
        std::unordered_map<double, int64_t> merged;
        for (const auto &in_view : inputs) {
          auto counts = decode_particle_value_counts(in_view.byte_view());
          if (merged.empty()) {
            merged = std::move(counts);
            continue;
          }
          for (const auto &[value, count] : counts) {
            merged[value] += count;
          }
        }
        std::vector<std::uint8_t> encoded;
        encode_particle_value_counts(merged, encoded);
        outputs[0].assign_dynamic_bytes(encoded);
        return hpx::make_ready_future();
      },
      [](const DynamicOutputBoundContext &context)
          -> std::optional<std::uint64_t> {
        std::uint64_t input_bytes = 0;
        for (const auto &input : context.inputs) {
          if (!input.storage_known)
            return std::nullopt;
          input_bytes = checked_add(input_bytes, input.payload_bytes);
        }
        const auto output_bytes =
            std::max<std::uint64_t>(sizeof(std::uint64_t), input_bytes);
        const auto width = scalar_size(context.scalar);
        return output_bytes / width +
               static_cast<std::uint64_t>(output_bytes % width != 0);
      });
  /**
   * @brief Sums signed 64-bit scalar partial results.
   * @par Chunk inputs `inputs[0..N)` each contain one signed 64-bit scalar.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is one signed 64-bit total.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_int64_sum_reduce",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }
        int64_t total = 0;
        for (const auto &in_view : inputs) {
          if (in_view.bytes() < sizeof(int64_t)) {
            continue;
          }
          total += in_view.array<std::int64_t>()(0);
        }
        outputs[0].mutable_array<std::int64_t>()(0) = total;
        return hpx::make_ready_future();
      });
  /**
   * @brief Reduces finite scalar partial results to their minimum.
   * @par Chunk inputs `inputs[0..N)` each contain one f64 scalar.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is the smallest finite input, or positive
   * infinity when none is finite.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_scalar_min_reduce",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }
        double out_v = std::numeric_limits<double>::infinity();
        bool any = false;
        for (const auto &in_view : inputs) {
          if (in_view.bytes() < sizeof(double)) {
            continue;
          }
          const double v = in_view.array<double>()(0);
          if (!std::isfinite(v)) {
            continue;
          }
          if (!any || v < out_v) {
            out_v = v;
            any = true;
          }
        }
        outputs[0].mutable_array<double>()(0) =
            any ? out_v : std::numeric_limits<double>::infinity();
        return hpx::make_ready_future();
      });
  /**
   * @brief Reduces finite scalar partial results to their maximum.
   * @par Chunk inputs `inputs[0..N)` each contain one f64 scalar.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is the largest finite input, or negative
   * infinity when none is finite.
   */
  registry.register_kernel(
      KernelDesc{.name = "particle_scalar_max_reduce",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }
        double out_v = -std::numeric_limits<double>::infinity();
        bool any = false;
        for (const auto &in_view : inputs) {
          if (in_view.bytes() < sizeof(double)) {
            continue;
          }
          const double v = in_view.array<double>()(0);
          if (!std::isfinite(v)) {
            continue;
          }
          if (!any || v > out_v) {
            out_v = v;
            any = true;
          }
        }
        outputs[0].mutable_array<double>()(0) =
            any ? out_v : -std::numeric_limits<double>::infinity();
        return hpx::make_ready_future();
      });
}

} // namespace kangaroo
