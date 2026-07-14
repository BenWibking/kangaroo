#include "default_kernel_families.hpp"

#include "kernel_buffer_support.hpp"
#include "kernel_param_support.hpp"
#include "particle_kernel_support.hpp"
#include "projection_kernel_support.hpp"

#include "kangaroo/runtime.hpp"

namespace kangaroo {

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

          const auto &box = level.boxes[static_cast<std::size_t>(block)];
          const int box_lo[3] = {box.lo.x, box.lo.y, box.lo.z};
          const int box_hi[3] = {box.hi.x, box.hi.y, box.hi.z};

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

          const int nu = box_hi[u_axis] - box_lo[u_axis] + 1;
          const int nv = box_hi[v_axis] - box_lo[v_axis] + 1;
          if (nu <= 0 || nv <= 0) {
            return hpx::make_ready_future();
          }

          const double a_lo =
              std::min(params.axis_bounds[0], params.axis_bounds[1]);
          const double a_hi =
              std::max(params.axis_bounds[0], params.axis_bounds[1]);
          const double rect_u_lo = std::min(params.rect[0], params.rect[2]);
          const double rect_u_hi = std::max(params.rect[0], params.rect[2]);
          const double rect_v_lo = std::min(params.rect[1], params.rect[3]);
          const double rect_v_hi = std::max(params.rect[1], params.rect[3]);
          const double out_du =
              (rect_u_hi - rect_u_lo) / static_cast<double>(out_nx);
          const double out_dv =
              (rect_v_hi - rect_v_lo) / static_cast<double>(out_ny);
          if (out_du <= 0.0 || out_dv <= 0.0) {
            return hpx::make_ready_future();
          }
          const double inv_out_du = 1.0 / out_du;
          const double inv_out_dv = 1.0 / out_dv;

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

          std::vector<double> native_mass(
              static_cast<std::size_t>(nu) * static_cast<std::size_t>(nv), 0.0);
          auto native_index = [nu](int iu_local, int iv_local) -> std::size_t {
            return static_cast<std::size_t>(iv_local) *
                       static_cast<std::size_t>(nu) +
                   static_cast<std::size_t>(iu_local);
          };
          std::size_t covered_skips = 0;
          std::size_t bounds_skips = 0;
          std::size_t deposited_cells = 0;
          auto native_add = [&](int iu, int iv, int ia, double wmass) {
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
                jj > box_hi[1] || kk < box_lo[2] || kk > box_hi[2]) {
              ++bounds_skips;
              return;
            }
            if (covered(ii, jj, kk)) {
              ++covered_skips;
              return;
            }
            const int iu_local = iu - box_lo[u_axis];
            const int iv_local = iv - box_lo[v_axis];
            if (iu_local < 0 || iu_local >= nu || iv_local < 0 ||
                iv_local >= nv) {
              ++bounds_skips;
              return;
            }
            native_mass[native_index(iu_local, iv_local)] += wmass;
            ++deposited_cells;
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
          std::size_t candidates = 0;

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
            if (a < a_cell_lo || a >= a_cell_hi) {
              ++bounds_skips;
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

            native_add(iu0, iv0, ia, m * (1.0 - tu) * (1.0 - tv));
            native_add(iu0 + 1, iv0, ia, m * tu * (1.0 - tv));
            native_add(iu0, iv0 + 1, ia, m * (1.0 - tu) * tv);
            native_add(iu0 + 1, iv0 + 1, ia, m * tu * tv);
          }

          const double cell_area = du * dv;
          if (cell_area <= 0.0) {
            return hpx::make_ready_future();
          }

          for (int iv_local = 0; iv_local < nv; ++iv_local) {
            const int iv = box_lo[v_axis] + iv_local;
            const double v_center =
                x0[v_axis] +
                (static_cast<double>(iv - origin[v_axis]) + 0.5) * dv;
            const double v_lo = v_center - 0.5 * dv;
            const double v_hi = v_center + 0.5 * dv;
            if (v_hi <= rect_v_lo || v_lo >= rect_v_hi) {
              continue;
            }

            int iy_lo =
                static_cast<int>(std::floor((v_lo - rect_v_lo) * inv_out_dv));
            int iy_hi =
                static_cast<int>(std::floor((v_hi - rect_v_lo) * inv_out_dv));
            if (iy_hi < 0 || iy_lo >= out_ny) {
              continue;
            }
            iy_lo = std::max(0, iy_lo);
            iy_hi = std::min(out_ny - 1, iy_hi);

            for (int iu_local = 0; iu_local < nu; ++iu_local) {
              const double mass = native_mass[native_index(iu_local, iv_local)];
              if (mass == 0.0) {
                continue;
              }
              const int iu = box_lo[u_axis] + iu_local;
              const double u_center =
                  x0[u_axis] +
                  (static_cast<double>(iu - origin[u_axis]) + 0.5) * du;
              const double u_lo = u_center - 0.5 * du;
              const double u_hi = u_center + 0.5 * du;
              if (u_hi <= rect_u_lo || u_lo >= rect_u_hi) {
                continue;
              }

              int ix_lo =
                  static_cast<int>(std::floor((u_lo - rect_u_lo) * inv_out_du));
              int ix_hi =
                  static_cast<int>(std::floor((u_hi - rect_u_lo) * inv_out_du));
              if (ix_hi < 0 || ix_lo >= out_nx) {
                continue;
              }
              ix_lo = std::max(0, ix_lo);
              ix_hi = std::min(out_nx - 1, ix_hi);

              for (int iy = iy_lo; iy <= iy_hi; ++iy) {
                const double pv_lo =
                    rect_v_lo + static_cast<double>(iy) * out_dv;
                const double pv_hi = pv_lo + out_dv;
                const double ov = std::max(0.0, std::min(v_hi, pv_hi) -
                                                    std::max(v_lo, pv_lo));
                if (ov <= 0.0) {
                  continue;
                }
                const std::size_t row = static_cast<std::size_t>(iy) *
                                        static_cast<std::size_t>(out_nx);
                for (int ix = ix_lo; ix <= ix_hi; ++ix) {
                  const double pu_lo =
                      rect_u_lo + static_cast<double>(ix) * out_du;
                  const double pu_hi = pu_lo + out_du;
                  const double ou = std::max(0.0, std::min(u_hi, pu_hi) -
                                                      std::max(u_lo, pu_lo));
                  if (ou <= 0.0) {
                    continue;
                  }
                  const double w = (ou * ov) / cell_area;
                  const auto index = row + static_cast<std::size_t>(ix);
                  store_buffer_scalar<double>(
                      out.data(), index,
                      load_buffer_scalar<double>(out.data(), index) + mass * w);
                }
              }
            }
          }
          double total_sum = 0.0;
          for (std::size_t idx = 0; idx < static_cast<std::size_t>(out_nx) *
                                              static_cast<std::size_t>(out_ny);
               ++idx) {
            total_sum += out[idx];
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
