#include "default_kernel_families.hpp"

#include "flux_geometry_support.hpp"
#include "kernel_buffer_support.hpp"
#include "kernel_param_support.hpp"

namespace kangaroo {

void register_flux_kernels(KernelRegistry &registry) {
  {
    using Params = FluxSurfaceParams;

    /**
     * @brief Accumulates fluxes through concentric spherical surfaces.
     * @par Chunk inputs Real block grids ordered as density, momentum x/y/z,
     * total energy, passive scalar, and magnetic field x/y/z; an optional tenth
     * input supplies temperature when temperature binning is enabled.
     * @par Typed parameters `radii` (or `radius`), `radius_indices`,
     * `num_radii`, `temperature_bins`, `gamma`, and `covered_boxes` select the
     * surfaces, output slots, thermodynamic bins, equation of state, and AMR
     * mask.
     * @par Chunk outputs `outputs[0]` is an f64 tensor of inward/outward mass,
     * momentum, energy, and passive-scalar flux for each radius and temperature
     * bin.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "flux_surface_integral_accumulate",
                   .n_inputs = 9,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block,
           std::span<const ChunkBuffer> inputs, const NeighborViews &,
           std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params = require_kernel_params<Params>(
              kernel_params, "flux_surface_integral_accumulate");

          const bool use_temperature_bins = params.temperature_bins.size() >= 2;
          const std::size_t num_temperature_bins =
              use_temperature_bins ? params.temperature_bins.size() - 1 : 1;
          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          auto output_storage_bytes = outputs[0].mutable_byte_view();
          std::fill(output_storage_bytes.begin(), output_storage_bytes.end(),
                    std::uint8_t{0});
          if (inputs.size() < 9 ||
              (use_temperature_bins && inputs.size() < 10) ||
              params.radii.empty() ||
              params.radius_indices.size() != params.radii.size() ||
              !std::isfinite(params.gamma) || params.gamma <= 1.0) {
            return hpx::make_ready_future();
          }
          if (use_temperature_bins) {
            for (double edge : params.temperature_bins) {
              if (!std::isfinite(edge)) {
                return hpx::make_ready_future();
              }
            }
            for (std::size_t i = 1; i < params.temperature_bins.size(); ++i) {
              if (params.temperature_bins[i] <=
                  params.temperature_bins[i - 1]) {
                return hpx::make_ready_future();
              }
            }
          }
          std::vector<double> radii2;
          radii2.reserve(params.radii.size());
          for (std::size_t radius_idx = 0; radius_idx < params.radii.size();
               ++radius_idx) {
            const double radius = params.radii[radius_idx];
            const int32_t output_idx = params.radius_indices[radius_idx];
            if (!std::isfinite(radius) || radius <= 0.0) {
              return hpx::make_ready_future();
            }
            if (output_idx < 0 ||
                static_cast<std::size_t>(output_idx) >= params.num_radii) {
              return hpx::make_ready_future();
            }
            radii2.push_back(radius * radius);
          }
          if (block < 0 ||
              static_cast<std::size_t>(block) >= level.boxes.size()) {
            return hpx::make_ready_future();
          }

          const auto &box = level.boxes.at(static_cast<std::size_t>(block));
          const int nx = box.hi.x - box.lo.x + 1;
          const int ny = box.hi.y - box.lo.y + 1;
          const int nz = box.hi.z - box.lo.z + 1;
          if (nx <= 0 || ny <= 0 || nz <= 0) {
            return hpx::make_ready_future();
          }

          auto logical_index = [&](int i, int j, int k) -> std::size_t {
            return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) +
                    static_cast<std::size_t>(j)) *
                       static_cast<std::size_t>(nz) +
                   static_cast<std::size_t>(k);
          };
          std::vector<std::uint8_t> covered_mask;
          if (params.covered_boxes && !params.covered_boxes->empty()) {
            const std::size_t block_cells = static_cast<std::size_t>(nx) *
                                            static_cast<std::size_t>(ny) *
                                            static_cast<std::size_t>(nz);
            for (const auto &b : *params.covered_boxes) {
              const int gx0 = std::max(box.lo.x, b.lo[0]);
              const int gy0 = std::max(box.lo.y, b.lo[1]);
              const int gz0 = std::max(box.lo.z, b.lo[2]);
              const int gx1 = std::min(box.hi.x, b.hi[0]);
              const int gy1 = std::min(box.hi.y, b.hi[1]);
              const int gz1 = std::min(box.hi.z, b.hi[2]);
              if (gx0 > gx1 || gy0 > gy1 || gz0 > gz1) {
                continue;
              }
              if (covered_mask.empty()) {
                covered_mask.assign(block_cells, 0);
              }
              for (int gx = gx0; gx <= gx1; ++gx) {
                const int i = gx - box.lo.x;
                for (int gy = gy0; gy <= gy1; ++gy) {
                  const int j = gy - box.lo.y;
                  for (int gz = gz0; gz <= gz1; ++gz) {
                    const int k = gz - box.lo.z;
                    covered_mask[logical_index(i, j, k)] = 1;
                  }
                }
              }
            }
          }
          std::array<RealGridAccessor, 10> input_views{};
          auto bind_inputs = [&](auto... typed_inputs) {
            std::size_t input_index = 0;
            ((input_views[input_index++] =
                  make_real_grid_accessor(typed_inputs.grid())),
             ...);
          };
          if (use_temperature_bins) {
            visit_real_buffers_exact<10>(inputs.first(10), bind_inputs);
          } else {
            visit_real_buffers_exact<9>(inputs.first(9), bind_inputs);
          }
          auto cell_edge = [&](int axis, int global_idx) -> double {
            return level.geom.x0[axis] +
                   (static_cast<double>(global_idx -
                                        level.geom.index_origin[axis]) *
                    level.geom.dx[axis]);
          };

          const double gamma_minus_one = params.gamma - 1.0;
          auto box_lo_axis = [&](int axis) -> int32_t {
            if (axis == 0) {
              return box.lo.x;
            }
            return axis == 1 ? box.lo.y : box.lo.z;
          };
          auto box_hi_axis = [&](int axis) -> int32_t {
            if (axis == 0) {
              return box.hi.x;
            }
            return axis == 1 ? box.hi.y : box.hi.z;
          };
          auto axis_band = [&](int axis, double lo, double hi) {
            return cells_intersecting_axis_band(
                lo, hi, level.geom.x0[axis], level.geom.dx[axis],
                level.geom.index_origin[axis], box_lo_axis(axis),
                box_hi_axis(axis));
          };

          auto accumulate_cell = [&](std::size_t radius_idx, int i, int j,
                                     int k, double x0, double x1, double y0,
                                     double y1, double z0, double z1) {
            if (!sphere_may_intersect_cell(radii2[radius_idx], x0, x1, y0, y1,
                                           z0, z1)) {
              return;
            }

            const auto idx = logical_index(i, j, k);
            if (!covered_mask.empty() && covered_mask[idx] != 0) {
              return;
            }

            const double x = 0.5 * (x0 + x1);
            const double y = 0.5 * (y0 + y1);
            const double z = 0.5 * (z0 + z1);
            const double r = std::sqrt(x * x + y * y + z * z);

            const double rho = static_cast<double>(input_views[0](i, j, k));
            if (r <= 0.0 || rho <= 0.0 || !std::isfinite(rho)) {
              return;
            }

            const double momx = static_cast<double>(input_views[1](i, j, k));
            const double momy = static_cast<double>(input_views[2](i, j, k));
            const double momz = static_cast<double>(input_views[3](i, j, k));
            const double energy_density =
                static_cast<double>(input_views[4](i, j, k));
            const double scalar_density =
                static_cast<double>(input_views[5](i, j, k));
            const double bx = static_cast<double>(input_views[6](i, j, k));
            const double by = static_cast<double>(input_views[7](i, j, k));
            const double bz = static_cast<double>(input_views[8](i, j, k));
            double temperature = 0.0;
            if (use_temperature_bins) {
              temperature = static_cast<double>(input_views[9](i, j, k));
            }
            if (!std::isfinite(momx) || !std::isfinite(momy) ||
                !std::isfinite(momz) || !std::isfinite(energy_density) ||
                !std::isfinite(scalar_density) || !std::isfinite(bx) ||
                !std::isfinite(by) || !std::isfinite(bz) ||
                (use_temperature_bins && !std::isfinite(temperature))) {
              return;
            }
            std::size_t temperature_bin = 0;
            if (use_temperature_bins) {
              const auto upper =
                  std::upper_bound(params.temperature_bins.begin(),
                                   params.temperature_bins.end(), temperature);
              if (upper == params.temperature_bins.begin() ||
                  upper == params.temperature_bins.end()) {
                if (temperature != params.temperature_bins.back()) {
                  return;
                }
                temperature_bin = params.temperature_bins.size() - 2;
              } else {
                temperature_bin = static_cast<std::size_t>(
                    std::distance(params.temperature_bins.begin(), upper) - 1);
              }
            }
            if (temperature_bin >= num_temperature_bins) {
              return;
            }

            const double vx = momx / rho;
            const double vy = momy / rho;
            const double vz = momz / rho;
            const double vr = (x * momx + y * momy + z * momz) / (rho * r);
            const double rhat_x = x / r;
            const double rhat_y = y / r;
            const double rhat_z = z / r;

            const double kinetic =
                0.5 * (momx * momx + momy * momy + momz * momz) / rho;
            const double emag = 0.5 * (bx * bx + by * by + bz * bz);
            const double ehydro = energy_density - emag;
            const double pgas = gamma_minus_one * (ehydro - kinetic);
            if (!std::isfinite(vr) || !std::isfinite(pgas)) {
              return;
            }

            const double area = spherical_section_area_in_intersecting_cell(
                params.radii[radius_idx], x0, x1, y0, y1, z0, z1);
            if (area <= 0.0) {
              return;
            }

            const double bdotv = vx * bx + vy * by + vz * bz;
            const double br = rhat_x * bx + rhat_y * by + rhat_z * bz;
            const double mass_flux = rho * vr * area;
            const double hydro_energy_flux = (ehydro + pgas) * vr * area;
            const double mhd_energy_flux =
                ((energy_density + pgas + emag) * vr - bdotv * br) * area;
            const double scalar_flux = scalar_density * vr * area;
            const std::array<double, 4> fluxes{mass_flux, hydro_energy_flux,
                                               mhd_energy_flux, scalar_flux};
            const std::size_t radius_base =
                static_cast<std::size_t>(params.radius_indices[radius_idx]) *
                2 * num_temperature_bins * 4;
            for (std::size_t component = 0; component < fluxes.size();
                 ++component) {
              const std::size_t sign_bin = fluxes[component] < 0.0 ? 0 : 1;
              const auto output_index = radius_base +
                                        sign_bin * num_temperature_bins * 4 +
                                        temperature_bin * 4 + component;
              store_buffer_scalar(
                  output_storage_bytes.data(), output_index,
                  load_buffer_scalar<double>(output_storage_bytes.data(),
                                             output_index) +
                      fluxes[component]);
            }
          };

          for (std::size_t radius_idx = 0; radius_idx < params.radii.size();
               ++radius_idx) {
            const double radius = params.radii[radius_idx];
            const double radius2 = radii2[radius_idx];
            const auto i_range = axis_band(0, -radius, radius);
            if (i_range.empty()) {
              continue;
            }

            for (int i = i_range.first; i <= i_range.last; ++i) {
              const int gi = box.lo.x + i;
              const double x0 = cell_edge(0, gi);
              const double x1 = x0 + level.geom.dx[0];
              const double min_x = min_dist_sq_to_interval(x0, x1);
              if (min_x > radius2) {
                continue;
              }

              const double y_extent = std::sqrt(std::max(0.0, radius2 - min_x));
              const auto j_range = axis_band(1, -y_extent, y_extent);
              if (j_range.empty()) {
                continue;
              }

              const double max_x = max_dist_sq_to_interval(x0, x1);
              for (int j = j_range.first; j <= j_range.last; ++j) {
                const int gj = box.lo.y + j;
                const double y0 = cell_edge(1, gj);
                const double y1 = y0 + level.geom.dx[1];
                const double min_xy = min_x + min_dist_sq_to_interval(y0, y1);
                if (min_xy > radius2) {
                  continue;
                }

                const double max_xy = max_x + max_dist_sq_to_interval(y0, y1);
                const double near_z =
                    std::sqrt(std::max(0.0, radius2 - min_xy));
                const double far_z2 = radius2 - max_xy;

                std::array<CellIndexRange, 2> k_ranges{};
                int num_k_ranges = 0;
                if (far_z2 <= 0.0) {
                  k_ranges[0] = axis_band(2, -near_z, near_z);
                  num_k_ranges = 1;
                } else {
                  const double far_z = std::sqrt(far_z2);
                  k_ranges[0] = axis_band(2, -near_z, -far_z);
                  k_ranges[1] = axis_band(2, far_z, near_z);
                  num_k_ranges = 2;
                }

                int last_k = -1;
                for (int range_idx = 0; range_idx < num_k_ranges; ++range_idx) {
                  auto k_range = k_ranges[static_cast<std::size_t>(range_idx)];
                  if (k_range.empty()) {
                    continue;
                  }
                  k_range.first = std::max(k_range.first, last_k + 1);
                  if (k_range.empty()) {
                    continue;
                  }
                  for (int k = k_range.first; k <= k_range.last; ++k) {
                    const int gk = box.lo.z + k;
                    const double z0 = cell_edge(2, gk);
                    const double z1 = z0 + level.geom.dx[2];
                    accumulate_cell(radius_idx, i, j, k, x0, x1, y0, y1, z0,
                                    z1);
                  }
                  last_k = k_range.last;
                }
              }
            }
          }
          return hpx::make_ready_future();
        });
  }
  {
    using Params = CylindricalFluxParams;

    /**
     * @brief Accumulates fluxes through concentric cylindrical surfaces.
     * @par Chunk inputs Real block grids ordered as density, momentum x/y/z,
     * total energy, passive scalar, and magnetic field x/y/z; an optional tenth
     * input supplies temperature when temperature binning is enabled.
     * @par Typed parameters `radius`, `heights`, `height_indices`,
     * `num_heights`, `temperature_bins`, `gamma`, and `covered_boxes` select
     * the cylinder sections, output slots, thermodynamic bins, equation of
     * state, and AMR mask.
     * @par Chunk outputs `outputs[0]` is an f64 tensor of inward/outward mass,
     * momentum, energy, and passive-scalar flux for each cylindrical section
     * and temperature bin.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "cylindrical_flux_surface_integral_accumulate",
                   .n_inputs = 9,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block,
           std::span<const ChunkBuffer> inputs, const NeighborViews &,
           std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params = require_kernel_params<Params>(
              kernel_params, "cylindrical_flux_surface_integral_accumulate");

          const bool use_temperature_bins = params.temperature_bins.size() >= 2;
          const std::size_t num_temperature_bins =
              use_temperature_bins ? params.temperature_bins.size() - 1 : 1;
          constexpr std::size_t num_geometric_sections = 2;
          constexpr std::size_t endcaps_section = 0;
          constexpr std::size_t walls_section = 1;
          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          auto output_storage_bytes = outputs[0].mutable_byte_view();
          std::fill(output_storage_bytes.begin(), output_storage_bytes.end(),
                    std::uint8_t{0});
          if (inputs.size() < 9 ||
              (use_temperature_bins && inputs.size() < 10) ||
              params.heights.empty() ||
              params.height_indices.size() != params.heights.size() ||
              !std::isfinite(params.radius) || params.radius <= 0.0 ||
              !std::isfinite(params.gamma) || params.gamma <= 1.0) {
            return hpx::make_ready_future();
          }
          if (use_temperature_bins) {
            for (double edge : params.temperature_bins) {
              if (!std::isfinite(edge)) {
                return hpx::make_ready_future();
              }
            }
            for (std::size_t i = 1; i < params.temperature_bins.size(); ++i) {
              if (params.temperature_bins[i] <=
                  params.temperature_bins[i - 1]) {
                return hpx::make_ready_future();
              }
            }
          }
          for (std::size_t height_idx = 0; height_idx < params.heights.size();
               ++height_idx) {
            const double height = params.heights[height_idx];
            const int32_t output_idx = params.height_indices[height_idx];
            if (!std::isfinite(height) || height <= 0.0) {
              return hpx::make_ready_future();
            }
            if (output_idx < 0 ||
                static_cast<std::size_t>(output_idx) >= params.num_heights) {
              return hpx::make_ready_future();
            }
          }
          if (block < 0 ||
              static_cast<std::size_t>(block) >= level.boxes.size()) {
            return hpx::make_ready_future();
          }

          const auto &box = level.boxes.at(static_cast<std::size_t>(block));
          const int nx = box.hi.x - box.lo.x + 1;
          const int ny = box.hi.y - box.lo.y + 1;
          const int nz = box.hi.z - box.lo.z + 1;
          if (nx <= 0 || ny <= 0 || nz <= 0) {
            return hpx::make_ready_future();
          }

          auto logical_index = [&](int i, int j, int k) -> std::size_t {
            return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) +
                    static_cast<std::size_t>(j)) *
                       static_cast<std::size_t>(nz) +
                   static_cast<std::size_t>(k);
          };
          std::vector<std::uint8_t> covered_mask;
          if (params.covered_boxes && !params.covered_boxes->empty()) {
            const std::size_t block_cells = static_cast<std::size_t>(nx) *
                                            static_cast<std::size_t>(ny) *
                                            static_cast<std::size_t>(nz);
            for (const auto &b : *params.covered_boxes) {
              const int gx0 = std::max(box.lo.x, b.lo[0]);
              const int gy0 = std::max(box.lo.y, b.lo[1]);
              const int gz0 = std::max(box.lo.z, b.lo[2]);
              const int gx1 = std::min(box.hi.x, b.hi[0]);
              const int gy1 = std::min(box.hi.y, b.hi[1]);
              const int gz1 = std::min(box.hi.z, b.hi[2]);
              if (gx0 > gx1 || gy0 > gy1 || gz0 > gz1) {
                continue;
              }
              if (covered_mask.empty()) {
                covered_mask.assign(block_cells, 0);
              }
              for (int gx = gx0; gx <= gx1; ++gx) {
                const int i = gx - box.lo.x;
                for (int gy = gy0; gy <= gy1; ++gy) {
                  const int j = gy - box.lo.y;
                  for (int gz = gz0; gz <= gz1; ++gz) {
                    const int k = gz - box.lo.z;
                    covered_mask[logical_index(i, j, k)] = 1;
                  }
                }
              }
            }
          }
          std::array<RealGridAccessor, 10> input_views{};
          auto bind_inputs = [&](auto... typed_inputs) {
            std::size_t input_index = 0;
            ((input_views[input_index++] =
                  make_real_grid_accessor(typed_inputs.grid())),
             ...);
          };
          if (use_temperature_bins) {
            visit_real_buffers_exact<10>(inputs.first(10), bind_inputs);
          } else {
            visit_real_buffers_exact<9>(inputs.first(9), bind_inputs);
          }
          auto cell_edge = [&](int axis, int global_idx) -> double {
            return level.geom.x0[axis] +
                   (static_cast<double>(global_idx -
                                        level.geom.index_origin[axis]) *
                    level.geom.dx[axis]);
          };

          const double radius2 = params.radius * params.radius;
          const double gamma_minus_one = params.gamma - 1.0;
          auto box_lo_axis = [&](int axis) -> int32_t {
            if (axis == 0) {
              return box.lo.x;
            }
            return axis == 1 ? box.lo.y : box.lo.z;
          };
          auto box_hi_axis = [&](int axis) -> int32_t {
            if (axis == 0) {
              return box.hi.x;
            }
            return axis == 1 ? box.hi.y : box.hi.z;
          };
          auto axis_band = [&](int axis, double lo, double hi) {
            return cells_intersecting_axis_band(
                lo, hi, level.geom.x0[axis], level.geom.dx[axis],
                level.geom.index_origin[axis], box_lo_axis(axis),
                box_hi_axis(axis));
          };

          auto accumulate_flux = [&](std::size_t height_idx, int i, int j,
                                     int k, double nx, double ny, double nz,
                                     double area,
                                     std::size_t geometric_section) {
            if (area <= 0.0 || geometric_section >= num_geometric_sections) {
              return;
            }
            const auto idx = logical_index(i, j, k);
            if (!covered_mask.empty() && covered_mask[idx] != 0) {
              return;
            }

            const double rho = static_cast<double>(input_views[0](i, j, k));
            if (rho <= 0.0 || !std::isfinite(rho)) {
              return;
            }

            const double momx = static_cast<double>(input_views[1](i, j, k));
            const double momy = static_cast<double>(input_views[2](i, j, k));
            const double momz = static_cast<double>(input_views[3](i, j, k));
            const double energy_density =
                static_cast<double>(input_views[4](i, j, k));
            const double scalar_density =
                static_cast<double>(input_views[5](i, j, k));
            const double bx = static_cast<double>(input_views[6](i, j, k));
            const double by = static_cast<double>(input_views[7](i, j, k));
            const double bz = static_cast<double>(input_views[8](i, j, k));
            double temperature = 0.0;
            if (use_temperature_bins) {
              temperature = static_cast<double>(input_views[9](i, j, k));
            }
            if (!std::isfinite(momx) || !std::isfinite(momy) ||
                !std::isfinite(momz) || !std::isfinite(energy_density) ||
                !std::isfinite(scalar_density) || !std::isfinite(bx) ||
                !std::isfinite(by) || !std::isfinite(bz) ||
                (use_temperature_bins && !std::isfinite(temperature))) {
              return;
            }
            std::size_t temperature_bin = 0;
            if (use_temperature_bins) {
              const auto upper =
                  std::upper_bound(params.temperature_bins.begin(),
                                   params.temperature_bins.end(), temperature);
              if (upper == params.temperature_bins.begin() ||
                  upper == params.temperature_bins.end()) {
                if (temperature != params.temperature_bins.back()) {
                  return;
                }
                temperature_bin = params.temperature_bins.size() - 2;
              } else {
                temperature_bin = static_cast<std::size_t>(
                    std::distance(params.temperature_bins.begin(), upper) - 1);
              }
            }
            if (temperature_bin >= num_temperature_bins) {
              return;
            }

            const double vx = momx / rho;
            const double vy = momy / rho;
            const double vz = momz / rho;
            const double vnormal = vx * nx + vy * ny + vz * nz;

            const double kinetic =
                0.5 * (momx * momx + momy * momy + momz * momz) / rho;
            const double emag = 0.5 * (bx * bx + by * by + bz * bz);
            const double ehydro = energy_density - emag;
            const double pgas = gamma_minus_one * (ehydro - kinetic);
            if (!std::isfinite(vnormal) || !std::isfinite(pgas)) {
              return;
            }

            const double bdotv = vx * bx + vy * by + vz * bz;
            const double bnormal = nx * bx + ny * by + nz * bz;
            const double mass_flux = rho * vnormal * area;
            const double hydro_energy_flux = (ehydro + pgas) * vnormal * area;
            const double mhd_energy_flux =
                ((energy_density + pgas + emag) * vnormal - bdotv * bnormal) *
                area;
            const double scalar_flux = scalar_density * vnormal * area;
            const std::array<double, 4> fluxes{mass_flux, hydro_energy_flux,
                                               mhd_energy_flux, scalar_flux};
            const std::size_t height_base =
                static_cast<std::size_t>(params.height_indices[height_idx]) *
                2 * num_temperature_bins * num_geometric_sections * 4;
            for (std::size_t component = 0; component < fluxes.size();
                 ++component) {
              const std::size_t sign_bin = fluxes[component] < 0.0 ? 0 : 1;
              const auto output_index =
                  height_base +
                  sign_bin * num_temperature_bins * num_geometric_sections * 4 +
                  temperature_bin * num_geometric_sections * 4 +
                  geometric_section * 4 + component;
              store_buffer_scalar(
                  output_storage_bytes.data(), output_index,
                  load_buffer_scalar<double>(output_storage_bytes.data(),
                                             output_index) +
                      fluxes[component]);
            }
          };

          const auto i_range = axis_band(0, -params.radius, params.radius);
          if (!i_range.empty()) {
            for (int i = i_range.first; i <= i_range.last; ++i) {
              const int gi = box.lo.x + i;
              const double x0 = cell_edge(0, gi);
              const double x1 = x0 + level.geom.dx[0];
              const double min_x = min_dist_sq_to_interval(x0, x1);
              if (min_x > radius2) {
                continue;
              }

              const double y_extent = std::sqrt(std::max(0.0, radius2 - min_x));
              const auto j_range = axis_band(1, -y_extent, y_extent);
              if (j_range.empty()) {
                continue;
              }

              for (int j = j_range.first; j <= j_range.last; ++j) {
                const int gj = box.lo.y + j;
                const double y0 = cell_edge(1, gj);
                const double y1 = y0 + level.geom.dx[1];
                const double min_xy = min_x + min_dist_sq_to_interval(y0, y1);
                const double max_xy = max_dist_sq_to_interval(x0, x1) +
                                      max_dist_sq_to_interval(y0, y1);
                if (min_xy > radius2) {
                  continue;
                }

                for (std::size_t height_idx = 0;
                     height_idx < params.heights.size(); ++height_idx) {
                  const double height = params.heights[height_idx];
                  const auto k_range = axis_band(2, -height, height);
                  if (k_range.empty()) {
                    continue;
                  }
                  for (int k = k_range.first; k <= k_range.last; ++k) {
                    const int gk = box.lo.z + k;
                    const double z0 = cell_edge(2, gk);
                    const double z1 = z0 + level.geom.dx[2];
                    if (max_xy >= radius2 &&
                        cylinder_may_intersect_cell(radius2, height, x0, x1, y0,
                                                    y1, z0, z1)) {
                      const double x = 0.5 * (x0 + x1);
                      const double y = 0.5 * (y0 + y1);
                      const double rxy = std::sqrt(x * x + y * y);
                      if (rxy > 0.0) {
                        const double area =
                            cylindrical_section_area_in_intersecting_cell(
                                params.radius, height, x0, x1, y0, y1, z0, z1);
                        accumulate_flux(height_idx, i, j, k, x / rxy, y / rxy,
                                        0.0, area, walls_section);
                      }
                    }
                    if (z0 <= height && z1 >= height) {
                      const double area = plane_box_section_area(
                          x0, x1, y0, y1, z0, z1, 0.0, 0.0, 1.0, height);
                      accumulate_flux(height_idx, i, j, k, 0.0, 0.0, 1.0, area,
                                      endcaps_section);
                    }
                    if (z0 <= -height && z1 >= -height) {
                      const double area = plane_box_section_area(
                          x0, x1, y0, y1, z0, z1, 0.0, 0.0, 1.0, -height);
                      accumulate_flux(height_idx, i, j, k, 0.0, 0.0, -1.0, area,
                                      endcaps_section);
                    }
                  }
                }
              }
            }
          }
          return hpx::make_ready_future();
        });
  }
}

} // namespace kangaroo
