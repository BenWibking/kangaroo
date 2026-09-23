#include "default_kernel_families.hpp"

#include "kernel_buffer_support.hpp"
#include "kernel_param_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace kangaroo {

void register_toomre_kernels(KernelRegistry &registry) {
  using Params = ToomreProfileParams;

  /**
   * @brief Accumulates AMR-aware two-dimensional cylindrical moments.
   *
   * Bins uncovered cells by cylindrical radius and absolute height
   * relative to ``center`` using ``radial_edges`` and ``z_edges``, and
   * accumulates two moments per bin: the volume integral of the input
   * field and the sampled cell volume.  Unlike the Toomre profile kernel
   * this needs a single cell-centered input, no neighbor halos, and no
   * potential gradient, so a whole family of enclosed profiles can be
   * derived from one pass by cumulative summation.
   * @par Chunk inputs One matching real-valued block grid.
   * @par Typed parameters Radial bin edges, |z| bin edges, center, and
   * covered coarse cells.
   * @par Chunk outputs An f64 ``(bins_r * bins_z, 2)`` array whose row
   * ``radial_bin * bins_z + z_bin`` holds ``(integral, volume)``.
   */
  registry.register_typed_kernel<Params>(
      KernelDesc{.name = "cylindrical_moments_accumulate",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &level, int32_t block,
         std::span<const ChunkBuffer> inputs, const NeighborViews &,
         std::span<ChunkBuffer> outputs,
         const KernelParamsIR &kernel_params) {
        constexpr std::size_t num_moments = 2;
        const auto &params = require_kernel_params<Params>(
            kernel_params, "cylindrical_moments_accumulate");
        if (outputs.empty() || inputs.empty() ||
            params.radial_edges.size() < 2 || params.z_edges.size() < 2) {
          return hpx::make_ready_future();
        }

        const int radial_bins =
            static_cast<int>(params.radial_edges.size() - 1);
        const int z_bins = static_cast<int>(params.z_edges.size() - 1);
        const int total_bins = radial_bins * z_bins;
        auto out = outputs[0].mutable_view<double, 2>();
        for (int bin = 0; bin < total_bins; ++bin)
          for (std::size_t component = 0; component < num_moments;
               ++component)
            out(bin, component) = 0.0;

        const double rmin = params.radial_edges.front();
        const double rmax = params.radial_edges.back();
        const double zmin = params.z_edges.front();
        const double zmax = params.z_edges.back();
        if (block < 0 || static_cast<std::size_t>(block) >= level.boxes.size() ||
            !std::isfinite(rmin) || !std::isfinite(rmax) || rmin < 0.0 ||
            rmax <= rmin || !std::isfinite(zmin) || !std::isfinite(zmax) ||
            zmin < 0.0 || zmax <= zmin ||
            std::adjacent_find(
                params.radial_edges.begin(), params.radial_edges.end(),
                [](double left, double right) {
                  return !std::isfinite(left) || !std::isfinite(right) ||
                         right <= left;
                }) != params.radial_edges.end() ||
            std::adjacent_find(
                params.z_edges.begin(), params.z_edges.end(),
                [](double left, double right) {
                  return !std::isfinite(left) || !std::isfinite(right) ||
                         right <= left;
                }) != params.z_edges.end() ||
            std::any_of(params.center.begin(), params.center.end(),
                        [](double value) { return !std::isfinite(value); })) {
          return hpx::make_ready_future();
        }

        const auto &box = level.boxes.at(static_cast<std::size_t>(block));
        const int nx = box.hi.x - box.lo.x + 1;
        const int ny = box.hi.y - box.lo.y + 1;
        const int nz = box.hi.z - box.lo.z + 1;
        if (nx <= 0 || ny <= 0 || nz <= 0) {
          return hpx::make_ready_future();
        }

        const auto field = make_real_grid_accessor(inputs[0]);

        const std::size_t mask_nx = static_cast<std::size_t>(nx) + 1;
        const std::size_t mask_ny = static_cast<std::size_t>(ny) + 1;
        const std::size_t mask_nz = static_cast<std::size_t>(nz) + 1;
        std::vector<int32_t> covered_mask;
        auto mask_index = [=](int i, int j, int k) {
          return (static_cast<std::size_t>(i) * mask_ny +
                  static_cast<std::size_t>(j)) *
                     mask_nz +
                 static_cast<std::size_t>(k);
        };
        if (params.covered_boxes && !params.covered_boxes->empty()) {
          covered_mask.assign(mask_nx * mask_ny * mask_nz, 0);
          for (const auto &covered_box : *params.covered_boxes) {
            const int lo[3] = {
                std::max(covered_box.lo[0], box.lo.x) - box.lo.x,
                std::max(covered_box.lo[1], box.lo.y) - box.lo.y,
                std::max(covered_box.lo[2], box.lo.z) - box.lo.z};
            const int hi[3] = {
                std::min(covered_box.hi[0], box.hi.x) - box.lo.x + 1,
                std::min(covered_box.hi[1], box.hi.y) - box.lo.y + 1,
                std::min(covered_box.hi[2], box.hi.z) - box.lo.z + 1};
            if (lo[0] >= hi[0] || lo[1] >= hi[1] || lo[2] >= hi[2])
              continue;
            for (int di = 0; di < 2; ++di)
              for (int dj = 0; dj < 2; ++dj)
                for (int dk = 0; dk < 2; ++dk) {
                  const int sign = ((di + dj + dk) % 2 == 0) ? 1 : -1;
                  covered_mask[mask_index(di ? hi[0] : lo[0],
                                          dj ? hi[1] : lo[1],
                                          dk ? hi[2] : lo[2])] += sign;
                }
          }
          for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
              for (int k = 0; k < nz; ++k) {
                const auto index = mask_index(i, j, k);
                if (i > 0)
                  covered_mask[index] += covered_mask[mask_index(i - 1, j, k)];
                if (j > 0)
                  covered_mask[index] += covered_mask[mask_index(i, j - 1, k)];
                if (k > 0)
                  covered_mask[index] += covered_mask[mask_index(i, j, k - 1)];
                if (i > 0 && j > 0)
                  covered_mask[index] -=
                      covered_mask[mask_index(i - 1, j - 1, k)];
                if (i > 0 && k > 0)
                  covered_mask[index] -=
                      covered_mask[mask_index(i - 1, j, k - 1)];
                if (j > 0 && k > 0)
                  covered_mask[index] -=
                      covered_mask[mask_index(i, j - 1, k - 1)];
                if (i > 0 && j > 0 && k > 0)
                  covered_mask[index] +=
                      covered_mask[mask_index(i - 1, j - 1, k - 1)];
              }
        }

        auto cell_edge = [&](int axis, int global_index) {
          return level.geom.x0[axis] +
                 static_cast<double>(global_index -
                                     level.geom.index_origin[axis]) *
                     level.geom.dx[axis];
        };
        auto bin_index = [](const std::vector<double> &edges, double value,
                             int bins) {
          if (value < edges.front() || value > edges.back())
            return -1;
          const auto upper_edge =
              std::upper_bound(edges.begin(), edges.end(), value);
          const auto upper_index =
              std::distance(edges.begin(), upper_edge);
          const int bin =
              value == edges.back()
                  ? bins - 1
                  : static_cast<int>(upper_index) - 1;
          return (bin < 0 || bin >= bins) ? -1 : bin;
        };

        for (int i = 0; i < nx; ++i) {
          const int gi = box.lo.x + i;
          const double x = 0.5 * (cell_edge(0, gi) + cell_edge(0, gi + 1));
          const double rx = x - params.center[0];
          for (int j = 0; j < ny; ++j) {
            const int gj = box.lo.y + j;
            const double y = 0.5 * (cell_edge(1, gj) + cell_edge(1, gj + 1));
            const double ry = y - params.center[1];
            const double radius = std::sqrt(rx * rx + ry * ry);
            const int radial_bin =
                bin_index(params.radial_edges, radius, radial_bins);
            if (radial_bin < 0)
              continue;

            for (int k = 0; k < nz; ++k) {
              if (!covered_mask.empty() &&
                  covered_mask[mask_index(i, j, k)] != 0)
                continue;
              const int gk = box.lo.z + k;
              const double z =
                  0.5 * (cell_edge(2, gk) + cell_edge(2, gk + 1));
              const double height = std::abs(z - params.center[2]);
              const int z_bin = bin_index(params.z_edges, height, z_bins);
              if (z_bin < 0)
                continue;

              const double value = field(i, j, k);
              if (!std::isfinite(value))
                continue;

              const double volume = std::abs(level.geom.dx[0] *
                                             level.geom.dx[1] *
                                             level.geom.dx[2]);
              const int bin = radial_bin * z_bins + z_bin;
              out(bin, 0) = static_cast<double>(out(bin, 0)) + value * volume;
              out(bin, 1) = static_cast<double>(out(bin, 1)) + volume;
            }
          }
        }
        return hpx::make_ready_future();
      });

  /**
   * @brief Accumulates AMR-aware annular moments used by gas Toomre-Q profiles.
   * @par Chunk inputs Density, x/y momentum, internal energy, three magnetic
   * field components, and a three-component potential-gradient grid.
   * @par Typed parameters Radial bin edges, vertical bounds, center, and
   * covered coarse cells.
   * @par Chunk outputs An f64 `(bins, 7)` array of annular moments.
   */
  registry.register_typed_kernel<Params>(
      KernelDesc{.name = "toomre_profile_accumulate",
                 .n_inputs = 8,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &level, int32_t block,
         std::span<const ChunkBuffer> inputs, const NeighborViews &,
         std::span<ChunkBuffer> outputs,
         const KernelParamsIR &kernel_params) {
        constexpr std::size_t num_components = 7;
        const auto &params = require_kernel_params<Params>(
            kernel_params, "toomre_profile_accumulate");
        if (outputs.empty() || inputs.size() < 8 ||
            params.radial_edges.size() < 2) {
          return hpx::make_ready_future();
        }

        const int bins = static_cast<int>(params.radial_edges.size() - 1);
        auto out = outputs[0].mutable_view<double, 2>();
        for (int bin = 0; bin < bins; ++bin)
          for (std::size_t component = 0; component < num_components;
               ++component)
            out(bin, component) = 0.0;

        const double rmin = params.radial_edges.front();
        const double rmax = params.radial_edges.back();
        const double zmin = params.z_bounds[0];
        const double zmax = params.z_bounds[1];
        if (block < 0 || static_cast<std::size_t>(block) >= level.boxes.size() ||
            !std::isfinite(rmin) || !std::isfinite(rmax) || rmin < 0.0 ||
            rmax <= rmin || !std::isfinite(zmin) || !std::isfinite(zmax) ||
            zmax <= zmin ||
            std::adjacent_find(
                params.radial_edges.begin(), params.radial_edges.end(),
                [](double left, double right) {
                  return !std::isfinite(left) || !std::isfinite(right) ||
                         right <= left;
                }) != params.radial_edges.end() ||
            std::any_of(params.center.begin(), params.center.end(),
                        [](double value) { return !std::isfinite(value); })) {
          return hpx::make_ready_future();
        }

        const auto &box = level.boxes.at(static_cast<std::size_t>(block));
        const int nx = box.hi.x - box.lo.x + 1;
        const int ny = box.hi.y - box.lo.y + 1;
        const int nz = box.hi.z - box.lo.z + 1;
        if (nx <= 0 || ny <= 0 || nz <= 0) {
          return hpx::make_ready_future();
        }

        std::array<RealGridAccessor, 7> fields;
        for (std::size_t index = 0; index < fields.size(); ++index)
          fields[index] = make_real_grid_accessor(inputs[index]);
        const auto gradient = inputs[7].view<double, 4>();

        const std::size_t mask_nx = static_cast<std::size_t>(nx) + 1;
        const std::size_t mask_ny = static_cast<std::size_t>(ny) + 1;
        const std::size_t mask_nz = static_cast<std::size_t>(nz) + 1;
        std::vector<int32_t> covered_mask;
        auto mask_index = [=](int i, int j, int k) {
          return (static_cast<std::size_t>(i) * mask_ny +
                  static_cast<std::size_t>(j)) *
                     mask_nz +
                 static_cast<std::size_t>(k);
        };
        if (params.covered_boxes && !params.covered_boxes->empty()) {
          covered_mask.assign(mask_nx * mask_ny * mask_nz, 0);
          for (const auto &covered_box : *params.covered_boxes) {
            const int lo[3] = {
                std::max(covered_box.lo[0], box.lo.x) - box.lo.x,
                std::max(covered_box.lo[1], box.lo.y) - box.lo.y,
                std::max(covered_box.lo[2], box.lo.z) - box.lo.z};
            const int hi[3] = {
                std::min(covered_box.hi[0], box.hi.x) - box.lo.x + 1,
                std::min(covered_box.hi[1], box.hi.y) - box.lo.y + 1,
                std::min(covered_box.hi[2], box.hi.z) - box.lo.z + 1};
            if (lo[0] >= hi[0] || lo[1] >= hi[1] || lo[2] >= hi[2])
              continue;
            for (int di = 0; di < 2; ++di)
              for (int dj = 0; dj < 2; ++dj)
                for (int dk = 0; dk < 2; ++dk) {
                  const int sign = ((di + dj + dk) % 2 == 0) ? 1 : -1;
                  covered_mask[mask_index(di ? hi[0] : lo[0],
                                          dj ? hi[1] : lo[1],
                                          dk ? hi[2] : lo[2])] += sign;
                }
          }
          for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
              for (int k = 0; k < nz; ++k) {
                const auto index = mask_index(i, j, k);
                if (i > 0)
                  covered_mask[index] += covered_mask[mask_index(i - 1, j, k)];
                if (j > 0)
                  covered_mask[index] += covered_mask[mask_index(i, j - 1, k)];
                if (k > 0)
                  covered_mask[index] += covered_mask[mask_index(i, j, k - 1)];
                if (i > 0 && j > 0)
                  covered_mask[index] -=
                      covered_mask[mask_index(i - 1, j - 1, k)];
                if (i > 0 && k > 0)
                  covered_mask[index] -=
                      covered_mask[mask_index(i - 1, j, k - 1)];
                if (j > 0 && k > 0)
                  covered_mask[index] -=
                      covered_mask[mask_index(i, j - 1, k - 1)];
                if (i > 0 && j > 0 && k > 0)
                  covered_mask[index] +=
                      covered_mask[mask_index(i - 1, j - 1, k - 1)];
              }
        }

        auto cell_edge = [&](int axis, int global_index) {
          return level.geom.x0[axis] +
                 static_cast<double>(global_index -
                                     level.geom.index_origin[axis]) *
                     level.geom.dx[axis];
        };
        const double dx = std::abs(level.geom.dx[0]);
        const double dy = std::abs(level.geom.dx[1]);

        for (int i = 0; i < nx; ++i) {
          const int gi = box.lo.x + i;
          const double x = 0.5 * (cell_edge(0, gi) + cell_edge(0, gi + 1));
          const double rx = x - params.center[0];
          for (int j = 0; j < ny; ++j) {
            const int gj = box.lo.y + j;
            const double y = 0.5 * (cell_edge(1, gj) + cell_edge(1, gj + 1));
            const double ry = y - params.center[1];
            const double radius = std::sqrt(rx * rx + ry * ry);
            if (radius < rmin || radius > rmax)
              continue;
            const auto upper_edge =
                std::upper_bound(params.radial_edges.begin(),
                                 params.radial_edges.end(), radius);
            const auto upper_index =
                std::distance(params.radial_edges.begin(), upper_edge);
            const int radial_bin =
                radius == rmax
                    ? bins - 1
                    : static_cast<int>(upper_index) - 1;
            if (radial_bin < 0 || radial_bin >= bins)
              continue;

            for (int k = 0; k < nz; ++k) {
              if (!covered_mask.empty() &&
                  covered_mask[mask_index(i, j, k)] != 0)
                continue;
              const int gk = box.lo.z + k;
              const double cell_z0 = cell_edge(2, gk);
              const double cell_z1 = cell_edge(2, gk + 1);
              const double overlap = std::max(
                  0.0, std::min(cell_z1, zmax) - std::max(cell_z0, zmin));
              if (overlap <= 0.0)
                continue;

              const double rho = fields[0](i, j, k);
              const double momx = fields[1](i, j, k);
              const double momy = fields[2](i, j, k);
              const double internal_energy = fields[3](i, j, k);
              const double bx = fields[4](i, j, k);
              const double by = fields[5](i, j, k);
              const double bz = fields[6](i, j, k);
              const double grad_x = gradient(i, j, k, 0);
              const double grad_y = gradient(i, j, k, 1);
              if (!std::isfinite(rho) || rho <= 0.0 ||
                  !std::isfinite(momx) || !std::isfinite(momy) ||
                  !std::isfinite(internal_energy) || !std::isfinite(bx) ||
                  !std::isfinite(by) || !std::isfinite(bz) ||
                  !std::isfinite(grad_x) || !std::isfinite(grad_y))
                continue;

              const double volume = dx * dy * overlap;
              const double mass = rho * volume;
              const bool at_center = radius == 0.0;
              const double radial_velocity =
                  at_center ? 0.0
                            : (rx * momx + ry * momy) / (radius * rho);
              const double radial_gravity =
                  at_center ? 0.0 : (rx * grad_x + ry * grad_y) / radius;
              if (!std::isfinite(radial_velocity) ||
                  !std::isfinite(radial_gravity))
                continue;
              auto add = [&](std::size_t component, double value) {
                out(radial_bin, component) =
                    static_cast<double>(out(radial_bin, component)) + value;
              };
              add(0, mass);
              add(1, internal_energy * volume);
              add(2, (bx * bx + by * by + bz * bz) * volume);
              add(3, mass * radial_velocity);
              add(4, mass * radial_velocity * radial_velocity);
              add(5, mass * radial_gravity);
              add(6, volume);
            }
          }
        }
        return hpx::make_ready_future();
      });
}

} // namespace kangaroo
