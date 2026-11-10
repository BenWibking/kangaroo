#include "default_kernel_families.hpp"
#include "kernel_buffer_support.hpp"
#include "kernel_param_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace kangaroo {

void register_potential_kernels(KernelRegistry &registry) {
  registry.register_typed_kernel<PotentialSamplesParams>(
      KernelDesc{.name = "potential_samples_accumulate",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &level, int32_t block,
         std::span<const ChunkBuffer> inputs, const NeighborViews &,
         std::span<ChunkBuffer> outputs, const KernelParamsIR &kernel_params) {
        const auto &p = require_kernel_params<PotentialSamplesParams>(
            kernel_params, "potential_samples_accumulate");
        if (p.points.empty() || p.points.size() % 3 != 0 ||
            !std::isfinite(p.half_width) || p.half_width <= 0 ||
            std::any_of(p.points.begin(), p.points.end(),
                        [](double v) { return !std::isfinite(v); }))
          throw std::invalid_argument(
              "invalid potential sample coordinates/support");
        const auto &box = level.boxes.at(block);
        const int lo[3] = {box.lo.x, box.lo.y, box.lo.z};
        const int hi[3] = {box.hi.x, box.hi.y, box.hi.z};
        const auto &g = level.geom;
        for (int a = 0; a < 3; ++a)
          if (!std::isfinite(g.dx[a]) || g.dx[a] <= 0)
            throw std::invalid_argument(
                "potential sampling requires positive 3D spacing");
        const auto field = make_real_grid_accessor(inputs[0]);
        auto out = outputs[0].mutable_view<double, 2>();
        const std::size_t ns = p.points.size() / 3;
        const double h = p.half_width;
        const int ny = hi[1] - lo[1] + 1, nz = hi[2] - lo[2] + 1;
        const auto index = [=](int i, int j, int k) {
          return (static_cast<std::size_t>(i - lo[0]) * ny + j - lo[1]) * nz +
                 k - lo[2];
        };
        // Valid boxes only: never read ghosts or cells replaced by finer data.
        std::vector<bool> covered(
            static_cast<std::size_t>(hi[0] - lo[0] + 1) * ny * nz, false);
        if (p.covered_boxes)
          for (const auto &b : *p.covered_boxes)
            for (int i = std::max(lo[0], b.lo[0]);
                 i <= std::min(hi[0], b.hi[0]); ++i)
              for (int j = std::max(lo[1], b.lo[1]);
                   j <= std::min(hi[1], b.hi[1]); ++j)
                for (int k = std::max(lo[2], b.lo[2]);
                     k <= std::min(hi[2], b.hi[2]); ++k)
                  covered[index(i, j, k)] = true;
        for (std::size_t n = 0; n < ns; ++n) {
          for (int c = 0; c < 114; ++c)
            out(n, c) = 0;
          int begin[3], end[3];
          for (int a = 0; a < 3; ++a) {
            const double q = p.points[3 * n + a];
            // Clamp in floating point before converting possibly distant
            // coordinates.
            const double l =
                std::floor((q - h - g.x0[a]) / g.dx[a]) + g.index_origin[a];
            const double u =
                std::ceil((q + h - g.x0[a]) / g.dx[a]) + g.index_origin[a] - 1;
            begin[a] = static_cast<int>(
                std::clamp(l, double(lo[a]), double(hi[a]) + 1));
            end[a] = static_cast<int>(
                std::clamp(u, double(lo[a]) - 1, double(hi[a])));
          }
          for (int i = begin[0]; i <= end[0]; ++i)
            for (int j = begin[1]; j <= end[1]; ++j)
              for (int k = begin[2]; k <= end[2]; ++k) {
                if (covered[index(i, j, k)])
                  continue;
                const int ijk[3] = {i, j, k};
                double v[3], weight = 1;
                for (int a = 0; a < 3; ++a) {
                  const double edge =
                      g.x0[a] + (ijk[a] - g.index_origin[a]) * g.dx[a];
                  const double q = p.points[3 * n + a];
                  const double overlap =
                      std::min(edge + g.dx[a], q + h) - std::max(edge, q - h);
                  weight *= std::max(0.0, overlap / (2 * h));
                  v[a] = (edge + 0.5 * g.dx[a] - q) / h;
                }
                if (weight == 0)
                  continue;
                out(n, 110) = static_cast<double>(out(n, 110)) + weight;
                const double value = field(i - lo[0], j - lo[1], k - lo[2]);
                if (!std::isfinite(value)) {
                  out(n, 111) = static_cast<double>(out(n, 111)) + weight;
                  continue;
                }
                const double x = v[0], y = v[1], z = v[2];
                const std::array<double, 10> basis{
                    1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z};
                for (int a = 0; a < 10; ++a) {
                  out(n, 100 + a) = static_cast<double>(out(n, 100 + a)) +
                                    weight * basis[a] * value;
                  for (int b = 0; b < 10; ++b)
                    out(n, 10 * a + b) =
                        static_cast<double>(out(n, 10 * a + b)) +
                        weight * basis[a] * basis[b];
                }
                out(n, 112) = static_cast<double>(out(n, 112)) + 1;
                out(n, 113) =
                    static_cast<double>(out(n, 113)) +
                    weight * std::max({g.dx[0], g.dx[1], g.dx[2]}) / h;
              }
        }
        return hpx::make_ready_future();
      });
}
} // namespace kangaroo
