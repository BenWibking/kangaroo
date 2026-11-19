#include "default_kernel_families.hpp"

#include "kernel_buffer_support.hpp"
#include "kernel_param_support.hpp"
#include "kangaroo/box_spatial_index.hpp"

#include <optional>

namespace kangaroo {

void register_histogram_kernels(KernelRegistry &registry) {
  {
    using Params = Histogram1DParams;

    /**
     * @brief Accumulates uncovered grid-cell values into a one-dimensional
     * histogram.
     * @par Chunk inputs `inputs[0]` is a real-valued block grid; optional
     * `inputs[1]` is a matching real-valued weight grid.
     * @par Typed parameters `range`, `bins`, and `covered_boxes` define
     * the histogram interval, bin count, and excluded AMR cells.
     * @par Chunk outputs `outputs[0]` is an f64 array of `bins` accumulated
     * counts or weights.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "histogram1d_accumulate",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block,
           std::span<const ChunkBuffer> inputs, const NeighborViews &,
           std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params = require_kernel_params<Params>(
              kernel_params, "histogram1d_accumulate");

          if (outputs.empty() || inputs.empty() || params.bins <= 0) {
            return hpx::make_ready_future();
          }
          const double lo = params.range[0];
          const double hi = params.range[1];
          if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo) {
            return hpx::make_ready_future();
          }

          auto out = outputs[0].mutable_array<double>();
          for (int bin = 0; bin < params.bins; ++bin)
            out(bin) = 0.0;

          if (inputs[0].desc().rank != 3) {
            const auto count =
                static_cast<std::size_t>(inputs[0].desc().element_count());
            const auto values = make_real_buffer_accessor(inputs[0]);
            std::optional<RealBufferAccessor> weights;
            if (inputs.size() == 2) {
              if (inputs[1].desc().element_count() != count) {
                throw BufferContractError(
                    BufferContractReason::kDescriptorStorageMismatch,
                    "bounded histogram values and weights must have matching extents");
              }
              weights = make_real_buffer_accessor(inputs[1]);
            }
            const double inv_dx = static_cast<double>(params.bins) / (hi - lo);
            for (std::size_t index = 0; index < count; ++index) {
              const double value = values(index);
              if (!std::isfinite(value) || value < lo || value > hi)
                continue;
              const int bin =
                  value == hi
                      ? params.bins - 1
                      : static_cast<int>(std::floor((value - lo) * inv_dx));
              if (bin < 0 || bin >= params.bins)
                continue;
              const double weight = weights ? (*weights)(index) : 1.0;
              if (std::isfinite(weight))
                out(bin) = static_cast<double>(out(bin)) + weight;
            }
            return hpx::make_ready_future();
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

          auto covered = [&](int ix, int iy, int iz) -> bool {
            if (!params.covered_boxes) {
              return false;
            }
            for (const auto &b : *params.covered_boxes) {
              if (covered_box_contains(b, ix, iy, iz)) {
                return true;
              }
            }
            return false;
          };
          const double inv_dx = static_cast<double>(params.bins) / (hi - lo);
          auto accumulate = [&](auto... typed_inputs) {
            static_assert(sizeof...(typed_inputs) == 1 ||
                          sizeof...(typed_inputs) == 2);
            const auto typed = std::tuple{typed_inputs...};
            const auto values = std::get<0>(typed).grid();
            for (int i = 0; i < nx; ++i) {
              const int gi = box.lo.x + i;
              for (int j = 0; j < ny; ++j) {
                const int gj = box.lo.y + j;
                for (int k = 0; k < nz; ++k) {
                  const int gk = box.lo.z + k;
                  if (covered(gi, gj, gk))
                    continue;
                  const double value = static_cast<double>(values(i, j, k));
                  if (!std::isfinite(value) || value < lo || value > hi)
                    continue;
                  const int bin =
                      value == hi
                          ? params.bins - 1
                          : static_cast<int>(std::floor((value - lo) * inv_dx));
                  if (bin < 0 || bin >= params.bins)
                    continue;
                  double weight = 1.0;
                  if constexpr (sizeof...(typed_inputs) == 2) {
                    weight =
                        static_cast<double>(std::get<1>(typed).grid()(i, j, k));
                    if (!std::isfinite(weight))
                      continue;
                  }
                  out(bin) = static_cast<double>(out(bin)) + weight;
                }
              }
            }
          };
          if (inputs.size() == 1) {
            visit_real_buffers_exact<1>(inputs, accumulate);
          } else {
            visit_real_buffers_exact<2>(inputs, accumulate);
          }
          return hpx::make_ready_future();
        });
  }
  {
    using Params = Histogram2DParams;

    /**
     * @brief Accumulates uncovered grid-cell samples into a two-dimensional
     * histogram.
     * @par Chunk inputs `inputs[0]` and `inputs[1]` are matching real-valued x
     * and y block grids; optional `inputs[2]` is a matching weight grid.
     * @par Typed parameters `x_range`, `y_range`, `bins`, `weight_mode`,
     * and `log_bins` define binning and weighting. The coverage index excludes
     * cells represented on finer AMR levels.
     * @par Chunk outputs `outputs[0]` is an f64 two-dimensional histogram.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "histogram2d_accumulate",
                   .n_inputs = 2,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block,
           std::span<const ChunkBuffer> inputs, const NeighborViews &,
           std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params = require_kernel_params<Params>(
              kernel_params, "histogram2d_accumulate");

          const int nx_bins = params.bins[0];
          const int ny_bins = params.bins[1];
          if (outputs.empty() || inputs.size() < 2 || nx_bins <= 0 ||
              ny_bins <= 0) {
            return hpx::make_ready_future();
          }
          const double xlo = params.x_range[0];
          const double xhi = params.x_range[1];
          const double ylo = params.y_range[0];
          const double yhi = params.y_range[1];
          if (!std::isfinite(xlo) || !std::isfinite(xhi) ||
              !std::isfinite(ylo) || !std::isfinite(yhi) || xhi <= xlo ||
              yhi <= ylo || (params.log_bins && (xlo <= 0.0 || ylo <= 0.0))) {
            return hpx::make_ready_future();
          }

          const auto make_bin_index = [log_bins = params.log_bins](double lo, double hi, int bins) {
            std::vector<double> interior_edges;
            if (log_bins) {
              const double start = std::log10(lo);
              const double width = (std::log10(hi) - start) / bins;
              interior_edges.reserve(bins - 1);
              for (int i = 1; i < bins; ++i) {
                interior_edges.push_back(std::pow(10.0, start + i * width));
              }
            }
            const double inv_width = static_cast<double>(bins) / (hi - lo);
            return [=, edges = std::move(interior_edges)](double value) {
              if (!std::isfinite(value) || value < lo || value > hi) return -1;
              if (value == hi) return bins - 1;
              if (log_bins) {
                return static_cast<int>(std::upper_bound(edges.begin(), edges.end(), value) - edges.begin());
              }
              return std::min(bins - 1, static_cast<int>(std::floor((value - lo) * inv_width)));
            };
          };
          const auto x_bin = make_bin_index(xlo, xhi, nx_bins);
          const auto y_bin = make_bin_index(ylo, yhi, ny_bins);

          auto out = outputs[0].mutable_view<double, 2>();
          for (int ix = 0; ix < nx_bins; ++ix)
            for (int iy = 0; iy < ny_bins; ++iy)
              out(ix, iy) = 0.0;

          if (inputs[0].desc().rank != 3) {
            const auto count =
                static_cast<std::size_t>(inputs[0].desc().element_count());
            for (std::size_t input = 1; input < inputs.size(); ++input) {
              if (inputs[input].desc().element_count() != count) {
                throw BufferContractError(
                    BufferContractReason::kDescriptorStorageMismatch,
                    "bounded histogram inputs must have matching extents");
              }
            }
            const auto x_values = make_real_buffer_accessor(inputs[0]);
            const auto y_values = make_real_buffer_accessor(inputs[1]);
            std::optional<RealBufferAccessor> weights;
            if (inputs.size() == 3)
              weights = make_real_buffer_accessor(inputs[2]);
            for (std::size_t index = 0; index < count; ++index) {
              const int ix = x_bin(x_values(index));
              if (ix < 0) continue;
              const int iy = y_bin(y_values(index));
              if (iy < 0) continue;
              const double weight = weights ? (*weights)(index) : 1.0;
              if (std::isfinite(weight))
                out(ix, iy) = static_cast<double>(out(ix, iy)) + weight;
            }
            return hpx::make_ready_future();
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

          const int32_t block_lo[] = {box.lo.x, box.lo.y, box.lo.z};
          const int32_t block_hi[] = {box.hi.x, box.hi.y, box.hi.z};
          const auto* coverage = params.coverage_index.get();
          if (coverage && !coverage->intersects(block_lo, block_hi)) coverage = nullptr;
          const double cell_volume =
              level.geom.dx[0] * level.geom.dx[1] * level.geom.dx[2];
          auto accumulate = [&](auto... typed_inputs) {
            static_assert(sizeof...(typed_inputs) == 2 ||
                          sizeof...(typed_inputs) == 3);
            const auto typed = std::tuple{typed_inputs...};
            const auto x_values = std::get<0>(typed).grid();
            const auto y_values = std::get<1>(typed).grid();
            for (int i = 0; i < nx; ++i) {
              const int gi = box.lo.x + i;
              for (int j = 0; j < ny; ++j) {
                const int gj = box.lo.y + j;
                for (int k = 0; k < nz; ++k) {
                  const int32_t cell[] = {gi, gj, box.lo.z + k};
                  if (coverage && coverage->contains(cell)) continue;
                  const double x = static_cast<double>(x_values(i, j, k));
                  const int ix = x_bin(x);
                  if (ix < 0) continue;
                  const int iy = y_bin(static_cast<double>(y_values(i, j, k)));
                  if (iy < 0) continue;
                  double weight = 1.0;
                  if constexpr (sizeof...(typed_inputs) == 3) {
                    weight =
                        static_cast<double>(std::get<2>(typed).grid()(i, j, k));
                    if (!std::isfinite(weight))
                      continue;
                  } else if (params.weight_mode == "cell_mass") {
                    weight = x * cell_volume;
                    if (!std::isfinite(weight))
                      continue;
                  } else if (params.weight_mode == "cell_volume") {
                    weight = cell_volume;
                  }
                  out(ix, iy) = static_cast<double>(out(ix, iy)) + weight;
                }
              }
            }
          };
          if (inputs.size() == 2) {
            visit_real_buffers_exact<2>(inputs, accumulate);
          } else {
            visit_real_buffers_exact<3>(inputs, accumulate);
          }
          return hpx::make_ready_future();
        });
  }
}

} // namespace kangaroo
