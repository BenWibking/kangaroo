#include "default_kernel_families.hpp"

#include "default_kernel_support.hpp"

namespace kangaroo {

void register_histogram_kernels(KernelRegistry &registry) {
  {
    struct Params {
      std::array<double, 2> range{0.0, 1.0};
      int bins = 1;
      std::shared_ptr<const CoveredBoxListIR> covered_boxes;
    };

    auto decode_params = [](const msgpack::object &root) {
      Params params;
      if (const auto *range = find_msgpack_map_value(root, "range");
          range && range->type == msgpack::type::ARRAY &&
          range->via.array.size == 2) {
        params.range[0] = range->via.array.ptr[0].as<double>();
        params.range[1] = range->via.array.ptr[1].as<double>();
      }
      if (const auto *bins = find_msgpack_map_value(root, "bins");
          bins && (bins->type == msgpack::type::POSITIVE_INTEGER ||
                   bins->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bins = bins->as<int>();
      }
      params.covered_boxes = parse_covered_boxes_param(root);
      return params;
    };

    /**
     * @brief Accumulates uncovered grid-cell values into a one-dimensional histogram.
     * @par Chunk inputs `inputs[0]` is a real-valued block grid; optional
     * `inputs[1]` is a matching real-valued weight grid.
     * @par MessagePack parameters `range`, `bins`, and `covered_boxes` define the
     * histogram interval, bin count, and excluded AMR cells.
     * @par Chunk outputs `outputs[0]` is an f64 array of `bins` accumulated counts
     * or weights.
     */
    registry.register_kernel(
        KernelDesc{.name = "histogram1d_accumulate",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta &level, int32_t block,
                        std::span<const ChunkBuffer> inputs,
                        const NeighborViews &, std::span<ChunkBuffer> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto &params =
              decode_params_cached<Params>(params_msgpack, decode_params);

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
        },
        make_covered_box_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      std::array<double, 2> x_range{0.0, 1.0};
      std::array<double, 2> y_range{0.0, 1.0};
      std::array<int, 2> bins{1, 1};
      std::string weight_mode{"input"};
      std::shared_ptr<const CoveredBoxListIR> covered_boxes;
    };

    auto decode_params = [](const msgpack::object &root) {
      Params params;
      if (root.type == msgpack::type::MAP) {
        if (const auto *x_range = find_msgpack_map_value(root, "x_range");
            x_range && x_range->type == msgpack::type::ARRAY &&
            x_range->via.array.size == 2) {
          params.x_range[0] = x_range->via.array.ptr[0].as<double>();
          params.x_range[1] = x_range->via.array.ptr[1].as<double>();
        }
        if (const auto *y_range = find_msgpack_map_value(root, "y_range");
            y_range && y_range->type == msgpack::type::ARRAY &&
            y_range->via.array.size == 2) {
          params.y_range[0] = y_range->via.array.ptr[0].as<double>();
          params.y_range[1] = y_range->via.array.ptr[1].as<double>();
        }
        if (const auto *bins = find_msgpack_map_value(root, "bins");
            bins && bins->type == msgpack::type::ARRAY &&
            bins->via.array.size == 2) {
          params.bins[0] = bins->via.array.ptr[0].as<int>();
          params.bins[1] = bins->via.array.ptr[1].as<int>();
        }
        if (const auto *mode = find_msgpack_map_value(root, "weight_mode");
            mode && mode->type == msgpack::type::STR) {
          params.weight_mode = mode->as<std::string>();
        }
        params.covered_boxes = parse_covered_boxes_param(root);
      }
      return params;
    };

    /**
     * @brief Accumulates uncovered grid-cell samples into a two-dimensional histogram.
     * @par Chunk inputs `inputs[0]` and `inputs[1]` are matching real-valued x and
     * y block grids; optional `inputs[2]` is a matching weight grid.
     * @par MessagePack parameters `x_range`, `y_range`, `bins`, `weight_mode`, and
     * `covered_boxes` define binning, weighting, and excluded AMR cells.
     * @par Chunk outputs `outputs[0]` is an f64 two-dimensional histogram.
     */
    registry.register_kernel(
        KernelDesc{.name = "histogram2d_accumulate",
                   .n_inputs = 2,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta &level, int32_t block,
                        std::span<const ChunkBuffer> inputs,
                        const NeighborViews &, std::span<ChunkBuffer> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto &params =
              decode_params_cached<Params>(params_msgpack, decode_params);

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
              yhi <= ylo) {
            return hpx::make_ready_future();
          }

          auto out = outputs[0].mutable_view<double, 2>();
          for (int ix = 0; ix < nx_bins; ++ix)
            for (int iy = 0; iy < ny_bins; ++iy)
              out(ix, iy) = 0.0;

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
          const double cell_volume =
              level.geom.dx[0] * level.geom.dx[1] * level.geom.dx[2];
          const double inv_dx = static_cast<double>(nx_bins) / (xhi - xlo);
          const double inv_dy = static_cast<double>(ny_bins) / (yhi - ylo);
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
                  const int gk = box.lo.z + k;
                  if (covered(gi, gj, gk))
                    continue;
                  const double x = static_cast<double>(x_values(i, j, k));
                  const double y = static_cast<double>(y_values(i, j, k));
                  if (!std::isfinite(x) || !std::isfinite(y) || x < xlo ||
                      x > xhi || y < ylo || y > yhi)
                    continue;
                  const int ix =
                      x == xhi
                          ? nx_bins - 1
                          : static_cast<int>(std::floor((x - xlo) * inv_dx));
                  const int iy =
                      y == yhi
                          ? ny_bins - 1
                          : static_cast<int>(std::floor((y - ylo) * inv_dy));
                  if (ix < 0 || ix >= nx_bins || iy < 0 || iy >= ny_bins)
                    continue;
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
        },
        make_covered_box_params_preparer<Params>(decode_params));
  }
}

} // namespace kangaroo
