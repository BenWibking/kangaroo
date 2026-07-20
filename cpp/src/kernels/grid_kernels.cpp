#include "default_kernel_families.hpp"

#include "kernel_buffer_support.hpp"
#include "kernel_param_support.hpp"
#include "projection_kernel_support.hpp"

#include "amrexpr.hpp"

#include <iostream>
#include <mutex>
#include <unordered_map>
#include <variant>

#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/synchronization/mutex.hpp>

namespace kangaroo {

void register_grid_kernels(KernelRegistry &registry) {
  static const bool log_locality = []() {
    const char *env = std::getenv("KANGAROO_LOG_LOCALITY");
    return env != nullptr && *env != '\0' && *env != '0';
  }();
  {
    using Params = UniformSliceCellParams;

    /**
     * @brief Accumulates cell averages and sampled area on a uniform slice.
     * @par Chunk inputs `inputs[0]` is a real-valued cell-centered block grid.
     * @par Typed parameters `axis`, `coord`, optional `plane_index`,
     * `rect`, `resolution`, and `covered_boxes` define the sampled AMR plane.
     * @par Chunk outputs `outputs[0]` and `outputs[1]` are f64 images
     * containing the area-weighted value sum and sampled area, respectively.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "uniform_slice_cellavg_accumulate",
                   .n_inputs = 1,
                   .n_outputs = 2,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block,
           std::span<const ChunkBuffer> inputs, const NeighborViews &,
           std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params = require_kernel_params<Params>(
              kernel_params, "uniform_slice_cellavg_accumulate");

          const auto out_nx = params.resolution[0];
          const auto out_ny = params.resolution[1];
          if (outputs.size() < 2 || inputs.empty() || out_nx <= 0 ||
              out_ny <= 0) {
            return hpx::make_ready_future();
          }

          auto out_sum = outputs[0].mutable_byte_view();
          auto out_area = outputs[1].mutable_byte_view();
          std::fill(out_sum.begin(), out_sum.end(), std::uint8_t{0});
          std::fill(out_area.begin(), out_area.end(), std::uint8_t{0});

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

          const int axis = params.axis;
          int u_axis = 0;
          int v_axis = 1;
          if (axis == 0) {
            u_axis = 1;
            v_axis = 2;
          } else if (axis == 1) {
            u_axis = 0;
            v_axis = 2;
          } else {
            u_axis = 0;
            v_axis = 1;
          }

          auto cell_index = [&](int ax, double coord) -> int {
            const double x0 = level.geom.x0[ax];
            const double dx = level.geom.dx[ax];
            const int origin = level.geom.index_origin[ax];
            if (dx == 0.0) {
              return origin;
            }
            const double idx_f = (coord - x0) / dx;
            return static_cast<int>(std::floor(idx_f)) + origin;
          };

          const int k_global = params.has_plane_index
                                   ? params.plane_index
                                   : cell_index(axis, params.coord);
          const int k_local = (axis == 0 ? k_global - box.lo.x
                                         : (axis == 1 ? k_global - box.lo.y
                                                      : k_global - box.lo.z));
          if ((axis == 0 && (k_local < 0 || k_local >= nx)) ||
              (axis == 1 && (k_local < 0 || k_local >= ny)) ||
              (axis == 2 && (k_local < 0 || k_local >= nz))) {
            return hpx::make_ready_future();
          }

          const double u0 = params.rect[0];
          const double v0 = params.rect[1];
          const double u1 = params.rect[2];
          const double v1 = params.rect[3];
          const double umin = std::min(u0, u1);
          const double umax = std::max(u0, u1);
          const double vmin = std::min(v0, v1);
          const double vmax = std::max(v0, v1);
          const double du =
              (out_nx > 0) ? (umax - umin) / static_cast<double>(out_nx) : 0.0;
          const double dv =
              (out_ny > 0) ? (vmax - vmin) / static_cast<double>(out_ny) : 0.0;
          if (du == 0.0 || dv == 0.0) {
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

          auto cell_edge = [&](int ax, int idx) -> double {
            return level.geom.x0[ax] +
                   (idx - level.geom.index_origin[ax]) * level.geom.dx[ax];
          };

          visit_real_buffers_exact<1>(inputs.first(1), [&](auto input_buffer) {
            const auto input = input_buffer.grid();
            for (int v_local = 0;
                 v_local < (v_axis == 0 ? nx : (v_axis == 1 ? ny : nz));
                 ++v_local) {
              const int v_global =
                  (v_axis == 0 ? v_local + box.lo.x
                               : (v_axis == 1 ? v_local + box.lo.y
                                              : v_local + box.lo.z));
              for (int u_local = 0;
                   u_local < (u_axis == 0 ? nx : (u_axis == 1 ? ny : nz));
                   ++u_local) {
                const int u_global =
                    (u_axis == 0 ? u_local + box.lo.x
                                 : (u_axis == 1 ? u_local + box.lo.y
                                                : u_local + box.lo.z));
                int idx_global[3]{0, 0, 0};
                idx_global[axis] = k_global;
                idx_global[u_axis] = u_global;
                idx_global[v_axis] = v_global;
                if (covered(idx_global[0], idx_global[1], idx_global[2])) {
                  continue;
                }

                int idx_local[3]{0, 0, 0};
                idx_local[axis] = k_local;
                idx_local[u_axis] = u_local;
                idx_local[v_axis] = v_local;
                const double value = static_cast<double>(
                    input(idx_local[0], idx_local[1], idx_local[2]));

                const double u_cell_lo = cell_edge(u_axis, u_global);
                const double u_cell_hi = u_cell_lo + level.geom.dx[u_axis];
                const double v_cell_lo = cell_edge(v_axis, v_global);
                const double v_cell_hi = v_cell_lo + level.geom.dx[v_axis];

                int i0 = static_cast<int>(std::floor((u_cell_lo - umin) / du));
                int i1 = static_cast<int>(std::floor((u_cell_hi - umin) / du));
                int j0 = static_cast<int>(std::floor((v_cell_lo - vmin) / dv));
                int j1 = static_cast<int>(std::floor((v_cell_hi - vmin) / dv));
                if (i1 < 0 || j1 < 0 || i0 >= out_nx || j0 >= out_ny) {
                  continue;
                }
                i0 = std::max(i0, 0);
                j0 = std::max(j0, 0);
                i1 = std::min(i1, out_nx - 1);
                j1 = std::min(j1, out_ny - 1);

                for (int j = j0; j <= j1; ++j) {
                  const double pv0 = vmin + static_cast<double>(j) * dv;
                  const double pv1 = pv0 + dv;
                  const double ov = std::max(0.0, std::min(v_cell_hi, pv1) -
                                                      std::max(v_cell_lo, pv0));
                  if (ov <= 0.0) {
                    continue;
                  }
                  for (int i = i0; i <= i1; ++i) {
                    const double pu0 = umin + static_cast<double>(i) * du;
                    const double pu1 = pu0 + du;
                    const double ou =
                        std::max(0.0, std::min(u_cell_hi, pu1) -
                                          std::max(u_cell_lo, pu0));
                    if (ou <= 0.0) {
                      continue;
                    }
                    const double area = ou * ov;
                    const std::size_t out_idx =
                        static_cast<std::size_t>(j) * out_nx + i;
                    store_buffer_scalar(
                        out_sum.data(), out_idx,
                        load_buffer_scalar<double>(out_sum.data(), out_idx) +
                            value * area);
                    store_buffer_scalar(
                        out_area.data(), out_idx,
                        load_buffer_scalar<double>(out_area.data(), out_idx) +
                            area);
                  }
                }
              }
            }
          });

          return hpx::make_ready_future();
        });
  }
  {
    using Params = UniformProjectionParams;

    /**
     * @brief Projects uncovered grid cells onto a uniform image plane.
     * @par Chunk inputs `inputs[0]` is a real-valued cell-centered block grid.
     * @par Typed parameters `axis`, `axis_bounds`, `rect`, `resolution`,
     * and `covered_boxes` define the projection slab and AMR exclusion regions.
     * @par Chunk outputs `outputs[0]` is an f64 image of line-integrated
     * values.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "uniform_projection_accumulate",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block,
           std::span<const ChunkBuffer> inputs, const NeighborViews &,
           std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params = require_kernel_params<Params>(
              kernel_params, "uniform_projection_accumulate");

          const auto out_nx = params.resolution[0];
          const auto out_ny = params.resolution[1];
          if (outputs.empty() || inputs.empty() || out_nx <= 0 || out_ny <= 0) {
            return hpx::make_ready_future();
          }

          auto out_sum = outputs[0].mutable_byte_view();
          std::fill(out_sum.begin(), out_sum.end(), std::uint8_t{0});

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

          const int axis = params.axis;
          int u_axis = 0;
          int v_axis = 1;
          if (axis == 0) {
            u_axis = 1;
            v_axis = 2;
          } else if (axis == 1) {
            u_axis = 0;
            v_axis = 2;
          } else {
            u_axis = 0;
            v_axis = 1;
          }

          const double u0 = params.rect[0];
          const double v0 = params.rect[1];
          const double u1 = params.rect[2];
          const double v1 = params.rect[3];
          const double umin = std::min(u0, u1);
          const double umax = std::max(u0, u1);
          const double vmin = std::min(v0, v1);
          const double vmax = std::max(v0, v1);
          const double du =
              (out_nx > 0) ? (umax - umin) / static_cast<double>(out_nx) : 0.0;
          const double dv =
              (out_ny > 0) ? (vmax - vmin) / static_cast<double>(out_ny) : 0.0;
          if (du == 0.0 || dv == 0.0) {
            return hpx::make_ready_future();
          }

          const double a0 = params.axis_bounds[0];
          const double a1 = params.axis_bounds[1];
          const double amin = std::min(a0, a1);
          const double amax = std::max(a0, a1);
          if (!(amax > amin)) {
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
            covered_mask.assign(block_cells, 0);
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

          auto cell_edge = [&](int ax, int idx) -> double {
            return level.geom.x0[ax] +
                   (idx - level.geom.index_origin[ax]) * level.geom.dx[ax];
          };

          std::size_t candidate_cells = 0;
          std::size_t covered_skips = 0;
          std::size_t bounds_skips = 0;
          std::size_t deposited_cells = 0;

          visit_real_buffers_exact<1>(inputs.first(1), [&](auto input_buffer) {
            const auto input = input_buffer.grid();
            for (int i = 0; i < nx; ++i) {
              const int gx = box.lo.x + i;
              for (int j = 0; j < ny; ++j) {
                const int gy = box.lo.y + j;
                for (int k = 0; k < nz; ++k) {
                  ++candidate_cells;
                  const auto data_idx = logical_index(i, j, k);
                  if (!covered_mask.empty() && covered_mask[data_idx] != 0) {
                    ++covered_skips;
                    continue;
                  }
                  const int gz = box.lo.z + k;

                  const int a_global = axis == 0 ? gx : (axis == 1 ? gy : gz);
                  const double a_cell_lo = cell_edge(axis, a_global);
                  const double a_cell_hi = a_cell_lo + level.geom.dx[axis];
                  const double oa =
                      std::max(0.0, std::min(a_cell_hi, amax) -
                                        std::max(a_cell_lo, amin));
                  if (oa <= 0.0) {
                    ++bounds_skips;
                    continue;
                  }

                  const int u_global =
                      u_axis == 0 ? gx : (u_axis == 1 ? gy : gz);
                  const int v_global =
                      v_axis == 0 ? gx : (v_axis == 1 ? gy : gz);
                  const double u_cell_lo = cell_edge(u_axis, u_global);
                  const double u_cell_hi = u_cell_lo + level.geom.dx[u_axis];
                  const double v_cell_lo = cell_edge(v_axis, v_global);
                  const double v_cell_hi = v_cell_lo + level.geom.dx[v_axis];

                  int i0 =
                      static_cast<int>(std::floor((u_cell_lo - umin) / du));
                  int i1 =
                      static_cast<int>(std::floor((u_cell_hi - umin) / du));
                  int j0 =
                      static_cast<int>(std::floor((v_cell_lo - vmin) / dv));
                  int j1 =
                      static_cast<int>(std::floor((v_cell_hi - vmin) / dv));
                  if (i1 < 0 || j1 < 0 || i0 >= out_nx || j0 >= out_ny) {
                    ++bounds_skips;
                    continue;
                  }
                  i0 = std::max(i0, 0);
                  j0 = std::max(j0, 0);
                  i1 = std::min(i1, out_nx - 1);
                  j1 = std::min(j1, out_ny - 1);

                  const double value = static_cast<double>(input(i, j, k));

                  for (int jj = j0; jj <= j1; ++jj) {
                    const double pv0 = vmin + static_cast<double>(jj) * dv;
                    const double pv1 = pv0 + dv;
                    const double ov =
                        std::max(0.0, std::min(v_cell_hi, pv1) -
                                          std::max(v_cell_lo, pv0));
                    if (ov <= 0.0) {
                      continue;
                    }
                    for (int ii = i0; ii <= i1; ++ii) {
                      const double pu0 = umin + static_cast<double>(ii) * du;
                      const double pu1 = pu0 + du;
                      const double ou =
                          std::max(0.0, std::min(u_cell_hi, pu1) -
                                            std::max(u_cell_lo, pu0));
                      if (ou <= 0.0) {
                        continue;
                      }
                      const double volume = ou * ov * oa;
                      const std::size_t out_idx =
                          static_cast<std::size_t>(jj) * out_nx + ii;
                      store_buffer_scalar(
                          out_sum.data(), out_idx,
                          load_buffer_scalar<double>(out_sum.data(), out_idx) +
                              value * volume);
                      ++deposited_cells;
                    }
                  }
                }
              }
            }
          });

          double total_sum = 0.0;
          for (std::size_t idx = 0; idx < static_cast<std::size_t>(out_nx) *
                                              static_cast<std::size_t>(out_ny);
               ++idx) {
            total_sum += load_buffer_scalar<double>(out_sum.data(), idx);
          }
          log_projection_kernel_summary(
              "uniform_projection_accumulate", -1, block,
              covered_box_count(params.covered_boxes), candidate_cells,
              covered_skips, bounds_skips, deposited_cells, total_sum);

          return hpx::make_ready_future();
        });
  }
  {
    using Params = FieldExprParams;

    using FieldExprExecutor =
        std::variant<std::monostate, amrexpr::ParserExecutor<1>,
                     amrexpr::ParserExecutor<2>, amrexpr::ParserExecutor<3>,
                     amrexpr::ParserExecutor<4>, amrexpr::ParserExecutor<5>,
                     amrexpr::ParserExecutor<6>, amrexpr::ParserExecutor<7>,
                     amrexpr::ParserExecutor<8>>;

    struct PreparedParams {
      Params params;
      amrexpr::Parser parser;
      FieldExprExecutor executor;

      explicit PreparedParams(Params decoded) : params(std::move(decoded)) {
        if (params.expression.empty()) {
          throw std::runtime_error(
              "field_expr requires a non-empty expression");
        }
        if (params.variables.empty()) {
          throw std::runtime_error("field_expr requires at least one variable");
        }
        if (params.variables.size() > 8) {
          throw std::runtime_error(
              "field_expr currently supports at most 8 variables");
        }

        try {
          parser.define(params.expression);
          parser.registerVariables(params.variables);
          switch (params.variables.size()) {
          case 1:
            executor = parser.compileHost<1>();
            break;
          case 2:
            executor = parser.compileHost<2>();
            break;
          case 3:
            executor = parser.compileHost<3>();
            break;
          case 4:
            executor = parser.compileHost<4>();
            break;
          case 5:
            executor = parser.compileHost<5>();
            break;
          case 6:
            executor = parser.compileHost<6>();
            break;
          case 7:
            executor = parser.compileHost<7>();
            break;
          case 8:
            executor = parser.compileHost<8>();
            break;
          default:
            throw std::runtime_error(
                "field_expr variable count is out of range");
          }
        } catch (const std::runtime_error &e) {
          throw std::runtime_error(std::string("field_expr parse failed: ") +
                                   e.what());
        }
      }
    };

    struct FieldExprCache {
      hpx::mutex mutex;
      std::unordered_map<std::string, std::shared_ptr<const PreparedParams>>
          entries;
    };
    auto cache = std::make_shared<FieldExprCache>();
    auto prepare = [cache](const Params &params) {
      std::string key = params.expression;
      for (const auto &variable : params.variables) {
        key.push_back('\0');
        key.append(variable);
      }
      std::lock_guard<hpx::mutex> lock(cache->mutex);
      if (const auto it = cache->entries.find(key);
          it != cache->entries.end()) {
        return it->second;
      }
      auto prepared = std::make_shared<const PreparedParams>(params);
      cache->entries.emplace(std::move(key), prepared);
      return prepared;
    };

    /**
     * @brief Evaluates a scalar field expression independently in every grid
     * cell.
     * @par Chunk inputs `inputs[0..N)` are matching real-valued block grids.
     * @par Typed parameters `expression` is the parser expression and
     * `variables` names the inputs in order; one to eight variables are
     * supported.
     * @par Chunk outputs `outputs[0]` is the f32 or f64 expression value grid.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "field_expr",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [prepare](const LevelMeta &level, int32_t block,
                  std::span<const ChunkBuffer> inputs, const NeighborViews &,
                  std::span<ChunkBuffer> outputs,
                  const KernelParamsIR &kernel_params) {
          const auto &typed =
              require_kernel_params<Params>(kernel_params, "field_expr");
          const auto prepared = prepare(typed);
          const auto &params = prepared->params;

          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          if (params.variables.size() != inputs.size()) {
            throw std::runtime_error(
                "field_expr variables/input size mismatch");
          }
          const auto &input_desc = inputs.front().desc();
          for (const auto &input : inputs.subspan(1)) {
            const auto &desc = input.desc();
            if (desc.rank != input_desc.rank ||
                !std::equal(input_desc.extents.begin(),
                            input_desc.extents.begin() + input_desc.rank,
                            desc.extents.begin())) {
              throw BufferContractError(
                  BufferContractReason::kInvalidExtent,
                  "field_expr inputs must have matching shapes");
            }
          }
          std::array<RealBufferAccessor, 8> input_views{};
          for (std::size_t index = 0; index < inputs.size(); ++index)
            input_views[index] = make_real_buffer_accessor(inputs[index]);

          auto out = outputs[0].mutable_byte_view();
          const bool output_f64 = outputs[0].desc().scalar == ScalarType::kF64;
          if (!output_f64 && outputs[0].desc().scalar != ScalarType::kF32) {
            throw BufferContractError(
                BufferContractReason::kScalarMismatch,
                "field expression output must be f32 or f64");
          }
          const std::size_t cell_count = input_desc.element_count();
          auto read_value = [&](int variable, std::size_t index) {
            return input_views[static_cast<std::size_t>(variable)](index);
          };
          auto write_value = [&](std::size_t index, double value) {
            if (output_f64)
              store_buffer_scalar(out.data(), index, value);
            else
              store_buffer_scalar(out.data(), index, static_cast<float>(value));
          };

#define KANGAROO_FIELD_EXPR_CASE(N)                                            \
  case N: {                                                                    \
    const auto &executable =                                                   \
        std::get<amrexpr::ParserExecutor<N>>(prepared->executor);              \
    for (std::size_t index = 0; index < cell_count; ++index) {                 \
      double vars[N];                                                          \
      for (int variable = 0; variable < N; ++variable)                         \
        vars[variable] = read_value(variable, index);                          \
      write_value(index, executable(vars));                                    \
    }                                                                          \
    break;                                                                     \
  }
          switch (inputs.size()) {
            KANGAROO_FIELD_EXPR_CASE(1)
            KANGAROO_FIELD_EXPR_CASE(2)
            KANGAROO_FIELD_EXPR_CASE(3)
            KANGAROO_FIELD_EXPR_CASE(4)
            KANGAROO_FIELD_EXPR_CASE(5)
            KANGAROO_FIELD_EXPR_CASE(6)
            KANGAROO_FIELD_EXPR_CASE(7)
            KANGAROO_FIELD_EXPR_CASE(8)
          default:
            throw std::runtime_error(
                "field_expr variable count is out of range");
          }
#undef KANGAROO_FIELD_EXPR_CASE
          return hpx::make_ready_future();
        });

    /**
     * @brief Evaluates a real-valued comparison into a normalized u8 mask.
     * @par Chunk inputs `inputs[0..N)` are matching f32 or f64 grids.
     * @par Typed parameters use the same compiled expression representation as
     * `field_expr`.
     * @par Chunk outputs `outputs[0]` is a u8 grid containing only 0 or 1.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "mesh_compare",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [prepare](const LevelMeta &, int32_t,
                  std::span<const ChunkBuffer> inputs, const NeighborViews &,
                  std::span<ChunkBuffer> outputs,
                  const KernelParamsIR &kernel_params) {
          const auto &typed =
              require_kernel_params<Params>(kernel_params, "mesh_compare");
          const auto prepared = prepare(typed);
          const auto &params = prepared->params;
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          if (params.variables.size() != inputs.size()) {
            throw std::runtime_error(
                "mesh_compare variables/input size mismatch");
          }
          const auto &input_desc = inputs.front().desc();
          for (const auto &input : inputs.subspan(1)) {
            const auto &desc = input.desc();
            if (desc.rank != input_desc.rank ||
                !std::equal(input_desc.extents.begin(),
                            input_desc.extents.begin() + input_desc.rank,
                            desc.extents.begin())) {
              throw BufferContractError(
                  BufferContractReason::kInvalidExtent,
                  "mesh_compare inputs must have matching shapes");
            }
          }
          if (outputs[0].desc().scalar != ScalarType::kU8) {
            throw BufferContractError(BufferContractReason::kScalarMismatch,
                                      "mesh_compare output must be u8");
          }
          std::array<RealBufferAccessor, 8> input_views{};
          for (std::size_t index = 0; index < inputs.size(); ++index)
            input_views[index] = make_real_buffer_accessor(inputs[index]);

          auto out = outputs[0].mutable_byte_view();
          const std::size_t cell_count = input_desc.element_count();
          auto read_value = [&](int variable, std::size_t index) {
            return input_views[static_cast<std::size_t>(variable)](index);
          };
          auto write_value = [&](std::size_t index, double value) {
            store_buffer_scalar<std::uint8_t>(
                out.data(), index,
                value != 0.0 ? std::uint8_t{1} : std::uint8_t{0});
          };

#define KANGAROO_MESH_COMPARE_CASE(N)                                          \
  case N: {                                                                    \
    const auto &executable =                                                   \
        std::get<amrexpr::ParserExecutor<N>>(prepared->executor);              \
    for (std::size_t index = 0; index < cell_count; ++index) {                 \
      double vars[N];                                                          \
      for (int variable = 0; variable < N; ++variable)                         \
        vars[variable] = read_value(variable, index);                          \
      write_value(index, executable(vars));                                    \
    }                                                                          \
    break;                                                                     \
  }
          switch (inputs.size()) {
            KANGAROO_MESH_COMPARE_CASE(1)
            KANGAROO_MESH_COMPARE_CASE(2)
            KANGAROO_MESH_COMPARE_CASE(3)
            KANGAROO_MESH_COMPARE_CASE(4)
            KANGAROO_MESH_COMPARE_CASE(5)
            KANGAROO_MESH_COMPARE_CASE(6)
            KANGAROO_MESH_COMPARE_CASE(7)
            KANGAROO_MESH_COMPARE_CASE(8)
          default:
            throw std::runtime_error(
                "mesh_compare variable count is out of range");
          }
#undef KANGAROO_MESH_COMPARE_CASE
          return hpx::make_ready_future();
        });
  }
  /**
   * @brief Computes elementwise logical conjunction of two mesh masks.
   * @par Chunk inputs `inputs[0]` and `inputs[1]` are matching u8 grids.
   * @par Chunk outputs `outputs[0]` is a normalized u8 mask.
   */
  registry.register_kernel(
      KernelDesc{.name = "mesh_mask_and",
                 .n_inputs = 2,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
         const NeighborViews &, std::span<ChunkBuffer> outputs,
         const KernelParamsIR &) {
        if (inputs.size() < 2 || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto &left_desc = inputs[0].desc();
        const auto &right_desc = inputs[1].desc();
        const auto &output_desc = outputs[0].desc();
        if (left_desc.scalar != ScalarType::kU8 ||
            right_desc.scalar != ScalarType::kU8 ||
            output_desc.scalar != ScalarType::kU8) {
          throw BufferContractError(BufferContractReason::kScalarMismatch,
                                    "mesh_mask_and requires u8 buffers");
        }
        if (left_desc.rank != right_desc.rank ||
            left_desc.rank != output_desc.rank ||
            !std::equal(left_desc.extents.begin(),
                        left_desc.extents.begin() + left_desc.rank,
                        right_desc.extents.begin()) ||
            !std::equal(left_desc.extents.begin(),
                        left_desc.extents.begin() + left_desc.rank,
                        output_desc.extents.begin())) {
          throw BufferContractError(
              BufferContractReason::kInvalidExtent,
              "mesh_mask_and inputs and output must have matching shapes");
        }
        const auto left = inputs[0].byte_view();
        const auto right = inputs[1].byte_view();
        auto output = outputs[0].mutable_byte_view();
        const auto count = left_desc.element_count();
        for (std::size_t index = 0; index < count; ++index) {
          output[index] = left[index] != 0 && right[index] != 0
                              ? std::uint8_t{1}
                              : std::uint8_t{0};
        }
        return hpx::make_ready_future();
      });
  {
    using Params = UniformSliceParams;

    /**
     * @brief Samples a grid block onto its overlapping region of a uniform
     * slice.
     * @par Chunk inputs `inputs[0]` is a real-valued cell-centered block grid.
     * @par Typed parameters `axis`, `coord`, `rect`, and `resolution`
     * define the sampling plane and output image.
     * @par Chunk outputs `outputs[0]` is an f32 or f64 nearest-cell slice
     * image; pixels outside this block are NaN.
     */
    registry.register_typed_kernel<Params>(
        KernelDesc{.name = "uniform_slice",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block,
           std::span<const ChunkBuffer> inputs, const NeighborViews &,
           std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          if (log_locality) {
            std::cout << "[kangaroo] uniform_slice block=" << block
                      << " locality=" << hpx::get_locality_id() << std::endl;
          }
          const auto &params =
              require_kernel_params<Params>(kernel_params, "uniform_slice");

          const auto out_nx = params.resolution[0];
          const auto out_ny = params.resolution[1];
          if (outputs.empty() || inputs.empty() || out_nx <= 0 || out_ny <= 0) {
            return hpx::make_ready_future();
          }
          auto output_storage = outputs[0].mutable_byte_view();
          std::fill(output_storage.begin(), output_storage.end(),
                    std::uint8_t{0});

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

          const int axis = params.axis;
          int u_axis = 0;
          int v_axis = 1;
          if (axis == 0) {
            u_axis = 1;
            v_axis = 2;
          } else if (axis == 1) {
            u_axis = 0;
            v_axis = 2;
          } else {
            u_axis = 0;
            v_axis = 1;
          }

          auto cell_index = [&](int ax, double coord) -> int {
            const double x0 = level.geom.x0[ax];
            const double dx = level.geom.dx[ax];
            const int origin = level.geom.index_origin[ax];
            if (dx == 0.0) {
              return origin;
            }
            const double idx_f = (coord - x0) / dx;
            return static_cast<int>(std::floor(idx_f)) + origin;
          };

          const int k_global = cell_index(axis, params.coord);
          const int k_local = (axis == 0 ? k_global - box.lo.x
                                         : (axis == 1 ? k_global - box.lo.y
                                                      : k_global - box.lo.z));
          if ((axis == 0 && (k_local < 0 || k_local >= nx)) ||
              (axis == 1 && (k_local < 0 || k_local >= ny)) ||
              (axis == 2 && (k_local < 0 || k_local >= nz))) {
            return hpx::make_ready_future();
          }

          const double u0 = params.rect[0];
          const double v0 = params.rect[1];
          const double u1 = params.rect[2];
          const double v1 = params.rect[3];
          const double du =
              (out_nx > 0) ? (u1 - u0) / static_cast<double>(out_nx) : 0.0;
          const double dv =
              (out_ny > 0) ? (v1 - v0) / static_cast<double>(out_ny) : 0.0;

          const auto input = make_real_grid_accessor(inputs[0]);
          auto sample_input = [&](int u_local, int v_local) -> double {
            if (u_local < 0 || v_local < 0) {
              return 0.0;
            }
            if (!((u_axis == 0 && u_local < nx) ||
                  (u_axis == 1 && u_local < ny) ||
                  (u_axis == 2 && u_local < nz))) {
              return 0.0;
            }
            if (!((v_axis == 0 && v_local < nx) ||
                  (v_axis == 1 && v_local < ny) ||
                  (v_axis == 2 && v_local < nz))) {
              return 0.0;
            }

            int ii = 0;
            int jj = 0;
            int kk = 0;
            if (axis == 0) {
              ii = k_local;
              jj = u_axis == 1 ? u_local : v_local;
              kk = u_axis == 2 ? u_local : v_local;
            } else if (axis == 1) {
              ii = u_axis == 0 ? u_local : v_local;
              jj = k_local;
              kk = u_axis == 2 ? u_local : v_local;
            } else {
              ii = u_axis == 0 ? u_local : v_local;
              jj = u_axis == 1 ? u_local : v_local;
              kk = k_local;
            }
            return input(ii, jj, kk);
          };

          auto fill_output = [&]<typename T>() {
            auto out = outputs[0].mutable_view<T, 2>();
            for (int j = 0; j < out_ny; ++j) {
              const double v = v0 + (static_cast<double>(j) + 0.5) * dv;
              const int v_global = cell_index(v_axis, v);
              const int v_local = v_axis == 0
                                      ? v_global - box.lo.x
                                      : (v_axis == 1 ? v_global - box.lo.y
                                                     : v_global - box.lo.z);
              for (int i = 0; i < out_nx; ++i) {
                const double u = u0 + (static_cast<double>(i) + 0.5) * du;
                const int u_global = cell_index(u_axis, u);
                const int u_local = u_axis == 0
                                        ? u_global - box.lo.x
                                        : (u_axis == 1 ? u_global - box.lo.y
                                                       : u_global - box.lo.z);
                out(j, i) = static_cast<T>(sample_input(u_local, v_local));
              }
            }
          };
          if (outputs[0].desc().scalar == ScalarType::kF32)
            fill_output.template operator()<float>();
          else if (outputs[0].desc().scalar == ScalarType::kF64)
            fill_output.template operator()<double>();
          else
            throw BufferContractError(
                BufferContractReason::kScalarMismatch,
                "uniform_slice output must be f32 or f64");
          return hpx::make_ready_future();
        });
  }
  /**
   * @brief Computes the cell-centered magnitude of the velocity curl.
   * @par Chunk inputs Either one real block grid with three gradient components
   * per cell, or three such grids holding the gradients of velocity x/y/z.
   * @par Typed parameters None.
   * @par Chunk outputs `outputs[0]` is an f64 scalar block grid containing
   * gradient magnitude for one input or velocity-curl magnitude for three
   * inputs.
   */
  registry.register_kernel(
      KernelDesc{.name = "vorticity_mag",
                 .n_inputs = 1,
                 .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta &level, int32_t block,
         std::span<const ChunkBuffer> inputs, const NeighborViews &,
         std::span<ChunkBuffer> outputs, const KernelParamsIR &) {
        if (outputs.empty() || block < 0 ||
            static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }
        const auto &box = level.boxes.at(static_cast<std::size_t>(block));
        const int32_t nx = box.hi.x - box.lo.x + 1;
        const int32_t ny = box.hi.y - box.lo.y + 1;
        const int32_t nz = box.hi.z - box.lo.z + 1;
        if (nx <= 0 || ny <= 0 || nz <= 0) {
          return hpx::make_ready_future();
        }
        auto out = outputs[0].mutable_view<double, 3>();

        // Preferred path: three scalar gradient inputs [grad(vx), grad(vy),
        // grad(vz)].
        if (inputs.size() >= 3) {
          visit_real_buffers_exact<3>(
              inputs.first(3),
              [&](auto du_buffer, auto dv_buffer, auto dw_buffer) {
                const auto du = du_buffer.template tensor<4>();
                const auto dv = dv_buffer.template tensor<4>();
                const auto dw = dw_buffer.template tensor<4>();
                for (int i = 0; i < nx; ++i)
                  for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k) {
                      const double wx = static_cast<double>(dw(i, j, k, 1)) -
                                        static_cast<double>(dv(i, j, k, 2));
                      const double wy = static_cast<double>(du(i, j, k, 2)) -
                                        static_cast<double>(dw(i, j, k, 0));
                      const double wz = static_cast<double>(dv(i, j, k, 0)) -
                                        static_cast<double>(du(i, j, k, 1));
                      out(i, j, k) = std::sqrt(wx * wx + wy * wy + wz * wz);
                    }
              });
          return hpx::make_ready_future();
        }

        // One scalar gradient input -> |grad(S)|.
        if (!inputs.empty()) {
          visit_real_buffers_exact<1>(
              inputs.first(1), [&](auto gradient_buffer) {
                const auto gradient = gradient_buffer.template tensor<4>();
                for (int i = 0; i < nx; ++i)
                  for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k) {
                      const double gx =
                          static_cast<double>(gradient(i, j, k, 0));
                      const double gy =
                          static_cast<double>(gradient(i, j, k, 1));
                      const double gz =
                          static_cast<double>(gradient(i, j, k, 2));
                      out(i, j, k) = std::sqrt(gx * gx + gy * gy + gz * gz);
                    }
              });
        }
        return hpx::make_ready_future();
      });
}

} // namespace kangaroo
