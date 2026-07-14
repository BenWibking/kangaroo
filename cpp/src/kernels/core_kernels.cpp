#include "default_kernel_families.hpp"

#include "amr_sampling_support.hpp"
#include "kernel_buffer_support.hpp"
#include "kernel_param_support.hpp"
#include "plotfile_kernel_support.hpp"

#include "kangaroo/amr_patch_codec.hpp"
#include "kangaroo/runtime.hpp"

namespace kangaroo {

void register_core_kernels(KernelRegistry &registry) {
  {
    using Params = AmrSubboxPackParams;

    /**
     * @brief Fetches an AMR sub-box and packs its selected cells contiguously.
     * @par Chunk inputs None; source chunks are fetched from the data service.
     * @par MessagePack parameters `input_field`, `input_version`, `input_step`,
     * `input_level`, and `halo_cells` identify and expand the source region.
     * @par Chunk outputs `outputs[0]` is an opaque, dynamically sized packed
     * AMR sub-box payload.
     */
    registry.register_kernel(
        KernelDesc{.name = "amr_subbox_fetch_pack",
                   .n_inputs = 0,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block, std::span<const ChunkBuffer>,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          if (outputs.empty() || block < 0 ||
              static_cast<std::size_t>(block) >= level.boxes.size()) {
            return hpx::make_ready_future();
          }

          const auto &params = require_kernel_params<Params>(
              kernel_params, "amr_subbox_fetch_pack");

          if (params.input_field < 0) {
            outputs[0].commit_dynamic_extent(0);
            return hpx::make_ready_future();
          }

          const RunMeta &meta = current_runmeta();
          if (params.input_step < 0 ||
              static_cast<std::size_t>(params.input_step) >=
                  meta.steps.size()) {
            outputs[0].commit_dynamic_extent(0);
            return hpx::make_ready_future();
          }
          const auto &step_meta =
              meta.steps.at(static_cast<std::size_t>(params.input_step));
          if (params.input_level < 0 ||
              static_cast<std::size_t>(params.input_level) >=
                  step_meta.levels.size()) {
            outputs[0].commit_dynamic_extent(0);
            return hpx::make_ready_future();
          }

          const auto &box = level.boxes.at(static_cast<std::size_t>(block));
          const int halo = std::max(1, params.halo_cells);
          const auto &target_geom = level.geom;
          double query_lo[3] = {0.0, 0.0, 0.0};
          double query_hi[3] = {0.0, 0.0, 0.0};
          for (int ax = 0; ax < 3; ++ax) {
            const int32_t lo =
                ax == 0 ? box.lo.x : (ax == 1 ? box.lo.y : box.lo.z);
            const int32_t hi =
                ax == 0 ? box.hi.x : (ax == 1 ? box.hi.y : box.hi.z);
            query_lo[ax] = cell_edge(target_geom, ax, lo) -
                           static_cast<double>(halo) * target_geom.dx[ax];
            query_hi[ax] = cell_edge(target_geom, ax, hi + 1) +
                           static_cast<double>(halo) * target_geom.dx[ax];
          }

          struct PendingPatch {
            int16_t level = 0;
            LevelGeom geom;
          };
          std::vector<PendingPatch> pending_patches;
          std::vector<hpx::future<SubboxView>> pending_subboxes;
          auto output_writer = outputs[0].begin_async_dynamic_write();

          DataServiceLocal data_service;
          for (int16_t lev = 0;
               lev < static_cast<int16_t>(step_meta.levels.size()); ++lev) {
            const auto &lev_meta =
                step_meta.levels.at(static_cast<std::size_t>(lev));
            int32_t req_lo[3];
            int32_t req_hi[3];
            for (int ax = 0; ax < 3; ++ax) {
              req_lo[ax] = coord_to_index(lev_meta.geom, ax, query_lo[ax]);
              req_hi[ax] = coord_to_index(lev_meta.geom, ax, query_hi[ax]);
            }
            for (int32_t b = 0; b < static_cast<int32_t>(lev_meta.boxes.size());
                 ++b) {
              if (lev == params.input_level && b == block) {
                continue;
              }
              const auto &ob = lev_meta.boxes.at(static_cast<std::size_t>(b));
              IndexBox3 request_box;
              request_box.lo[0] = std::max(ob.lo.x, req_lo[0]);
              request_box.lo[1] = std::max(ob.lo.y, req_lo[1]);
              request_box.lo[2] = std::max(ob.lo.z, req_lo[2]);
              request_box.hi[0] = std::min(ob.hi.x, req_hi[0]);
              request_box.hi[1] = std::min(ob.hi.y, req_hi[1]);
              request_box.hi[2] = std::min(ob.hi.z, req_hi[2]);
              if (request_box.hi[0] < request_box.lo[0] ||
                  request_box.hi[1] < request_box.lo[1] ||
                  request_box.hi[2] < request_box.lo[2]) {
                continue;
              }

              ChunkSubboxRef ref;
              ref.chunk = ChunkRef{params.input_step, lev, params.input_field,
                                   params.input_version, b};
              ref.chunk_box.lo[0] = ob.lo.x;
              ref.chunk_box.lo[1] = ob.lo.y;
              ref.chunk_box.lo[2] = ob.lo.z;
              ref.chunk_box.hi[0] = ob.hi.x;
              ref.chunk_box.hi[1] = ob.hi.y;
              ref.chunk_box.hi[2] = ob.hi.z;
              ref.request_box = request_box;
              pending_patches.push_back(
                  PendingPatch{.level = lev, .geom = lev_meta.geom});
              pending_subboxes.push_back(data_service.get_subbox(ref));
            }
          }

          return hpx::when_all(std::move(pending_subboxes))
              .then([pending_patches = std::move(pending_patches),
                     output_writer](auto &&all) mutable {
                auto ready_subboxes = all.get();
                std::vector<AmrPatchRecord> packed_patches;
                packed_patches.reserve(ready_subboxes.size());
                for (std::size_t i = 0; i < ready_subboxes.size(); ++i) {
                  auto sub = ready_subboxes[i].get();
                  if (sub.box.hi[0] < sub.box.lo[0] ||
                      sub.box.hi[1] < sub.box.lo[1] ||
                      sub.box.hi[2] < sub.box.lo[2] || sub.data.empty()) {
                    continue;
                  }

                  AmrPatchRecord pp;
                  pp.level = pending_patches[i].level;
                  pp.box = sub.box;
                  pp.geom = pending_patches[i].geom;
                  pp.data = std::move(sub.data);
                  packed_patches.push_back(std::move(pp));
                }

                const auto packed = encode_amr_patch_payload(packed_patches);
                output_writer.replace(packed.byte_view());
                return;
              });
        },
        [](const DynamicOutputBoundContext &context) {
          return estimate_amr_subbox_pack_capacity(context,
                                                   context.params<Params>());
        });
  }
  {
    using Params = GradStencilParams;

    /**
     * @brief Computes the selected velocity-gradient component on a grid block.
     * @par Chunk inputs `inputs[0]` is the local scalar grid; `inputs[1]`, when
     * supplied, is the opaque packed AMR halo payload.
     * @par MessagePack parameters `input_field`, `input_version`, `input_step`,
     * `input_level`, and `stencil_radius` identify the halo source and stencil.
     * @par Chunk outputs `outputs[0]` is an f64 block grid with three gradient
     * components per cell.
     */
    registry.register_kernel(
        KernelDesc{.name = "gradU_stencil",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block,
           std::span<const ChunkBuffer> inputs, const NeighborViews &,
           std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          if (inputs.size() < 2 || inputs[0].empty() || outputs.empty() ||
              block < 0 ||
              static_cast<std::size_t>(block) >= level.boxes.size()) {
            return hpx::make_ready_future();
          }

          const auto &params =
              require_kernel_params<Params>(kernel_params, "gradU_stencil");

          const auto &box = level.boxes.at(static_cast<std::size_t>(block));
          const int32_t nx = box.hi.x - box.lo.x + 1;
          const int32_t ny = box.hi.y - box.lo.y + 1;
          const int32_t nz = box.hi.z - box.lo.z + 1;
          if (nx <= 0 || ny <= 0 || nz <= 0) {
            return hpx::make_ready_future();
          }
          const int32_t stencil_radius = std::max(1, params.stencil_radius);
          auto out = outputs[0].mutable_view<double, 4>();

          SamplePatch self;
          self.level = 0;
          self.box.lo[0] = box.lo.x;
          self.box.lo[1] = box.lo.y;
          self.box.lo[2] = box.lo.z;
          self.box.hi[0] = box.hi.x;
          self.box.hi[1] = box.hi.y;
          self.box.hi[2] = box.hi.z;
          self.geom = level.geom;
          self.view = inputs[0];
          self.values = make_real_grid_accessor(self.view);

          const RunMeta &meta = current_runmeta();
          if (params.input_step < 0 ||
              static_cast<std::size_t>(params.input_step) >=
                  meta.steps.size()) {
            return hpx::make_ready_future();
          }
          const auto &step_meta =
              meta.steps.at(static_cast<std::size_t>(params.input_step));
          const int16_t target_level = params.input_level;
          if (target_level < 0 || static_cast<std::size_t>(target_level) >=
                                      step_meta.levels.size()) {
            return hpx::make_ready_future();
          }

          std::vector<SamplePatch> patches;
          patches.reserve(64);
          self.level = target_level;
          patches.push_back(std::move(self));
          if (!inputs[1].empty() &&
              inputs[1].desc().scalar == ScalarType::kOpaque) {
            auto decoded = decode_amr_patch_payload(inputs[1].byte_view());
            for (auto &record : decoded) {
              SamplePatch patch;
              patch.level = record.level;
              patch.box = record.box;
              patch.geom = record.geom;
              patch.view = std::move(record.data);
              patch.values = make_real_grid_accessor(patch.view);
              patches.push_back(std::move(patch));
            }
          }

          const auto &target_geom = level.geom;
          double query_lo[3] = {0.0, 0.0, 0.0};
          double query_hi[3] = {0.0, 0.0, 0.0};
          for (int ax = 0; ax < 3; ++ax) {
            const int32_t lo =
                ax == 0 ? box.lo.x : (ax == 1 ? box.lo.y : box.lo.z);
            const int32_t hi =
                ax == 0 ? box.hi.x : (ax == 1 ? box.hi.y : box.hi.z);
            query_lo[ax] = cell_edge(target_geom, ax, lo) - target_geom.dx[ax];
            query_hi[ax] =
                cell_edge(target_geom, ax, hi + 1) + target_geom.dx[ax];
          }

          int32_t domain_lo[3] = {std::numeric_limits<int32_t>::max(),
                                  std::numeric_limits<int32_t>::max(),
                                  std::numeric_limits<int32_t>::max()};
          int32_t domain_hi[3] = {std::numeric_limits<int32_t>::min(),
                                  std::numeric_limits<int32_t>::min(),
                                  std::numeric_limits<int32_t>::min()};
          if (step_meta.levels.empty()) {
            return hpx::make_ready_future();
          }
          if (!step_meta.levels.empty()) {
            for (const auto &b : step_meta.levels.front().boxes) {
              domain_lo[0] = std::min(domain_lo[0], b.lo.x);
              domain_lo[1] = std::min(domain_lo[1], b.lo.y);
              domain_lo[2] = std::min(domain_lo[2], b.lo.z);
              domain_hi[0] = std::max(domain_hi[0], b.hi.x);
              domain_hi[1] = std::max(domain_hi[1], b.hi.y);
              domain_hi[2] = std::max(domain_hi[2], b.hi.z);
            }
          }
          if (domain_hi[0] < domain_lo[0] || domain_hi[1] < domain_lo[1] ||
              domain_hi[2] < domain_lo[2]) {
            return hpx::make_ready_future();
          }
          bool is_periodic[3] = {level.geom.is_periodic[0],
                                 level.geom.is_periodic[1],
                                 level.geom.is_periodic[2]};
          double domain_lo_edge[3] = {
              cell_edge(step_meta.levels.front().geom, 0, domain_lo[0]),
              cell_edge(step_meta.levels.front().geom, 1, domain_lo[1]),
              cell_edge(step_meta.levels.front().geom, 2, domain_lo[2]),
          };
          double domain_hi_edge[3] = {
              cell_edge(step_meta.levels.front().geom, 0, domain_hi[0] + 1),
              cell_edge(step_meta.levels.front().geom, 1, domain_hi[1] + 1),
              cell_edge(step_meta.levels.front().geom, 2, domain_hi[2] + 1),
          };

          const auto self_input = make_real_grid_accessor(inputs[0]);

          for (int i = 0; i < nx; ++i) {
            const int32_t gi = box.lo.x + i;
            const double xc = cell_center(target_geom, 0, gi);
            for (int j = 0; j < ny; ++j) {
              const int32_t gj = box.lo.y + j;
              const double yc = cell_center(target_geom, 1, gj);
              for (int k = 0; k < nz; ++k) {
                const int32_t gk = box.lo.z + k;
                const double zc = cell_center(target_geom, 2, gk);
                const double f0 = self_input(i, j, k);

                std::vector<SamplePoint> samples;
                const int32_t width = 2 * stencil_radius + 1;
                samples.reserve(
                    static_cast<std::size_t>(width * width * width - 1));
                for (int ox = -stencil_radius; ox <= stencil_radius; ++ox) {
                  for (int oy = -stencil_radius; oy <= stencil_radius; ++oy) {
                    for (int oz = -stencil_radius; oz <= stencil_radius; ++oz) {
                      if (ox == 0 && oy == 0 && oz == 0) {
                        continue;
                      }
                      const double xp =
                          xc + static_cast<double>(ox) * target_geom.dx[0];
                      const double yp =
                          yc + static_cast<double>(oy) * target_geom.dx[1];
                      const double zp =
                          zc + static_cast<double>(oz) * target_geom.dx[2];
                      auto s = composite_sample_at(
                          patches,
                          static_cast<int>(step_meta.levels.size()) - 1,
                          domain_lo, domain_hi, is_periodic, domain_lo_edge,
                          domain_hi_edge, xp, yp, zp);
                      if (!s.has_value()) {
                        continue;
                      }
                      bool duplicate = false;
                      for (const auto &existing : samples) {
                        if (std::abs(existing.x - s->x) < 1e-14 &&
                            std::abs(existing.y - s->y) < 1e-14 &&
                            std::abs(existing.z - s->z) < 1e-14) {
                          duplicate = true;
                          break;
                        }
                      }
                      if (!duplicate) {
                        samples.push_back(*s);
                      }
                    }
                  }
                }

                double grad[3] = {0.0, 0.0, 0.0};
                if (samples.size() >= 3) {
                  double a[3][3] = {
                      {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
                  double bvec[3] = {0.0, 0.0, 0.0};
                  for (const auto &s : samples) {
                    const double rx = s.x - xc;
                    const double ry = s.y - yc;
                    const double rz = s.z - zc;
                    const double df = s.value - f0;
                    const double r2 = rx * rx + ry * ry + rz * rz;
                    if (r2 <= 0.0) {
                      continue;
                    }
                    const double w = 1.0 / (r2 + 1e-30);
                    a[0][0] += w * rx * rx;
                    a[0][1] += w * rx * ry;
                    a[0][2] += w * rx * rz;
                    a[1][0] += w * ry * rx;
                    a[1][1] += w * ry * ry;
                    a[1][2] += w * ry * rz;
                    a[2][0] += w * rz * rx;
                    a[2][1] += w * rz * ry;
                    a[2][2] += w * rz * rz;
                    bvec[0] += w * rx * df;
                    bvec[1] += w * ry * df;
                    bvec[2] += w * rz * df;
                  }
                  solve_3x3(a, bvec, grad);
                }

                out(i, j, k, 0) = grad[0];
                out(i, j, k, 1) = grad[1];
                out(i, j, k, 2) = grad[2];
              }
            }
          }
          return hpx::make_ready_future();
        });
  }
  {
    using Params = PlotfileLoadParams;

    /**
     * @brief Loads one plotfile field component for a grid block.
     * @par Chunk inputs None.
     * @par MessagePack parameters `plotfile` is the source path, `level`
     * selects the AMR level, and `comp` selects the field component.
     * @par Chunk outputs `outputs[0]` is the selected component on the block
     * grid.
     */
    registry.register_kernel(
        KernelDesc{.name = "plotfile_load",
                   .n_inputs = 0,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &level, int32_t block, std::span<const ChunkBuffer>,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          if (outputs.empty()) {
            return hpx::make_ready_future();
          }

          const auto &params =
              require_kernel_params<Params>(kernel_params, "plotfile_load");

          if (params.plotfile.empty()) {
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

          auto reader = get_plotfile_reader(params.plotfile);
          if (!reader || params.level < 0 ||
              params.level >= reader->num_levels()) {
            return hpx::make_ready_future();
          }
          if (block >= reader->num_fabs(params.level)) {
            return hpx::make_ready_future();
          }

          auto data = reader->read_fab(params.level, block, params.comp, 1);
          if (data.ncomp < 1 || data.nx != nx || data.ny != ny ||
              data.nz != nz) {
            return hpx::make_ready_future();
          }

          const ScalarType file_scalar =
              data.type == plotfile::RealType::kFloat32 ? ScalarType::kF32
                                                        : ScalarType::kF64;
          auto source = ChunkBuffer::wrap(
              SharedByteBuffer(std::move(data.bytes)),
              BufferDesc::plotfile_grid(file_scalar,
                                        {static_cast<std::uint64_t>(nx),
                                         static_cast<std::uint64_t>(ny),
                                         static_cast<std::uint64_t>(nz)}));
          if (plotfile_zero_copy_reads_enabled() &&
              outputs[0].desc().scalar == file_scalar) {
            outputs[0] = std::move(source);
            return hpx::make_ready_future();
          }
          outputs[0].copy_from(source);

          return hpx::make_ready_future();
        });
  }
}

} // namespace kangaroo
