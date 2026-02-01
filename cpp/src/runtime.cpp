#include "kangaroo/runtime.hpp"

#include "kangaroo/plan_decode.hpp"
#include "kangaroo/plotfile_reader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <msgpack.hpp>

#include <hpx/include/actions.hpp>
#include <hpx/collectives/broadcast_direct.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime.hpp>

namespace kangaroo {

namespace {

std::mutex g_ctx_mutex;
std::once_flag g_hpx_start_once;
thread_local bool g_hpx_thread_registered = false;
bool g_hpx_started = false;
bool g_hpx_cfg_set = false;
bool g_hpx_cmdline_set = false;
std::vector<std::string> g_hpx_cfg;
std::vector<std::string> g_hpx_cmdline;
RunMeta g_runmeta;
bool g_has_runmeta = false;
DatasetHandle g_dataset;
bool g_has_dataset = false;
KernelRegistry* g_kernel_registry = nullptr;
std::unordered_map<int32_t, PlanIR> g_plans;

std::shared_ptr<plotfile::PlotfileReader> get_plotfile_reader(const std::string& path) {
  static std::mutex reader_mutex;
  static std::unordered_map<std::string, std::weak_ptr<plotfile::PlotfileReader>> readers;

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

template <typename InT, typename OutT>
void transpose_plotfile_axes(const InT* in, OutT* out, int nx, int ny, int nz) {
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::size_t in_idx = (static_cast<std::size_t>(k) * ny + j) * nx + i;
        const std::size_t out_idx = (static_cast<std::size_t>(i) * ny + j) * nz + k;
        out[out_idx] = static_cast<OutT>(in[in_idx]);
      }
    }
  }
}

void register_default_kernels(KernelRegistry& registry) {
  static const bool log_locality = []() {
    const char* env = std::getenv("KANGAROO_LOG_LOCALITY");
    return env != nullptr && *env != '\0' && *env != '0';
  }();
  registry.register_kernel(
      KernelDesc{.name = "gradU_stencil", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = true},
      [](const LevelMeta&, int32_t, std::span<const HostView>, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (!outputs.empty()) {
          outputs[0].data.assign(1, 0);
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "plotfile_load", .n_inputs = 0, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta& level, int32_t block, std::span<const HostView>, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t> params_msgpack) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }

        struct Params {
          std::string plotfile;
          int level = 0;
          int comp = 0;
          int bytes_per_value = 4;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            auto get_key = [&](const char* key) -> const msgpack::object* {
              for (uint32_t i = 0; i < root.via.map.size; ++i) {
                const auto& k = root.via.map.ptr[i].key;
                if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
                  return &root.via.map.ptr[i].val;
                }
              }
              return nullptr;
            };
            if (const auto* path = get_key("plotfile"); path && path->type == msgpack::type::STR) {
              params.plotfile = path->as<std::string>();
            }
            if (const auto* lvl = get_key("level"); lvl &&
                                                 (lvl->type == msgpack::type::POSITIVE_INTEGER ||
                                                  lvl->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.level = lvl->as<int>();
            }
            if (const auto* comp = get_key("comp"); comp &&
                                                 (comp->type == msgpack::type::POSITIVE_INTEGER ||
                                                  comp->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.comp = comp->as<int>();
            }
            if (const auto* bpv = get_key("bytes_per_value"); bpv &&
                                                         (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                                                          bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.bytes_per_value = bpv->as<int>();
            }
          }
        }

        if (params.plotfile.empty()) {
          return hpx::make_ready_future();
        }

        if (block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }

        const auto& box = level.boxes.at(static_cast<std::size_t>(block));
        const int nx = box.hi.x - box.lo.x + 1;
        const int ny = box.hi.y - box.lo.y + 1;
        const int nz = box.hi.z - box.lo.z + 1;
        if (nx <= 0 || ny <= 0 || nz <= 0) {
          return hpx::make_ready_future();
        }

        std::size_t out_bytes = 0;
        if (params.bytes_per_value > 0) {
          out_bytes = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                      static_cast<std::size_t>(nz) *
                      static_cast<std::size_t>(params.bytes_per_value);
        }
        if (out_bytes == 0) {
          return hpx::make_ready_future();
        }

        if (outputs[0].data.size() != out_bytes) {
          outputs[0].data.assign(out_bytes, 0);
        }

        auto reader = get_plotfile_reader(params.plotfile);
        if (!reader || params.level < 0 || params.level >= reader->num_levels()) {
          return hpx::make_ready_future();
        }
        if (block >= reader->num_fabs(params.level)) {
          return hpx::make_ready_future();
        }

        auto data = reader->read_fab(params.level, block, params.comp, 1);
        if (data.ncomp < 1 || data.nx != nx || data.ny != ny || data.nz != nz) {
          return hpx::make_ready_future();
        }

        if (data.type == plotfile::RealType::kFloat32) {
          const auto* in = reinterpret_cast<const float*>(data.bytes.data());
          if (params.bytes_per_value == 4) {
            auto* out = reinterpret_cast<float*>(outputs[0].data.data());
            transpose_plotfile_axes(in, out, nx, ny, nz);
          } else if (params.bytes_per_value == 8) {
            auto* out = reinterpret_cast<double*>(outputs[0].data.data());
            transpose_plotfile_axes(in, out, nx, ny, nz);
          }
        } else if (data.type == plotfile::RealType::kFloat64) {
          const auto* in = reinterpret_cast<const double*>(data.bytes.data());
          if (params.bytes_per_value == 8) {
            auto* out = reinterpret_cast<double*>(outputs[0].data.data());
            transpose_plotfile_axes(in, out, nx, ny, nz);
          } else if (params.bytes_per_value == 4) {
            auto* out = reinterpret_cast<float*>(outputs[0].data.data());
            transpose_plotfile_axes(in, out, nx, ny, nz);
          }
        }

        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "vorticity_mag", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView>, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (!outputs.empty()) {
          outputs[0].data.assign(1, 0);
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "uniform_slice", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
         const NeighborViews&, std::span<HostView> outputs,
         std::span<const std::uint8_t> params_msgpack) {
        if (log_locality) {
          std::cout << "[kangaroo] uniform_slice block=" << block
                    << " locality=" << hpx::get_locality_id() << std::endl;
        }
        struct Params {
          int axis = 2;
          double coord = 0.0;
          std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
          std::array<int, 2> resolution{1, 1};
          int bytes_per_value = 4;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            auto get_key = [&](const char* key) -> const msgpack::object* {
              for (uint32_t i = 0; i < root.via.map.size; ++i) {
                const auto& k = root.via.map.ptr[i].key;
                if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
                  return &root.via.map.ptr[i].val;
                }
              }
              return nullptr;
            };
            if (const auto* axis = get_key("axis"); axis &&
                                                   (axis->type == msgpack::type::POSITIVE_INTEGER ||
                                                    axis->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.axis = axis->as<int>();
            }
            if (const auto* coord = get_key("coord"); coord &&
                                                     (coord->type == msgpack::type::FLOAT ||
                                                      coord->type == msgpack::type::POSITIVE_INTEGER ||
                                                      coord->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.coord = coord->as<double>();
            }
            if (const auto* rect = get_key("rect"); rect && rect->type == msgpack::type::ARRAY &&
                                                   rect->via.array.size == 4) {
              for (uint32_t i = 0; i < 4; ++i) {
                params.rect[i] = rect->via.array.ptr[i].as<double>();
              }
            }
            if (const auto* res = get_key("resolution"); res && res->type == msgpack::type::ARRAY &&
                                                       res->via.array.size == 2) {
              params.resolution[0] = res->via.array.ptr[0].as<int>();
              params.resolution[1] = res->via.array.ptr[1].as<int>();
            }
            if (const auto* bpv = get_key("bytes_per_value"); bpv &&
                                                         (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                                                          bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.bytes_per_value = bpv->as<int>();
            }
          }
        }

        const auto out_nx = params.resolution[0];
        const auto out_ny = params.resolution[1];
        std::size_t bytes = 0;
        if (out_nx > 0 && out_ny > 0 && params.bytes_per_value > 0) {
          bytes = static_cast<std::size_t>(out_nx) * static_cast<std::size_t>(out_ny) *
                  static_cast<std::size_t>(params.bytes_per_value);
        }

        if (outputs.empty() || inputs.empty() || bytes == 0) {
          return hpx::make_ready_future();
        }

        if (outputs[0].data.size() != bytes) {
          outputs[0].data.assign(bytes, 0);
        } else {
          std::fill(outputs[0].data.begin(), outputs[0].data.end(), 0);
        }

        if (block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }

        const auto& box = level.boxes.at(static_cast<std::size_t>(block));
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
                                       : (axis == 1 ? k_global - box.lo.y : k_global - box.lo.z));
        if ((axis == 0 && (k_local < 0 || k_local >= nx)) ||
            (axis == 1 && (k_local < 0 || k_local >= ny)) ||
            (axis == 2 && (k_local < 0 || k_local >= nz))) {
          return hpx::make_ready_future();
        }

        const double u0 = params.rect[0];
        const double v0 = params.rect[1];
        const double u1 = params.rect[2];
        const double v1 = params.rect[3];
        const double du = (out_nx > 0) ? (u1 - u0) / static_cast<double>(out_nx) : 0.0;
        const double dv = (out_ny > 0) ? (v1 - v0) / static_cast<double>(out_ny) : 0.0;

        auto in_index = [&](int i, int j, int k) -> std::size_t {
          return static_cast<std::size_t>((i * ny + j) * nz + k);
        };

        const auto& in = inputs[0].data;
        if (params.bytes_per_value == 4) {
          const auto* in_f = reinterpret_cast<const float*>(in.data());
          auto* out_f = reinterpret_cast<float*>(outputs[0].data.data());
          for (int j = 0; j < out_ny; ++j) {
            const double v = v0 + (static_cast<double>(j) + 0.5) * dv;
            const int v_global = cell_index(v_axis, v);
            const int v_local = v_axis == 0 ? v_global - box.lo.x
                                            : (v_axis == 1 ? v_global - box.lo.y
                                                           : v_global - box.lo.z);
            for (int i = 0; i < out_nx; ++i) {
              const double u = u0 + (static_cast<double>(i) + 0.5) * du;
              const int u_global = cell_index(u_axis, u);
              const int u_local = u_axis == 0 ? u_global - box.lo.x
                                              : (u_axis == 1 ? u_global - box.lo.y
                                                             : u_global - box.lo.z);
              float value = 0.0f;
              if (u_local >= 0 && v_local >= 0) {
                if ((u_axis == 0 && u_local < nx) || (u_axis == 1 && u_local < ny) ||
                    (u_axis == 2 && u_local < nz)) {
                  if ((v_axis == 0 && v_local < nx) || (v_axis == 1 && v_local < ny) ||
                      (v_axis == 2 && v_local < nz)) {
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
                    const auto idx = in_index(ii, jj, kk);
                    if (idx * sizeof(float) < in.size()) {
                      value = in_f[idx];
                    }
                  }
                }
              }
              out_f[static_cast<std::size_t>(j) * out_nx + i] = value;
            }
          }
        } else if (params.bytes_per_value == 8) {
          const auto* in_d = reinterpret_cast<const double*>(in.data());
          auto* out_d = reinterpret_cast<double*>(outputs[0].data.data());
          for (int j = 0; j < out_ny; ++j) {
            const double v = v0 + (static_cast<double>(j) + 0.5) * dv;
            const int v_global = cell_index(v_axis, v);
            const int v_local = v_axis == 0 ? v_global - box.lo.x
                                            : (v_axis == 1 ? v_global - box.lo.y
                                                           : v_global - box.lo.z);
            for (int i = 0; i < out_nx; ++i) {
              const double u = u0 + (static_cast<double>(i) + 0.5) * du;
              const int u_global = cell_index(u_axis, u);
              const int u_local = u_axis == 0 ? u_global - box.lo.x
                                              : (u_axis == 1 ? u_global - box.lo.y
                                                             : u_global - box.lo.z);
              double value = 0.0;
              if (u_local >= 0 && v_local >= 0) {
                if ((u_axis == 0 && u_local < nx) || (u_axis == 1 && u_local < ny) ||
                    (u_axis == 2 && u_local < nz)) {
                  if ((v_axis == 0 && v_local < nx) || (v_axis == 1 && v_local < ny) ||
                      (v_axis == 2 && v_local < nz)) {
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
                    const auto idx = in_index(ii, jj, kk);
                    if (idx * sizeof(double) < in.size()) {
                      value = in_d[idx];
                    }
                  }
                }
              }
              out_d[static_cast<std::size_t>(j) * out_nx + i] = value;
            }
          }
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "uniform_slice_reduce", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t> params_msgpack) {
        struct Params {
          int bytes_per_value = 4;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            for (uint32_t i = 0; i < root.via.map.size; ++i) {
              const auto& k = root.via.map.ptr[i].key;
              if (k.type == msgpack::type::STR && k.as<std::string>() == "bytes_per_value") {
                const auto& v = root.via.map.ptr[i].val;
                if (v.type == msgpack::type::POSITIVE_INTEGER ||
                    v.type == msgpack::type::NEGATIVE_INTEGER) {
                  params.bytes_per_value = v.as<int>();
                }
              }
            }
          }
        }

        if (outputs.empty() || inputs.empty()) {
          return hpx::make_ready_future();
        }

        auto& out = outputs[0].data;
        if (out.empty()) {
          return hpx::make_ready_future();
        }
        std::fill(out.begin(), out.end(), 0);

        if (params.bytes_per_value == 4) {
          const std::size_t n = out.size() / sizeof(float);
          auto* out_f = reinterpret_cast<float*>(out.data());

          for (const auto& in_view : inputs) {
            const auto& in = in_view.data;
            if (in.empty()) {
              continue;
            }
            const std::size_t n_in = std::min(n, in.size() / sizeof(float));
            const auto* in_f = reinterpret_cast<const float*>(in.data());
            for (std::size_t i = 0; i < n_in; ++i) {
              out_f[i] += in_f[i];
            }
          }
        } else if (params.bytes_per_value == 8) {
          const std::size_t n = out.size() / sizeof(double);
          auto* out_d = reinterpret_cast<double*>(out.data());

          for (const auto& in_view : inputs) {
            const auto& in = in_view.data;
            if (in.empty()) {
              continue;
            }
            const std::size_t n_in = std::min(n, in.size() / sizeof(double));
            const auto* in_d = reinterpret_cast<const double*>(in.data());
            for (std::size_t i = 0; i < n_in; ++i) {
              out_d[i] += in_d[i];
            }
          }
        }
        return hpx::make_ready_future();
      });
}

void set_runmeta_impl(const RunMeta& meta) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_runmeta = meta;
  g_has_runmeta = true;
}

void set_dataset_impl(const DatasetHandle& dataset) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_dataset = dataset;
  g_has_dataset = true;
}

void ensure_hpx_started() {
  std::call_once(g_hpx_start_once, []() {
    std::vector<std::string> argv_storage;
    if (g_hpx_cmdline_set && !g_hpx_cmdline.empty()) {
      argv_storage = g_hpx_cmdline;
    } else {
      argv_storage.emplace_back("kangaroo");
    }
    std::vector<char*> argv;
    argv.reserve(argv_storage.size());
    for (auto& arg : argv_storage) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    int argc = static_cast<int>(argv.size());
    hpx::init_params params;
    if (g_hpx_cfg_set && !g_hpx_cfg.empty()) {
      params.cfg = g_hpx_cfg;
    } else {
      params.cfg = {"hpx.os_threads=1"};
    }
    hpx::start(nullptr, argc, argv.data(), params);
    g_hpx_started = true;
  });
  if (auto* rt = hpx::get_runtime_ptr(); rt != nullptr) {
    if (!g_hpx_thread_registered) {
      try {
        g_hpx_thread_registered = hpx::register_thread(rt, "kangaroo_python");
      } catch (...) {
        g_hpx_thread_registered = true;
      }
    }
  }
}

}  // namespace

void set_global_runmeta(const RunMeta& meta) {
  set_runmeta_impl(meta);
}

const RunMeta& global_runmeta() {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  if (!g_has_runmeta) {
    throw std::runtime_error("global RunMeta not initialized");
  }
  return g_runmeta;
}

void set_global_dataset(const DatasetHandle& dataset) {
  set_dataset_impl(dataset);
}

const DatasetHandle& global_dataset() {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  if (!g_has_dataset) {
    throw std::runtime_error("global DatasetHandle not initialized");
  }
  return g_dataset;
}

void set_global_kernel_registry(KernelRegistry* registry) {
  g_kernel_registry = registry;
}

KernelRegistry& global_kernels() {
  if (!g_kernel_registry) {
    throw std::runtime_error("global KernelRegistry not initialized");
  }
  return *g_kernel_registry;
}

void set_global_plan(int32_t plan_id, const PlanIR& plan) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_plans[plan_id] = plan;
}

const PlanIR& global_plan(int32_t plan_id) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  auto it = g_plans.find(plan_id);
  if (it == g_plans.end()) {
    throw std::runtime_error("global PlanIR not initialized for plan id");
  }
  return it->second;
}

void erase_global_plan(int32_t plan_id) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_plans.erase(plan_id);
}

void set_runmeta_action(const RunMeta& meta) {
  set_runmeta_impl(meta);
}

void set_dataset_action(const DatasetHandle& dataset) {
  set_dataset_impl(dataset);
  DataServiceLocal::set_dataset(&g_dataset);
}

void set_plan_action(int32_t plan_id, const PlanIR& plan) {
  set_global_plan(plan_id, plan);
}

void erase_plan_action(int32_t plan_id) {
  erase_global_plan(plan_id);
}

void preload_action(const RunMeta& meta,
                    const DatasetHandle& dataset,
                    const std::vector<int32_t>& fields) {
  DataServiceLocal::preload(meta, dataset, fields);
}

}  // namespace kangaroo

HPX_PLAIN_ACTION(kangaroo::set_runmeta_action, kangaroo_set_runmeta_action)
HPX_PLAIN_ACTION(kangaroo::set_dataset_action, kangaroo_set_dataset_action)
HPX_PLAIN_ACTION(kangaroo::set_plan_action, kangaroo_set_plan_action)
HPX_PLAIN_ACTION(kangaroo::erase_plan_action, kangaroo_erase_plan_action)
HPX_PLAIN_ACTION(kangaroo::preload_action, kangaroo_preload_action)

namespace kangaroo {

Runtime::Runtime() {
  set_global_kernel_registry(&kernel_registry_);
  register_default_kernels(kernel_registry_);
}

Runtime::Runtime(const std::vector<std::string>& hpx_config,
                 const std::vector<std::string>& hpx_cmdline) {
  if (g_hpx_started) {
    throw std::runtime_error("HPX already started; cannot change config/args");
  }
  if (g_hpx_cfg_set || g_hpx_cmdline_set) {
    throw std::runtime_error("HPX config/args already set; use the default constructor");
  }
  if (!hpx_config.empty()) {
    g_hpx_cfg = hpx_config;
    g_hpx_cfg_set = true;
  }
  if (!hpx_cmdline.empty()) {
    g_hpx_cmdline = hpx_cmdline;
    g_hpx_cmdline_set = true;
  }
  set_global_kernel_registry(&kernel_registry_);
  register_default_kernels(kernel_registry_);
}

void DatasetHandle::set_chunk(const ChunkRef& ref, HostView view) {
  data[ref] = std::move(view);
}

std::optional<HostView> DatasetHandle::get_chunk(const ChunkRef& ref) const {
  auto it = data.find(ref);
  if (it == data.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool DatasetHandle::has_chunk(const ChunkRef& ref) const {
  return data.find(ref) != data.end();
}

int32_t Runtime::alloc_field_id(const std::string&) {
  return next_field_id_++;
}

void Runtime::mark_field_persistent(int32_t fid, const std::string& name) {
  persistent_fields_[fid] = name;
}

KernelRegistry& Runtime::kernels() {
  return kernel_registry_;
}

void Runtime::run_packed_plan(const std::vector<std::uint8_t>& packed,
                              const RunMetaHandle& runmeta,
                              const DatasetHandle& dataset) {
  ensure_hpx_started();
  PlanIR plan = decode_plan_msgpack(std::span<const std::uint8_t>(packed.data(), packed.size()));

  auto localities = hpx::find_all_localities();
  hpx::lcos::broadcast<::kangaroo_set_runmeta_action>(localities, runmeta.meta).get();
  hpx::lcos::broadcast<::kangaroo_set_dataset_action>(localities, dataset).get();
  int32_t plan_id = next_plan_id_++;
  hpx::lcos::broadcast<::kangaroo_set_plan_action>(localities, plan_id, plan).get();

  DataServiceLocal data;
  AdjacencyServiceLocal adjacency(runmeta.meta);
  Executor executor(plan_id, runmeta.meta, data, adjacency, kernel_registry_);

  executor.run(plan).get();
  hpx::lcos::broadcast<::kangaroo_erase_plan_action>(localities, plan_id).get();
}

void Runtime::preload_dataset(const RunMetaHandle& runmeta,
                              const DatasetHandle& dataset,
                              const std::vector<int32_t>& fields) {
  ensure_hpx_started();
  auto localities = hpx::find_all_localities();
  hpx::lcos::broadcast<::kangaroo_set_runmeta_action>(localities, runmeta.meta).get();
  hpx::lcos::broadcast<::kangaroo_set_dataset_action>(localities, dataset).get();
  hpx::lcos::broadcast<::kangaroo_preload_action>(localities, runmeta.meta, dataset, fields).get();
}

HostView Runtime::get_task_chunk(int32_t step,
                                 int16_t level,
                                 int32_t field,
                                 int32_t version,
                                 int32_t block) {
  ensure_hpx_started();
  DataServiceLocal data;
  ChunkRef ref{step, level, field, version, block};
  return data.get_host(ref).get();
}

}  // namespace kangaroo
