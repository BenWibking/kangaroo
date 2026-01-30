#include "kangaroo/runtime.hpp"

#include "kangaroo/plan_decode.hpp"

#include <array>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <msgpack.hpp>

#include <hpx/include/actions.hpp>
#include <hpx/collectives/broadcast_direct.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>
#include <hpx/runtime.hpp>

namespace kangaroo {

namespace {

std::mutex g_ctx_mutex;
std::once_flag g_hpx_start_once;
thread_local bool g_hpx_thread_registered = false;
RunMeta g_runmeta;
bool g_has_runmeta = false;
DatasetHandle g_dataset;
bool g_has_dataset = false;
KernelRegistry* g_kernel_registry = nullptr;
std::unordered_map<int32_t, PlanIR> g_plans;

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
    int argc = 1;
    char arg0[] = "kangaroo";
    char* argv[] = {arg0};
    hpx::init_params params;
    params.cfg = {"hpx.os_threads=1"};
    hpx::start(nullptr, argc, argv, params);
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
  kernel_registry_.register_kernel(
      KernelDesc{.name = "gradU_stencil", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = true},
      [](const LevelMeta&, int32_t, std::span<const HostView>, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (!outputs.empty()) {
          outputs[0].data.assign(1, 0);
        }
        return hpx::make_ready_future();
      });
  kernel_registry_.register_kernel(
      KernelDesc{.name = "vorticity_mag", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView>, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (!outputs.empty()) {
          outputs[0].data.assign(1, 0);
        }
        return hpx::make_ready_future();
      });
  kernel_registry_.register_kernel(
      KernelDesc{.name = "uniform_slice", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView>, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t> params_msgpack) {
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

        const auto nx = params.resolution[0];
        const auto ny = params.resolution[1];
        std::size_t bytes = 0;
        if (nx > 0 && ny > 0 && params.bytes_per_value > 0) {
          bytes = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                  static_cast<std::size_t>(params.bytes_per_value);
        }

        if (!outputs.empty()) {
          if (bytes > 0 && outputs[0].data.size() != bytes) {
            outputs[0].data.assign(bytes, 0);
          } else if (outputs[0].data.empty()) {
            outputs[0].data.assign(1, 0);
          } else {
            std::fill(outputs[0].data.begin(), outputs[0].data.end(), 0);
          }
        }
        return hpx::make_ready_future();
      });
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

}  // namespace kangaroo
