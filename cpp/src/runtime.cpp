#include "kangaroo/runtime.hpp"

#include "kangaroo/plan_decode.hpp"

#include <mutex>
#include <stdexcept>
#include <unordered_map>

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
KernelRegistry* g_kernel_registry = nullptr;
std::unordered_map<int32_t, PlanIR> g_plans;

void set_runmeta_impl(const RunMeta& meta) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_runmeta = meta;
  g_has_runmeta = true;
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

void set_plan_action(int32_t plan_id, const PlanIR& plan) {
  set_global_plan(plan_id, plan);
}

void erase_plan_action(int32_t plan_id) {
  erase_global_plan(plan_id);
}

}  // namespace kangaroo

HPX_PLAIN_ACTION(kangaroo::set_runmeta_action, kangaroo_set_runmeta_action)
HPX_PLAIN_ACTION(kangaroo::set_plan_action, kangaroo_set_plan_action)
HPX_PLAIN_ACTION(kangaroo::erase_plan_action, kangaroo_erase_plan_action)

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
                              const DatasetHandle&) {
  ensure_hpx_started();
  PlanIR plan = decode_plan_msgpack(std::span<const std::uint8_t>(packed.data(), packed.size()));

  auto localities = hpx::find_all_localities();
  hpx::lcos::broadcast<::kangaroo_set_runmeta_action>(localities, runmeta.meta).get();
  int32_t plan_id = next_plan_id_++;
  hpx::lcos::broadcast<::kangaroo_set_plan_action>(localities, plan_id, plan).get();

  DataServiceLocal data;
  AdjacencyServiceLocal adjacency(runmeta.meta);
  Executor executor(plan_id, runmeta.meta, data, adjacency, kernel_registry_);

  executor.run(plan).get();
  hpx::lcos::broadcast<::kangaroo_erase_plan_action>(localities, plan_id).get();
}

}  // namespace kangaroo
