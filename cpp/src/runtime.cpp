#include "kangaroo/runtime.hpp"

#include "kangaroo/plan_decode.hpp"

#include <hpx/include/lcos.hpp>

namespace kangaroo {

Runtime::Runtime() = default;

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
  PlanIR plan = decode_plan_msgpack(std::span<const std::uint8_t>(packed.data(), packed.size()));

  DataServiceLocal data;
  AdjacencyServiceLocal adjacency(runmeta.meta);
  Executor executor(runmeta.meta, data, adjacency, kernel_registry_);

  executor.run(plan).get();
}

}  // namespace kangaroo
