#pragma once

#include "kangaroo/adjacency.hpp"
#include "kangaroo/data_service_local.hpp"
#include "kangaroo/executor.hpp"
#include "kangaroo/kernel_registry.hpp"
#include "kangaroo/runmeta.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace kangaroo {

struct RunMetaHandle {
  RunMeta meta;
};

struct DatasetHandle {
  std::string uri;
  int32_t step = 0;
  int16_t level = 0;
};

class Runtime {
 public:
  Runtime();

  int32_t alloc_field_id(const std::string& name);
  void mark_field_persistent(int32_t fid, const std::string& name);

  KernelRegistry& kernels();

  void run_packed_plan(const std::vector<std::uint8_t>& packed,
                       const RunMetaHandle& runmeta,
                       const DatasetHandle& dataset);

 private:
  int32_t next_field_id_ = 1000;
  int32_t next_plan_id_ = 1;
  std::unordered_map<int32_t, std::string> persistent_fields_;

  KernelRegistry kernel_registry_;
};

void set_global_runmeta(const RunMeta& meta);
const RunMeta& global_runmeta();
void set_global_kernel_registry(KernelRegistry* registry);
KernelRegistry& global_kernels();
void set_global_plan(int32_t plan_id, const PlanIR& plan);
const PlanIR& global_plan(int32_t plan_id);
void erase_global_plan(int32_t plan_id);

}  // namespace kangaroo
