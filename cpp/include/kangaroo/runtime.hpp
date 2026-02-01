#pragma once

#include "kangaroo/adjacency.hpp"
#include "kangaroo/data_service.hpp"
#include "kangaroo/data_service_local.hpp"
#include "kangaroo/executor.hpp"
#include "kangaroo/kernel_registry.hpp"
#include "kangaroo/runmeta.hpp"

#include <cstdint>
#include <optional>
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
  std::unordered_map<ChunkRef, HostView, ChunkRefHash, ChunkRefEq> data;

  void set_chunk(const ChunkRef& ref, HostView view);
  std::optional<HostView> get_chunk(const ChunkRef& ref) const;
  bool has_chunk(const ChunkRef& ref) const;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& uri& step& level& data;
  }
};

class Runtime {
 public:
  Runtime();
  Runtime(const std::vector<std::string>& hpx_config,
          const std::vector<std::string>& hpx_cmdline);

  int32_t alloc_field_id(const std::string& name);
  void mark_field_persistent(int32_t fid, const std::string& name);

  KernelRegistry& kernels();

  void run_packed_plan(const std::vector<std::uint8_t>& packed,
                       const RunMetaHandle& runmeta,
                       const DatasetHandle& dataset);

  void preload_dataset(const RunMetaHandle& runmeta,
                       const DatasetHandle& dataset,
                       const std::vector<int32_t>& fields);

  HostView get_task_chunk(int32_t step,
                          int16_t level,
                          int32_t field,
                          int32_t version,
                          int32_t block);

  void set_event_log_path(const std::string& path);

 private:
  int32_t next_field_id_ = 1000;
  int32_t next_plan_id_ = 1;
  std::unordered_map<int32_t, std::string> persistent_fields_;

  KernelRegistry kernel_registry_;
};

struct TaskEvent {
  std::string id;
  std::string name;
  std::string kernel;
  std::string plane;
  std::string status;
  int32_t stage = -1;
  int32_t template_index = -1;
  int32_t block = -1;
  int32_t step = 0;
  int32_t level = 0;
  int32_t locality = -1;
  int32_t worker = -1;
  std::string worker_label;
  double ts = 0.0;
  double start = 0.0;
  double end = 0.0;
};

void set_event_log_path(const std::string& path);
bool has_event_log();
void log_task_event(const TaskEvent& event);

void set_global_runmeta(const RunMeta& meta);
const RunMeta& global_runmeta();
void set_global_dataset(const DatasetHandle& dataset);
const DatasetHandle& global_dataset();
void set_global_kernel_registry(KernelRegistry* registry);
KernelRegistry& global_kernels();
void set_global_plan(int32_t plan_id, const PlanIR& plan);
const PlanIR& global_plan(int32_t plan_id);
void erase_global_plan(int32_t plan_id);

}  // namespace kangaroo
