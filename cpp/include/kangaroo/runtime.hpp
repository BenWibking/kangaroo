#pragma once

#include "kangaroo/adjacency.hpp"
#include "kangaroo/backend_memory.hpp"
#include "kangaroo/backend_openpmd.hpp"
#include "kangaroo/backend_parthenon.hpp"
#include "kangaroo/backend_plotfile.hpp"
#include "kangaroo/data_service.hpp"
#include "kangaroo/data_service_local.hpp"
#include "kangaroo/dataset_backend.hpp"
#include "kangaroo/executor.hpp"
#include "kangaroo/kernel_registry.hpp"
#include "kangaroo/runmeta.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kangaroo {

struct RunMetaHandle {
  RunMeta meta;
};

struct DatasetHandle {
  std::string uri;
  int32_t step = 0;
  int16_t level = 0;

  std::shared_ptr<DatasetBackend> backend;

  void set_chunk(const ChunkRef& ref, HostView view);
  std::optional<HostView> get_chunk(const ChunkRef& ref) const;
  bool has_chunk(const ChunkRef& ref) const;

  template <typename Archive>
  void save(Archive& ar, unsigned) const {
    ar& uri& step& level;

    bool is_memory = false;
    bool is_plotfile = false;
    bool is_openpmd = false;
    bool is_parthenon = false;

    if (backend) {
      if (auto mem = std::dynamic_pointer_cast<MemoryBackend>(backend)) {
        is_memory = true;
        ar& is_memory& is_plotfile& is_openpmd& is_parthenon;
        ar& mem->data();
      } else if (auto plt = std::dynamic_pointer_cast<PlotfileBackend>(backend)) {
        is_plotfile = true;
        ar& is_memory& is_plotfile& is_openpmd& is_parthenon;
#ifdef KANGAROO_USE_OPENPMD
      } else if (auto opmd = std::dynamic_pointer_cast<OpenPMDBackend>(backend)) {
        is_openpmd = true;
        ar& is_memory& is_plotfile& is_openpmd& is_parthenon;
#endif
#ifdef KANGAROO_USE_PARTHENON_HDF5
      } else if (auto phdf = std::dynamic_pointer_cast<ParthenonBackend>(backend)) {
        (void)phdf;
        is_parthenon = true;
        ar& is_memory& is_plotfile& is_openpmd& is_parthenon;
#endif
      } else {
        ar& is_memory& is_plotfile& is_openpmd& is_parthenon;
      }
    } else {
      ar& is_memory& is_plotfile& is_openpmd& is_parthenon;
    }
  }

  template <typename Archive>
  void load(Archive& ar, unsigned) {
    ar& uri& step& level;

    bool is_memory = false;
    bool is_plotfile = false;
    bool is_openpmd = false;
    bool is_parthenon = false;
    ar& is_memory& is_plotfile& is_openpmd& is_parthenon;

    if (is_memory) {
      auto mem = std::make_shared<MemoryBackend>();
      std::unordered_map<ChunkRef, HostView, ChunkRefHash, ChunkRefEq> map;
      ar& map;
      mem->set_data(std::move(map));
      backend = mem;
    } else if (is_plotfile) {
      std::string path = uri;
      if (path.rfind("amrex://", 0) == 0) {
        path = path.substr(8);
      } else if (path.rfind("file://", 0) == 0) {
        path = path.substr(7);
      }
      backend = std::make_shared<PlotfileBackend>(path);
    } else if (is_openpmd) {
#ifdef KANGAROO_USE_OPENPMD
      backend = std::make_shared<OpenPMDBackend>(uri);
#else
      throw std::runtime_error("openPMD backend not enabled in this build");
#endif
    } else if (is_parthenon) {
#ifdef KANGAROO_USE_PARTHENON_HDF5
      std::string path = uri;
      if (path.rfind("parthenon://", 0) == 0) {
        path = path.substr(12);
      } else if (path.rfind("file://", 0) == 0) {
        path = path.substr(7);
      }
      backend = std::make_shared<ParthenonBackend>(path);
#else
      throw std::runtime_error("Parthenon HDF5 backend not enabled in this build");
#endif
    }
  }

  HPX_SERIALIZATION_SPLIT_MEMBER()
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
