#pragma once

#include "kangaroo/adjacency.hpp"
#include "kangaroo/data_service.hpp"
#include "kangaroo/data_service_local.hpp"
#include "kangaroo/dataset_backend.hpp"
#include "kangaroo/executor.hpp"
#include "kangaroo/kernel_registry.hpp"
#include "kangaroo/runmeta.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hpx/include/lcos.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/recursive_mutex.hpp>

namespace kangaroo {

struct RunMetaHandle {
  RunMeta meta;
};

struct DatasetHandle {
  std::string uri;
  int32_t step = 0;
  int16_t level = 0;

  std::shared_ptr<DatasetBackend> backend;

  void set_chunk(const ChunkRef& ref, ChunkBuffer view);
  std::optional<ChunkBuffer> get_chunk(const ChunkRef& ref) const;
  std::vector<std::optional<ChunkBuffer>> get_chunks(const std::vector<ChunkRef>& refs) const;
  bool has_chunk(const ChunkRef& ref) const;
  std::size_t estimate_chunk_bytes(const ChunkRef& ref) const;

  template <typename Archive>
  void save(Archive& ar, unsigned) const {
    ar& uri& step& level;
    DatasetBackendSnapshot snapshot;
    if (backend) {
      snapshot = backend->snapshot();
    }
    ar& snapshot.kind& snapshot.component_fields& snapshot.memory_chunks;
  }

  template <typename Archive>
  void load(Archive& ar, unsigned) {
    ar& uri& step& level;
    DatasetBackendSnapshot snapshot;
    ar& snapshot.kind& snapshot.component_fields& snapshot.memory_chunks;
    if (snapshot.kind.empty()) {
      backend.reset();
    } else {
      backend = restore_dataset_backend(uri, snapshot);
    }
  }

  HPX_SERIALIZATION_SPLIT_MEMBER()
};

struct ChunkSlot {
  std::shared_ptr<hpx::promise<std::shared_ptr<ChunkBuffer>>> promise;
  hpx::shared_future<std::shared_ptr<ChunkBuffer>> future;
  std::shared_ptr<ChunkBuffer> value;
  bool ready = false;
  bool dataset_load_started = false;
};

struct ChunkStore {
  using MapT = std::unordered_map<ChunkRef, ChunkSlot, ChunkRefHash, ChunkRefEq>;
  using ConsumerMapT = std::unordered_map<ChunkRef, std::int64_t, ChunkRefHash, ChunkRefEq>;
  using Mutex = hpx::recursive_mutex;

  Mutex mutex;
  MapT data;
  ConsumerMapT consumer_counts;
};

struct ExecutionContext {
  int32_t run_id = 0;
  RunMeta meta;
  DatasetHandle dataset;
  PlanIR plan;
  std::shared_ptr<AdjacencyService> adjacency;
  std::shared_ptr<ChunkStore> chunk_store;
};

class ScopedExecutionContext {
 public:
  explicit ScopedExecutionContext(int32_t run_id);
  ~ScopedExecutionContext();

 private:
  int32_t previous_run_id_ = 0;
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

  ChunkBuffer get_task_chunk(int32_t step,
                          int16_t level,
                          int32_t field,
                          int32_t version,
                          int32_t block,
                          const DatasetHandle* dataset = nullptr);

  int32_t locality_id();
  int32_t num_localities();
  int32_t chunk_home_rank(int32_t step, int16_t level, int32_t block);
  void wait_for_console_release();
  void release_console_workers();

  void set_event_log_path(const std::string& path);
  void set_perfetto_trace_path(const std::string& path);

 private:
  int32_t next_field_id_ = 1000;
  int32_t retained_output_run_id_ = 0;
  std::vector<int32_t> retained_output_run_ids_;
  int32_t preload_run_id_ = 0;
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

struct PhaseEvent {
  std::string name;
  std::string category;
  std::string status;
  int32_t locality = -1;
  int32_t worker = -1;
  std::string worker_label;
  double ts = 0.0;
  double start = 0.0;
  double end = 0.0;
};

struct DataEvent {
  std::string op;
  std::string mode;
  std::string status;
  std::string file;
  ChunkRef ref;
  int32_t locality = -1;
  int32_t target_locality = -1;
  int32_t worker = -1;
  std::string worker_label;
  std::size_t bytes = 0;
  std::size_t estimated_bytes = 0;
  int64_t file_offset = -1;
  int32_t comp_start = -1;
  int32_t comp_count = -1;
  int32_t queue_depth = -1;
  int32_t in_flight = -1;
  int32_t concurrency = -1;
  int64_t in_flight_bytes = -1;
  int64_t byte_limit = -1;
  double ts = 0.0;
  double start = 0.0;
  double end = 0.0;
};

void set_event_log_path(const std::string& path);
bool has_event_log();
void set_perfetto_trace_path(const std::string& path);
bool has_perfetto_trace();
void log_task_event(const TaskEvent& event);
void log_phase_event(const PhaseEvent& event);
void log_data_event(const DataEvent& event);

void set_execution_context(int32_t run_id,
                           const RunMeta& meta,
                           const DatasetHandle& dataset,
                           const PlanIR& plan);
std::shared_ptr<ExecutionContext> execution_context_shared(int32_t run_id);
const ExecutionContext& execution_context(int32_t run_id);
bool execution_context_may_produce_chunk(int32_t run_id, const ChunkRef& ref);
void erase_execution_context(int32_t run_id);
const RunMeta& current_runmeta();
const DatasetHandle& current_dataset();

void set_global_kernel_registry(KernelRegistry* registry);
KernelRegistry& global_kernels();

}  // namespace kangaroo
