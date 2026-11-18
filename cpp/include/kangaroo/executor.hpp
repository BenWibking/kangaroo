#pragma once

#include "kangaroo/adjacency.hpp"
#include "kangaroo/data_service.hpp"
#include "kangaroo/kernel_registry.hpp"
#include "kangaroo/plan_ir.hpp"
#include "kangaroo/runmeta.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <hpx/future.hpp>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace kangaroo {

struct DatasetHandle;

struct TaskProgress {
  std::atomic<std::uint64_t> completed{0};
};

void prepare_plan(PlanIR& plan, KernelRegistry& kernels);
void validate_plan_kernel_contracts(const PlanIR& plan,
                                    const KernelRegistry& kernels);
void validate_plan_output_bounds(const PlanIR& plan, const KernelRegistry& kernels);

struct ExecutorOptions {
  std::string mode = "eager";
  int32_t max_active_tasks_per_locality = 128;
  int32_t max_active_storage_units_per_locality = 0;
  int32_t max_active_tasks_per_stage = 0;
  std::size_t max_input_bytes_per_locality = 0;
  std::size_t max_output_storage_bytes_per_locality = 0;
  bool enable_task_level_stage_overlap = false;
  bool enable_early_release = false;
};

ExecutorOptions executor_options_from_environment();

enum class TaskKind {
  Chunk,
  Graph,
};

struct TaskInstance {
  TaskKind kind = TaskKind::Chunk;
  int32_t stage_idx = 0;
  int32_t tmpl_idx = 0;
  int32_t block_or_group = 0;
  int32_t target_locality = 0;
  std::vector<ChunkRef> input_refs;
  std::vector<ChunkRef> output_refs;
  std::size_t estimated_input_bytes = 0;
  std::size_t estimated_output_storage_bytes = 0;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& kind& stage_idx& tmpl_idx& block_or_group& target_locality& input_refs& output_refs&
        estimated_input_bytes& estimated_output_storage_bytes;
  }
};

class Executor {
 public:
  Executor(int32_t plan_id, const RunMeta& meta, DataService& data, AdjacencyService& adj,
           ExecutorOptions options, std::shared_ptr<TaskProgress> progress = nullptr);

  hpx::future<void> run(const PlanIR& plan);

 private:
  struct StreamingPlanEstimates;

  hpx::future<void> run_stage_eager(int32_t stage_idx, const StageIR& stage);
  hpx::future<void> run_stage_streaming(int32_t stage_idx, const StageIR& stage,
                                        const StreamingPlanEstimates& estimates);
  hpx::future<void> run_block_task(const TaskTemplateIR& tmpl, int32_t stage_idx, int32_t tmpl_idx,
                                   int32_t block);
  hpx::future<void> run_graph_task(const TaskTemplateIR& tmpl, int32_t stage_idx, int32_t tmpl_idx,
                                   int32_t group_idx);
  std::vector<TaskInstance> expand_stage_tasks(int32_t stage_idx, const StageIR& stage,
                                                const StreamingPlanEstimates& estimates) const;
  hpx::future<void> register_streaming_input_consumers(
      const PlanIR& plan, const StreamingPlanEstimates& estimates) const;
  std::size_t streaming_locality_task_window() const;
  std::size_t streaming_locality_storage_unit_window() const;

  int32_t plan_id_ = 0;
  const RunMeta& meta_;
  DataService& data_;
  AdjacencyService& adj_;
  ExecutorOptions options_;
  std::shared_ptr<TaskProgress> progress_;
  const PlanIR* current_plan_ = nullptr;
};

}  // namespace kangaroo
