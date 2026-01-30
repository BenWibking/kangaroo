#pragma once

#include "kangaroo/adjacency.hpp"
#include "kangaroo/data_service.hpp"
#include "kangaroo/kernel_registry.hpp"
#include "kangaroo/plan_ir.hpp"
#include "kangaroo/runmeta.hpp"

#include <hpx/future.hpp>

namespace kangaroo {

class Executor {
 public:
  Executor(int32_t plan_id, const RunMeta& meta, DataService& data, AdjacencyService& adj,
           KernelRegistry& kr);

  hpx::future<void> run(const PlanIR& plan);

 private:
  hpx::future<void> run_stage(int32_t stage_idx, const StageIR& stage);
  hpx::future<void> run_block_task(const TaskTemplateIR& tmpl, int32_t stage_idx, int32_t tmpl_idx,
                                   int32_t block);
  hpx::future<void> run_graph_task(const TaskTemplateIR& tmpl, int32_t stage_idx, int32_t tmpl_idx,
                                   int32_t group_idx);

  int32_t plan_id_ = 0;
  const RunMeta& meta_;
  DataService& data_;
  AdjacencyService& adj_;
  KernelRegistry& kernels_;
};

}  // namespace kangaroo
