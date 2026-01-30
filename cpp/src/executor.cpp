#include "kangaroo/executor.hpp"

#include "kangaroo/runtime.hpp"
#include "kangaroo/data_service_local.hpp"

#include <array>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/tuple.hpp>

namespace kangaroo {

namespace {

hpx::future<void> run_block_task_impl(const TaskTemplateIR& tmpl, int32_t block, const RunMeta& meta,
                                      DataService& data, AdjacencyService& adj,
                                      KernelRegistry& kernels) {
  if (tmpl.plane != ExecPlane::Chunk) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("only chunk plane is implemented"));
  }

  if (tmpl.inputs.empty() || tmpl.outputs.empty()) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("templates must specify inputs and outputs"));
  }
  if (tmpl.deps.kind == "FaceNeighbors" && tmpl.inputs.size() != 1) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("FaceNeighbors deps require exactly one input field"));
  }

  std::vector<hpx::future<HostView>> input_futures;
  input_futures.reserve(tmpl.inputs.size());
  for (const auto& input : tmpl.inputs) {
    ChunkRef cref{tmpl.domain.step, tmpl.domain.level, input.field, input.version, block};
    input_futures.push_back(data.get_host(cref));
  }

  auto fetch_face = [&](Face face, int32_t field, int32_t ver) -> hpx::future<std::vector<HostView>> {
    if (tmpl.deps.kind != "FaceNeighbors") {
      return hpx::make_ready_future(std::vector<HostView>{});
    }
    const int32_t face_idx = static_cast<int32_t>(face);
    if (!tmpl.deps.faces[face_idx]) {
      return hpx::make_ready_future(std::vector<HostView>{});
    }
    if (tmpl.deps.width <= 0) {
      return hpx::make_ready_future(std::vector<HostView>{});
    }

    std::unordered_set<int32_t> visited;
    std::vector<int32_t> frontier;
    frontier.push_back(block);
    visited.insert(block);

    for (int32_t depth = 0; depth < tmpl.deps.width; ++depth) {
      std::vector<int32_t> next;
      for (int32_t b : frontier) {
        NeighborSpan ns = adj.neighbors(tmpl.domain.step, tmpl.domain.level, b, face);
        for (int32_t i = 0; i < ns.n; ++i) {
          int32_t nbr = ns.ptr[i];
          if (visited.insert(nbr).second) {
            next.push_back(nbr);
          }
        }
      }
      frontier.swap(next);
      if (frontier.empty()) {
        break;
      }
    }

    std::vector<hpx::future<HostView>> futs;
    futs.reserve(visited.size() > 0 ? visited.size() - 1 : 0);
    for (int32_t nbr : visited) {
      if (nbr == block) {
        continue;
      }
      ChunkRef cref{tmpl.domain.step, tmpl.domain.level, field, ver, nbr};
      futs.push_back(data.get_host(cref));
    }
    return hpx::when_all(futs).then([](auto&& all) {
      std::vector<HostView> out;
      out.reserve(all.get().size());
      for (auto& f : all.get()) {
        out.push_back(f.get());
      }
      return out;
    });
  };

  const auto& nbr_field = tmpl.inputs.front();
  auto f_xm = fetch_face(Face::Xm, nbr_field.field, nbr_field.version);
  auto f_xp = fetch_face(Face::Xp, nbr_field.field, nbr_field.version);
  auto f_ym = fetch_face(Face::Ym, nbr_field.field, nbr_field.version);
  auto f_yp = fetch_face(Face::Yp, nbr_field.field, nbr_field.version);
  auto f_zm = fetch_face(Face::Zm, nbr_field.field, nbr_field.version);
  auto f_zp = fetch_face(Face::Zp, nbr_field.field, nbr_field.version);

  auto all_inputs = hpx::when_all(input_futures);

  return hpx::when_all(std::move(all_inputs), std::move(f_xm), std::move(f_xp), std::move(f_ym),
                       std::move(f_yp), std::move(f_zm), std::move(f_zp))
      .then([&meta, &data, &kernels, tmpl, block](auto&& all) -> hpx::future<void> {
        auto results = all.get();
        auto input_pack = hpx::get<0>(results).get();
        auto xm = hpx::get<1>(results).get();
        auto xp = hpx::get<2>(results).get();
        auto ym = hpx::get<3>(results).get();
        auto yp = hpx::get<4>(results).get();
        auto zm = hpx::get<5>(results).get();
        auto zp = hpx::get<6>(results).get();

        std::vector<HostView> inputs;
        inputs.reserve(input_pack.size());
        for (auto& f : input_pack) {
          inputs.push_back(f.get());
        }

        std::vector<HostView> outputs;
        outputs.reserve(tmpl.outputs.size());
        if (!tmpl.output_bytes.empty() && tmpl.output_bytes.size() != tmpl.outputs.size()) {
          throw std::runtime_error("output_bytes size must match outputs");
        }
        for (std::size_t i = 0; i < tmpl.outputs.size(); ++i) {
          std::size_t bytes = 0;
          if (!tmpl.output_bytes.empty()) {
            bytes = static_cast<std::size_t>(tmpl.output_bytes[i]);
          }
          const auto& out = tmpl.outputs[i];
          ChunkRef cref{tmpl.domain.step, tmpl.domain.level, out.field, out.version, block};
          outputs.push_back(data.alloc_host(cref, bytes));
        }

        NeighborViews nbrs{xm, xp, ym, yp, zm, zp};
        const auto& level = meta.steps.at(tmpl.domain.step).levels.at(tmpl.domain.level);
        auto& fn = kernels.get_by_name(tmpl.kernel);

        return fn(level, block, inputs, nbrs, outputs, tmpl.params_msgpack)
            .then([&data, tmpl, block, outputs = std::move(outputs)](auto&&) mutable {
              std::vector<hpx::future<void>> puts;
              puts.reserve(tmpl.outputs.size());
              for (std::size_t i = 0; i < tmpl.outputs.size(); ++i) {
                const auto& out = tmpl.outputs[i];
                ChunkRef cref{tmpl.domain.step, tmpl.domain.level, out.field, out.version, block};
                puts.push_back(data.put_host(cref, std::move(outputs[i])));
              }
              return hpx::when_all(puts).then([](auto&&) { return; });
            });
      });
}

hpx::future<void> run_block_task_remote(int32_t plan_id, int32_t stage_idx, int32_t tmpl_idx,
                                        int32_t block) {
  const auto& plan = global_plan(plan_id);
  const auto& stage = plan.stages.at(stage_idx);
  const auto& tmpl = stage.templates.at(tmpl_idx);
  DataServiceLocal data;
  AdjacencyServiceLocal adjacency(global_runmeta());
  return run_block_task_impl(tmpl, block, global_runmeta(), data, adjacency, global_kernels());
}

}  // namespace

}  // namespace kangaroo

HPX_PLAIN_ACTION(kangaroo::run_block_task_remote, kangaroo_run_block_task_action)

namespace kangaroo {

Executor::Executor(int32_t plan_id, const RunMeta& meta, DataService& data, AdjacencyService& adj,
                   KernelRegistry& kr)
    : plan_id_(plan_id), meta_(meta), data_(data), adj_(adj), kernels_(kr) {}

hpx::future<void> Executor::run(const PlanIR& plan) {
  const int32_t nstages = static_cast<int32_t>(plan.stages.size());
  std::vector<int32_t> indeg(nstages, 0);
  std::vector<std::vector<int32_t>> edges(nstages);

  for (int32_t i = 0; i < nstages; ++i) {
    for (int32_t dep : plan.stages[i].after) {
      if (dep < 0 || dep >= nstages) {
        throw std::runtime_error("stage dependency index out of range");
      }
      edges[dep].push_back(i);
      indeg[i] += 1;
    }
  }

  std::queue<int32_t> ready;
  for (int32_t i = 0; i < nstages; ++i) {
    if (indeg[i] == 0) {
      ready.push(i);
    }
  }

  std::vector<int32_t> order;
  order.reserve(nstages);
  while (!ready.empty()) {
    int32_t s = ready.front();
    ready.pop();
    order.push_back(s);
    for (int32_t nxt : edges[s]) {
      indeg[nxt] -= 1;
      if (indeg[nxt] == 0) {
        ready.push(nxt);
      }
    }
  }

  if (static_cast<int32_t>(order.size()) != nstages) {
    throw std::runtime_error("stage dependency cycle detected");
  }

  hpx::future<void> chain = hpx::make_ready_future();
  for (int32_t stage_idx : order) {
    const auto& stage = plan.stages[stage_idx];
    chain = chain.then([this, stage, stage_idx](auto&&) {
      return run_stage(stage_idx, stage);
    });
  }
  return chain;
}

hpx::future<void> Executor::run_stage(int32_t stage_idx, const StageIR& stage) {
  if (stage.plane != ExecPlane::Chunk) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("only chunk plane is implemented"));
  }

  std::vector<hpx::future<void>> futures;
  for (std::size_t tmpl_idx = 0; tmpl_idx < stage.templates.size(); ++tmpl_idx) {
    const auto& tmpl = stage.templates[tmpl_idx];
    const auto& level = meta_.steps.at(tmpl.domain.step).levels.at(tmpl.domain.level);
    int32_t nblocks = static_cast<int32_t>(level.boxes.size());

    if (tmpl.domain.blocks.has_value()) {
      for (int32_t block : tmpl.domain.blocks.value()) {
        futures.push_back(
            run_block_task(tmpl, stage_idx, static_cast<int32_t>(tmpl_idx), block));
      }
    } else {
      for (int32_t block = 0; block < nblocks; ++block) {
        futures.push_back(
            run_block_task(tmpl, stage_idx, static_cast<int32_t>(tmpl_idx), block));
      }
    }
  }

  return hpx::when_all(futures).then([](auto&&) { return; });
}

hpx::future<void> Executor::run_block_task(const TaskTemplateIR& tmpl, int32_t stage_idx,
                                           int32_t tmpl_idx, int32_t block) {
  if (tmpl.plane != ExecPlane::Chunk) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("only chunk plane is implemented"));
  }

  if (tmpl.inputs.empty() || tmpl.outputs.empty()) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("templates must specify inputs and outputs"));
  }

  const auto& first_input = tmpl.inputs.front();
  ChunkRef cref{tmpl.domain.step, tmpl.domain.level, first_input.field, first_input.version, block};

  int target = data_.home_rank(cref);
  int here = hpx::get_locality_id();
  if (target == here) {
    return run_block_task_impl(tmpl, block, meta_, data_, adj_, kernels_);
  }

  auto localities = hpx::find_all_localities();
  return hpx::async<::kangaroo_run_block_task_action>(localities.at(target), plan_id_, stage_idx,
                                                      tmpl_idx, block);
}

}  // namespace kangaroo
