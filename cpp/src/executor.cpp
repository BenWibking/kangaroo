#include "kangaroo/executor.hpp"

#include "kangaroo/runtime.hpp"
#include "kangaroo/data_service_local.hpp"

#include <algorithm>
#include <array>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#include <msgpack.hpp>

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/tuple.hpp>

namespace kangaroo {

namespace {

struct GraphReduceParams {
  int32_t fan_in = 1;
  int32_t num_inputs = 0;
  int32_t input_base = 0;
  int32_t output_base = 0;
};

GraphReduceParams parse_graph_reduce_params(const std::vector<std::uint8_t>& params_msgpack) {
  if (params_msgpack.empty()) {
    throw std::runtime_error("graph reduce params missing");
  }
  auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                params_msgpack.size());
  auto root = handle.get();
  if (root.type != msgpack::type::MAP) {
    throw std::runtime_error("graph reduce params must be a map");
  }
  auto get_key = [&](const char* key) -> const msgpack::object* {
    for (uint32_t i = 0; i < root.via.map.size; ++i) {
      const auto& k = root.via.map.ptr[i].key;
      if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
        return &root.via.map.ptr[i].val;
      }
    }
    return nullptr;
  };

  const auto* kind = get_key("graph_kind");
  if (!kind || kind->type != msgpack::type::STR || kind->as<std::string>() != "reduce") {
    throw std::runtime_error("graph template requires graph_kind=\"reduce\"");
  }

  GraphReduceParams params;
  if (const auto* fan_in = get_key("fan_in"); fan_in &&
                                                (fan_in->type == msgpack::type::POSITIVE_INTEGER ||
                                                 fan_in->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.fan_in = fan_in->as<int32_t>();
  }
  if (const auto* num = get_key("num_inputs"); num &&
                                                 (num->type == msgpack::type::POSITIVE_INTEGER ||
                                                  num->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.num_inputs = num->as<int32_t>();
  }
  if (const auto* base = get_key("input_base"); base &&
                                                 (base->type == msgpack::type::POSITIVE_INTEGER ||
                                                  base->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.input_base = base->as<int32_t>();
  }
  if (const auto* base = get_key("output_base"); base &&
                                                  (base->type == msgpack::type::POSITIVE_INTEGER ||
                                                   base->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.output_base = base->as<int32_t>();
  }

  if (params.fan_in <= 0) {
    params.fan_in = 1;
  }
  if (params.num_inputs <= 0) {
    throw std::runtime_error("graph reduce num_inputs must be positive");
  }
  return params;
}

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

  std::vector<int32_t> halo_inputs;
  if (tmpl.deps.kind == "FaceNeighbors") {
    if (tmpl.inputs.empty()) {
      return hpx::make_exceptional_future<void>(
          std::runtime_error("FaceNeighbors deps require at least one input field"));
    }
    const auto& requested = tmpl.deps.halo_inputs;
    if (requested.empty()) {
      halo_inputs.push_back(0);
    } else {
      std::unordered_set<int32_t> seen;
      for (int32_t idx : requested) {
        if (idx < 0 || idx >= static_cast<int32_t>(tmpl.inputs.size())) {
          return hpx::make_exceptional_future<void>(
              std::runtime_error("halo_inputs index out of range"));
        }
        if (seen.insert(idx).second) {
          halo_inputs.push_back(idx);
        }
      }
    }
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

  struct NeighborSlot {
    std::size_t input_pos;
    Face face;
  };

  std::vector<hpx::future<std::vector<HostView>>> neighbor_futures;
  std::vector<NeighborSlot> neighbor_slots;
  neighbor_futures.reserve(halo_inputs.size() * 6);
  neighbor_slots.reserve(halo_inputs.size() * 6);

  for (std::size_t pos = 0; pos < halo_inputs.size(); ++pos) {
    const auto& nbr_field = tmpl.inputs.at(halo_inputs[pos]);
    neighbor_futures.push_back(fetch_face(Face::Xm, nbr_field.field, nbr_field.version));
    neighbor_slots.push_back({pos, Face::Xm});
    neighbor_futures.push_back(fetch_face(Face::Xp, nbr_field.field, nbr_field.version));
    neighbor_slots.push_back({pos, Face::Xp});
    neighbor_futures.push_back(fetch_face(Face::Ym, nbr_field.field, nbr_field.version));
    neighbor_slots.push_back({pos, Face::Ym});
    neighbor_futures.push_back(fetch_face(Face::Yp, nbr_field.field, nbr_field.version));
    neighbor_slots.push_back({pos, Face::Yp});
    neighbor_futures.push_back(fetch_face(Face::Zm, nbr_field.field, nbr_field.version));
    neighbor_slots.push_back({pos, Face::Zm});
    neighbor_futures.push_back(fetch_face(Face::Zp, nbr_field.field, nbr_field.version));
    neighbor_slots.push_back({pos, Face::Zp});
  }

  auto all_inputs = hpx::when_all(input_futures);
  auto all_neighbors = hpx::when_all(neighbor_futures);

  return hpx::when_all(std::move(all_inputs), std::move(all_neighbors))
      .then([&meta, &data, &kernels, tmpl, block, halo_inputs = std::move(halo_inputs),
             neighbor_slots = std::move(neighbor_slots)](auto&& all) -> hpx::future<void> {
        auto results = all.get();
        auto input_pack = hpx::get<0>(results).get();
        auto neighbor_pack = hpx::get<1>(results).get();

        std::vector<HostView> inputs;
        inputs.reserve(input_pack.size());
        for (auto& f : input_pack) {
          inputs.push_back(f.get());
        }

        NeighborViews nbrs;
        nbrs.input_indices = halo_inputs;
        nbrs.inputs.resize(halo_inputs.size());

        auto assign_face = [](NeighborViews::FieldNeighbors& field, Face face,
                              std::vector<HostView>&& views) {
          switch (face) {
            case Face::Xm:
              field.xm = std::move(views);
              break;
            case Face::Xp:
              field.xp = std::move(views);
              break;
            case Face::Ym:
              field.ym = std::move(views);
              break;
            case Face::Yp:
              field.yp = std::move(views);
              break;
            case Face::Zm:
              field.zm = std::move(views);
              break;
            case Face::Zp:
              field.zp = std::move(views);
              break;
          }
        };

        for (std::size_t i = 0; i < neighbor_pack.size(); ++i) {
          auto views = neighbor_pack[i].get();
          const auto& slot = neighbor_slots.at(i);
          assign_face(nbrs.inputs.at(slot.input_pos), slot.face, std::move(views));
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

hpx::future<void> run_graph_task_impl(const TaskTemplateIR& tmpl, int32_t group_idx,
                                      const RunMeta& meta, DataService& data,
                                      KernelRegistry& kernels) {
  if (tmpl.plane != ExecPlane::Graph) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("graph tasks require graph plane"));
  }
  if (tmpl.deps.kind != "None") {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("graph tasks do not support deps"));
  }
  if (tmpl.inputs.empty() || tmpl.outputs.empty()) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("graph templates must specify inputs and outputs"));
  }

  const auto params = parse_graph_reduce_params(tmpl.params_msgpack);
  const int32_t fan_in = params.fan_in;
  const int32_t num_inputs = params.num_inputs;
  const int32_t input_base = params.input_base;
  const int32_t output_base = params.output_base;

  const int32_t start = group_idx * fan_in;
  const int32_t end = std::min(start + fan_in, num_inputs);
  if (start >= end) {
    return hpx::make_ready_future();
  }

  std::vector<hpx::future<HostView>> input_futures;
  input_futures.reserve(static_cast<std::size_t>((end - start) * tmpl.inputs.size()));
  for (const auto& input : tmpl.inputs) {
    for (int32_t idx = start; idx < end; ++idx) {
      ChunkRef cref{tmpl.domain.step, tmpl.domain.level, input.field, input.version,
                    input_base + idx};
      input_futures.push_back(data.get_host(cref));
    }
  }

  return hpx::when_all(std::move(input_futures))
      .then([&meta, &data, &kernels, tmpl, group_idx, output_base](auto&& all)
                -> hpx::future<void> {
        auto input_pack = all.get();
        std::vector<HostView> inputs;
        inputs.reserve(input_pack.size());
        for (auto& f : input_pack) {
          inputs.push_back(f.get());
        }

        if (!tmpl.output_bytes.empty() && tmpl.output_bytes.size() != tmpl.outputs.size()) {
          throw std::runtime_error("output_bytes size must match outputs");
        }

        const int32_t out_block = output_base + group_idx;

        std::vector<HostView> outputs;
        outputs.reserve(tmpl.outputs.size());
        for (std::size_t i = 0; i < tmpl.outputs.size(); ++i) {
          std::size_t bytes = 0;
          if (!tmpl.output_bytes.empty()) {
            bytes = static_cast<std::size_t>(tmpl.output_bytes[i]);
          }
          const auto& out = tmpl.outputs[i];
          ChunkRef cref{tmpl.domain.step, tmpl.domain.level, out.field, out.version, out_block};
          outputs.push_back(data.alloc_host(cref, bytes));
        }

        const auto& level = meta.steps.at(tmpl.domain.step).levels.at(tmpl.domain.level);
        auto& fn = kernels.get_by_name(tmpl.kernel);

        NeighborViews nbrs;
        return fn(level, out_block, inputs, nbrs, outputs, tmpl.params_msgpack)
            .then([&data, tmpl, out_block, outputs = std::move(outputs)](auto&&) mutable {
              std::vector<hpx::future<void>> puts;
              puts.reserve(tmpl.outputs.size());
              for (std::size_t i = 0; i < tmpl.outputs.size(); ++i) {
                const auto& out = tmpl.outputs[i];
                ChunkRef cref{tmpl.domain.step, tmpl.domain.level, out.field, out.version,
                              out_block};
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

hpx::future<void> run_graph_task_remote(int32_t plan_id, int32_t stage_idx, int32_t tmpl_idx,
                                        int32_t group_idx) {
  const auto& plan = global_plan(plan_id);
  const auto& stage = plan.stages.at(stage_idx);
  const auto& tmpl = stage.templates.at(tmpl_idx);
  DataServiceLocal data;
  return run_graph_task_impl(tmpl, group_idx, global_runmeta(), data, global_kernels());
}

}  // namespace

}  // namespace kangaroo

HPX_PLAIN_ACTION(kangaroo::run_block_task_remote, kangaroo_run_block_task_action)
HPX_PLAIN_ACTION(kangaroo::run_graph_task_remote, kangaroo_run_graph_task_action)

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
  std::vector<hpx::future<void>> futures;
  for (std::size_t tmpl_idx = 0; tmpl_idx < stage.templates.size(); ++tmpl_idx) {
    const auto& tmpl = stage.templates[tmpl_idx];
    if (stage.plane == ExecPlane::Chunk) {
      if (tmpl.plane != ExecPlane::Chunk) {
        return hpx::make_exceptional_future<void>(
            std::runtime_error("chunk stage requires chunk templates"));
      }
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
    } else if (stage.plane == ExecPlane::Graph) {
      if (tmpl.plane != ExecPlane::Graph) {
        return hpx::make_exceptional_future<void>(
            std::runtime_error("graph stage requires graph templates"));
      }
      const auto params = parse_graph_reduce_params(tmpl.params_msgpack);
      const int32_t fan_in = params.fan_in;
      const int32_t num_inputs = params.num_inputs;
      const int32_t n_groups = (num_inputs + fan_in - 1) / fan_in;
      for (int32_t group_idx = 0; group_idx < n_groups; ++group_idx) {
        futures.push_back(
            run_graph_task(tmpl, stage_idx, static_cast<int32_t>(tmpl_idx), group_idx));
      }
    } else {
      return hpx::make_exceptional_future<void>(
          std::runtime_error("unsupported execution plane"));
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

hpx::future<void> Executor::run_graph_task(const TaskTemplateIR& tmpl, int32_t stage_idx,
                                           int32_t tmpl_idx, int32_t group_idx) {
  if (tmpl.plane != ExecPlane::Graph) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("graph task requires graph plane"));
  }

  if (tmpl.outputs.empty()) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("graph templates must specify outputs"));
  }

  const auto params = parse_graph_reduce_params(tmpl.params_msgpack);
  const int32_t out_block = params.output_base + group_idx;
  const auto& out = tmpl.outputs.front();
  ChunkRef cref{tmpl.domain.step, tmpl.domain.level, out.field, out.version, out_block};

  int target = data_.home_rank(cref);
  int here = hpx::get_locality_id();
  if (target == here) {
    return run_graph_task_impl(tmpl, group_idx, meta_, data_, kernels_);
  }

  auto localities = hpx::find_all_localities();
  return hpx::async<::kangaroo_run_graph_task_action>(localities.at(target), plan_id_, stage_idx,
                                                      tmpl_idx, group_idx);
}

}  // namespace kangaroo
