#include "kangaroo/executor.hpp"

#include "kangaroo/data_service_local.hpp"
#include "kangaroo/param_decode.hpp"
#include "kangaroo/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_local/get_worker_thread_num.hpp>
#include <hpx/tuple.hpp>

namespace kangaroo {

namespace {

bool debug_dataflow_enabled() {
  static const bool enabled = std::getenv("KANGAROO_DEBUG_DATAFLOW") != nullptr;
  return enabled;
}

GraphReduceSpecIR parse_graph_reduce_params(const msgpack::object& root) {
  if (root.type == msgpack::type::NIL) {
    throw std::runtime_error("graph reduce params missing");
  }
  if (root.type != msgpack::type::MAP) {
    throw std::runtime_error("graph reduce params must be a map");
  }

  const auto* kind = find_msgpack_map_value(root, "graph_kind");
  if (!kind || kind->type != msgpack::type::STR || kind->as<std::string>() != "reduce") {
    throw std::runtime_error("graph template requires graph_kind=\"reduce\"");
  }

  GraphReduceSpecIR params;
  if (const auto* fan_in = find_msgpack_map_value(root, "fan_in"); fan_in &&
                                                (fan_in->type == msgpack::type::POSITIVE_INTEGER ||
                                                 fan_in->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.fan_in = fan_in->as<int32_t>();
  }
  if (const auto* num = find_msgpack_map_value(root, "num_inputs"); num &&
                                                 (num->type == msgpack::type::POSITIVE_INTEGER ||
                                                  num->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.num_inputs = num->as<int32_t>();
  }
  if (const auto* base = find_msgpack_map_value(root, "input_base"); base &&
                                                 (base->type == msgpack::type::POSITIVE_INTEGER ||
                                                  base->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.input_base = base->as<int32_t>();
  }
  if (const auto* base = find_msgpack_map_value(root, "output_base"); base &&
                                                  (base->type == msgpack::type::POSITIVE_INTEGER ||
                                                   base->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.output_base = base->as<int32_t>();
  }
  if (const auto* blocks = find_msgpack_map_value(root, "input_blocks");
      blocks && blocks->type == msgpack::type::ARRAY) {
    params.input_blocks.clear();
    params.input_blocks.reserve(blocks->via.array.size);
    for (uint32_t i = 0; i < blocks->via.array.size; ++i) {
      const auto& entry = blocks->via.array.ptr[i];
      if (entry.type == msgpack::type::POSITIVE_INTEGER ||
          entry.type == msgpack::type::NEGATIVE_INTEGER) {
        params.input_blocks.push_back(entry.as<int32_t>());
      }
    }
  }
  if (const auto* blocks = find_msgpack_map_value(root, "output_blocks");
      blocks && blocks->type == msgpack::type::ARRAY) {
    params.output_blocks.clear();
    params.output_blocks.reserve(blocks->via.array.size);
    for (uint32_t i = 0; i < blocks->via.array.size; ++i) {
      const auto& entry = blocks->via.array.ptr[i];
      if (entry.type == msgpack::type::POSITIVE_INTEGER ||
          entry.type == msgpack::type::NEGATIVE_INTEGER) {
        params.output_blocks.push_back(entry.as<int32_t>());
      }
    }
  }
  if (const auto* offsets = find_msgpack_map_value(root, "group_offsets");
      offsets && offsets->type == msgpack::type::ARRAY) {
    params.group_offsets.clear();
    params.group_offsets.reserve(offsets->via.array.size);
    for (uint32_t i = 0; i < offsets->via.array.size; ++i) {
      const auto& entry = offsets->via.array.ptr[i];
      if (entry.type == msgpack::type::POSITIVE_INTEGER ||
          entry.type == msgpack::type::NEGATIVE_INTEGER) {
        params.group_offsets.push_back(entry.as<int32_t>());
      }
    }
  }

  if (params.fan_in <= 0) {
    params.fan_in = 1;
  }
  if (params.num_inputs <= 0) {
    if (!params.input_blocks.empty()) {
      params.num_inputs = static_cast<int32_t>(params.input_blocks.size());
    } else {
      throw std::runtime_error("graph reduce num_inputs must be positive");
    }
  }
  if (!params.input_blocks.empty() &&
      params.num_inputs != static_cast<int32_t>(params.input_blocks.size())) {
    throw std::runtime_error("graph reduce num_inputs must match input_blocks size");
  }
  int32_t n_groups = (params.num_inputs + params.fan_in - 1) / params.fan_in;
  if (!params.group_offsets.empty()) {
    if (params.group_offsets.size() < 2) {
      throw std::runtime_error("graph reduce group_offsets must include start and end");
    }
    if (params.group_offsets.front() != 0 || params.group_offsets.back() != params.num_inputs) {
      throw std::runtime_error("graph reduce group_offsets must span num_inputs");
    }
    for (std::size_t i = 1; i < params.group_offsets.size(); ++i) {
      if (params.group_offsets[i] <= params.group_offsets[i - 1]) {
        throw std::runtime_error("graph reduce group_offsets must be strictly increasing");
      }
    }
    n_groups = static_cast<int32_t>(params.group_offsets.size() - 1);
  }
  if (!params.output_blocks.empty() &&
      n_groups != static_cast<int32_t>(params.output_blocks.size())) {
    throw std::runtime_error("graph reduce output_blocks size must match group count");
  }
  return params;
}

const KernelFn& prepared_kernel(const TaskTemplateIR& tmpl) {
  if (!tmpl.kernel_fn) {
    throw std::runtime_error("task template not prepared: missing kernel");
  }
  return *tmpl.kernel_fn;
}

const GraphReduceSpecIR& prepared_graph_reduce(const TaskTemplateIR& tmpl) {
  if (!tmpl.graph_reduce.has_value()) {
    throw std::runtime_error("graph task template not prepared: missing graph reduce params");
  }
  return tmpl.graph_reduce.value();
}

int32_t graph_reduce_output_block(const GraphReduceSpecIR& params, int32_t group_idx) {
  if (!params.output_blocks.empty()) {
    return params.output_blocks.at(static_cast<std::size_t>(group_idx));
  }
  return params.output_base + group_idx;
}

int32_t graph_reduce_group_count(const GraphReduceSpecIR& params) {
  if (!params.group_offsets.empty()) {
    return static_cast<int32_t>(params.group_offsets.size() - 1);
  }
  return (params.num_inputs + params.fan_in - 1) / params.fan_in;
}

int32_t graph_reduce_group_start(const GraphReduceSpecIR& params, int32_t group_idx) {
  if (!params.group_offsets.empty()) {
    return params.group_offsets.at(static_cast<std::size_t>(group_idx));
  }
  return group_idx * params.fan_in;
}

int32_t graph_reduce_group_end(const GraphReduceSpecIR& params, int32_t group_idx) {
  if (!params.group_offsets.empty()) {
    return params.group_offsets.at(static_cast<std::size_t>(group_idx + 1));
  }
  return std::min(graph_reduce_group_start(params, group_idx) + params.fan_in, params.num_inputs);
}

double now_seconds() {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

struct ViewSummary {
  std::size_t bytes = 0;
  bool interpreted_as_f64 = false;
  double sum = 0.0;
  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();
  std::size_t nonzero = 0;
};

ViewSummary summarize_view_f64(const HostView& view) {
  ViewSummary summary;
  summary.bytes = view.data.size();
  if (view.data.empty() || (view.data.size() % sizeof(double)) != 0) {
    return summary;
  }
  summary.interpreted_as_f64 = true;
  const std::size_t n = view.data.size() / sizeof(double);
  const auto* ptr = reinterpret_cast<const double*>(view.data.data());
  for (std::size_t i = 0; i < n; ++i) {
    const double value = ptr[i];
    summary.sum += value;
    if (std::isfinite(value)) {
      summary.min = std::min(summary.min, value);
      summary.max = std::max(summary.max, value);
    }
    if (value != 0.0) {
      ++summary.nonzero;
    }
  }
  if (!std::isfinite(summary.min)) {
    summary.min = std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(summary.max)) {
    summary.max = std::numeric_limits<double>::quiet_NaN();
  }
  return summary;
}

void log_projection_output_summary(const TaskTemplateIR& tmpl,
                                   int32_t block,
                                   std::size_t output_idx,
                                   const HostView& view) {
  if (!debug_dataflow_enabled()) {
    return;
  }
  if (tmpl.name.find("projection") == std::string::npos &&
      tmpl.kernel.find("projection") == std::string::npos &&
      tmpl.kernel != "uniform_slice_reduce" &&
      tmpl.kernel != "uniform_slice_add") {
    return;
  }
  const auto summary = summarize_view_f64(view);
  std::cout << "[kangaroo][output] name=" << tmpl.name
            << " kernel=" << tmpl.kernel
            << " locality=" << hpx::get_locality_id()
            << " step=" << tmpl.domain.step
            << " level=" << tmpl.domain.level
            << " block=" << block
            << " output_idx=" << output_idx
            << " bytes=" << summary.bytes;
  if (summary.interpreted_as_f64) {
    std::cout << " sum=" << summary.sum
              << " min=" << summary.min
              << " max=" << summary.max
              << " nonzero=" << summary.nonzero;
  }
  std::cout << std::endl;
}

struct InputLocation {
  int32_t step = 0;
  int16_t level = 0;
  int32_t block = 0;
};

InputLocation resolve_input_location(const TaskTemplateIR& tmpl,
                                     const FieldRefIR& input,
                                     int32_t task_block) {
  InputLocation loc{tmpl.domain.step, tmpl.domain.level, task_block};
  if (!input.domain.has_value()) {
    return loc;
  }

  const auto& dom = input.domain.value();
  loc.step = dom.step;
  loc.level = dom.level;
  if (!dom.blocks.has_value()) {
    return loc;
  }
  const auto& blocks = dom.blocks.value();
  if (blocks.empty()) {
    throw std::runtime_error("input domain blocks must be non-empty");
  }
  if (blocks.size() == 1) {
    loc.block = blocks.front();
    return loc;
  }
  if (std::find(blocks.begin(), blocks.end(), task_block) != blocks.end()) {
    loc.block = task_block;
    return loc;
  }
  throw std::runtime_error("task block not in input domain blocks");
}

TaskEvent base_task_event(const TaskTemplateIR& tmpl,
                          int32_t plan_id,
                          int32_t stage_idx,
                          int32_t tmpl_idx,
                          int32_t block) {
  TaskEvent event;
  event.id = std::to_string(plan_id) + ":" + std::to_string(stage_idx) + ":" +
             std::to_string(tmpl_idx) + ":" + std::to_string(block);
  event.name = tmpl.name;
  event.kernel = tmpl.kernel;
  event.plane = tmpl.plane == ExecPlane::Graph ? "graph" : "chunk";
  event.stage = stage_idx;
  event.template_index = tmpl_idx;
  event.block = block;
  event.step = tmpl.domain.step;
  event.level = tmpl.domain.level;
  event.locality = static_cast<int32_t>(hpx::get_locality_id());
  event.worker = static_cast<int32_t>(hpx::get_worker_thread_num());
  return event;
}

TaskEvent start_span(const TaskEvent& base, const std::string& suffix) {
  TaskEvent event = base;
  event.id = base.id + ":" + suffix;
  event.name = base.name + "/" + suffix;
  double ts = now_seconds();
  event.status = "start";
  event.ts = ts;
  event.start = ts;
  event.end = ts;
  log_task_event(event);
  return event;
}

bool event_log_enqueue_timing_enabled() {
  static const bool enabled = std::getenv("KANGAROO_TRACE_EVENT_LOG_OVERHEAD") != nullptr;
  return enabled;
}

void log_task_event_with_enqueue_timing(const TaskEvent& event,
                                        const TaskEvent& base,
                                        const std::string& suffix) {
  if (!event_log_enqueue_timing_enabled() || has_perfetto_trace()) {
    log_task_event(event);
    return;
  }

  const double start = now_seconds();
  log_task_event(event);
  const double end = now_seconds();

  TaskEvent timing = base;
  timing.id = base.id + ":" + suffix;
  timing.name = base.name + "/" + suffix;
  timing.status = "end";
  timing.ts = end;
  timing.start = start;
  timing.end = end;
  log_task_event(timing);
}

void end_span(TaskEvent event) {
  double ts = now_seconds();
  event.status = "end";
  event.ts = ts;
  event.end = ts;
  log_task_event(event);
}

hpx::future<void> run_block_task_impl(const TaskTemplateIR& tmpl,
                                      int32_t plan_id,
                                      int32_t stage_idx,
                                      int32_t tmpl_idx,
                                      int32_t block,
                                      const RunMeta& meta,
                                      DataService& data,
                                      AdjacencyService& adj) {
  if (tmpl.plane != ExecPlane::Chunk) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("only chunk plane is implemented"));
  }

  if (tmpl.outputs.empty()) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("templates must specify outputs"));
  }
  if (tmpl.inputs.empty() && tmpl.deps.kind == "FaceNeighbors") {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("FaceNeighbors deps require input fields"));
  }

  const bool log_enabled = has_event_log();
  TaskEvent base_event;
  if (log_enabled) {
    double start_ts = now_seconds();
    base_event = base_task_event(tmpl, plan_id, stage_idx, tmpl_idx, block);
    base_event.status = "start";
    base_event.ts = start_ts;
    base_event.start = start_ts;
    base_event.end = start_ts;
    log_task_event_with_enqueue_timing(base_event, base_event, "emit_task_start");
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

  TaskEvent get_inputs_event;
  if (log_enabled) {
    get_inputs_event = start_span(base_event, "get_inputs");
  }

  std::vector<ChunkRef> input_refs;
  input_refs.reserve(tmpl.inputs.size());
  TaskEvent resolve_inputs_event;
  if (log_enabled) {
    resolve_inputs_event = start_span(base_event, "resolve_inputs");
  }
  for (const auto& input : tmpl.inputs) {
    const auto loc = resolve_input_location(tmpl, input, block);
    input_refs.push_back(ChunkRef{loc.step, loc.level, input.field, input.version, loc.block});
  }
  if (log_enabled) {
    end_span(resolve_inputs_event);
  }

  std::vector<hpx::future<HostView>> input_futures;
  input_futures.reserve(input_refs.size());
  TaskEvent issue_inputs_event;
  if (log_enabled) {
    issue_inputs_event = start_span(base_event, "issue_get_inputs");
  }
  input_futures = data.get_hosts(input_refs);
  if (log_enabled) {
    end_span(issue_inputs_event);
  }

  auto fetch_face = [&](Face face,
                        int32_t field,
                        int32_t ver,
                        int32_t step,
                        int16_t level) -> hpx::future<std::vector<HostView>> {
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
        NeighborSpan ns = adj.neighbors(step, level, b, face);
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

    std::vector<ChunkRef> refs;
    refs.reserve(visited.size() > 0 ? visited.size() - 1 : 0);
    for (int32_t nbr : visited) {
      if (nbr == block) {
        continue;
      }
      ChunkRef cref{step, level, field, ver, nbr};
      refs.push_back(std::move(cref));
    }
    if (refs.empty()) {
      return hpx::make_ready_future(std::vector<HostView>{});
    }
    auto host_futures = data.get_hosts(refs);
    return hpx::when_all(host_futures).then([](auto&& all) {
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

  TaskEvent issue_neighbors_event;
  if (log_enabled) {
    issue_neighbors_event = start_span(base_event, "issue_neighbor_inputs");
  }
  for (std::size_t pos = 0; pos < halo_inputs.size(); ++pos) {
    const auto& nbr_field = tmpl.inputs.at(halo_inputs[pos]);
    int32_t step = tmpl.domain.step;
    int16_t level = tmpl.domain.level;
    if (nbr_field.domain.has_value()) {
      step = nbr_field.domain->step;
      level = nbr_field.domain->level;
    }
    neighbor_futures.push_back(fetch_face(Face::Xm, nbr_field.field, nbr_field.version, step, level));
    neighbor_slots.push_back({pos, Face::Xm});
    neighbor_futures.push_back(fetch_face(Face::Xp, nbr_field.field, nbr_field.version, step, level));
    neighbor_slots.push_back({pos, Face::Xp});
    neighbor_futures.push_back(fetch_face(Face::Ym, nbr_field.field, nbr_field.version, step, level));
    neighbor_slots.push_back({pos, Face::Ym});
    neighbor_futures.push_back(fetch_face(Face::Yp, nbr_field.field, nbr_field.version, step, level));
    neighbor_slots.push_back({pos, Face::Yp});
    neighbor_futures.push_back(fetch_face(Face::Zm, nbr_field.field, nbr_field.version, step, level));
    neighbor_slots.push_back({pos, Face::Zm});
    neighbor_futures.push_back(fetch_face(Face::Zp, nbr_field.field, nbr_field.version, step, level));
    neighbor_slots.push_back({pos, Face::Zp});
  }
  if (log_enabled) {
    end_span(issue_neighbors_event);
  }

  if (log_enabled) {
    end_span(get_inputs_event);
  }

  auto all_inputs = hpx::when_all(input_futures);
  auto all_neighbors = hpx::when_all(neighbor_futures);

  TaskEvent wait_inputs_event;
  if (log_enabled) {
    wait_inputs_event = start_span(base_event, "wait_inputs");
  }

  auto fut = hpx::when_all(std::move(all_inputs), std::move(all_neighbors))
      .then([&meta,
             &data,
             plan_id,
             tmpl,
             block,
             halo_inputs = std::move(halo_inputs),
             neighbor_slots = std::move(neighbor_slots),
             log_enabled,
             base_event,
             wait_inputs_event](auto&& all) mutable -> hpx::future<void> {
        if (log_enabled) {
          end_span(wait_inputs_event);
        }
        TaskEvent collect_inputs_event;
        if (log_enabled) {
          collect_inputs_event = start_span(base_event, "collect_inputs");
        }
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
        if (log_enabled) {
          end_span(collect_inputs_event);
        }

        TaskEvent alloc_outputs_event;
        if (log_enabled) {
          alloc_outputs_event = start_span(base_event, "alloc_outputs");
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
        if (log_enabled) {
          end_span(alloc_outputs_event);
        }

        auto inputs_ptr = std::make_shared<std::vector<HostView>>(std::move(inputs));
        auto nbrs_ptr = std::make_shared<NeighborViews>(std::move(nbrs));
        auto outputs_ptr = std::make_shared<std::vector<HostView>>(std::move(outputs));
        auto params_ptr = std::make_shared<std::vector<std::uint8_t>>(tmpl.params_msgpack);

        const auto& level = meta.steps.at(tmpl.domain.step).levels.at(tmpl.domain.level);
        auto& fn = prepared_kernel(tmpl);

        TaskEvent kernel_event;
        if (log_enabled) {
          kernel_event = start_span(base_event, "kernel");
        }
        ScopedExecutionContext active_context(plan_id);
        ScopedPreparedParams active_params(tmpl.prepared_params_type, tmpl.prepared_params);
        return fn(level,
                  block,
                  *inputs_ptr,
                  *nbrs_ptr,
                  *outputs_ptr,
                  std::span<const std::uint8_t>(params_ptr->data(), params_ptr->size()))
            .then([&data,
                   tmpl,
                   block,
                   inputs_ptr = std::move(inputs_ptr),
                   nbrs_ptr = std::move(nbrs_ptr),
                   outputs_ptr = std::move(outputs_ptr),
                   params_ptr = std::move(params_ptr),
                   log_enabled,
                   base_event,
                   kernel_event](auto&&) mutable {
              if (log_enabled) {
                end_span(kernel_event);
              }
              TaskEvent put_outputs_event;
              if (log_enabled) {
                put_outputs_event = start_span(base_event, "put_outputs");
              }
              std::vector<hpx::future<void>> puts;
              puts.reserve(tmpl.outputs.size());
              for (std::size_t i = 0; i < tmpl.outputs.size(); ++i) {
                log_projection_output_summary(tmpl, block, i, (*outputs_ptr)[i]);
                const auto& out = tmpl.outputs[i];
                ChunkRef cref{tmpl.domain.step, tmpl.domain.level, out.field, out.version, block};
                puts.push_back(data.put_host(cref, std::move((*outputs_ptr)[i])));
              }
              if (log_enabled) {
                end_span(put_outputs_event);
              }
              TaskEvent wait_outputs_event;
              if (log_enabled) {
                wait_outputs_event = start_span(base_event, "wait_outputs");
              }
              return hpx::when_all(puts).then(
                  [log_enabled, wait_outputs_event](auto&&) mutable {
                    if (log_enabled) {
                      end_span(wait_outputs_event);
                    }
                    return;
                  });
            });
      });
  if (!log_enabled) {
    return fut;
  }
  return fut.then([base_event](auto&& done) mutable {
    double end_ts = now_seconds();
    TaskEvent event = base_event;
    event.ts = end_ts;
    event.end = end_ts;
    try {
      done.get();
      event.status = "end";
      log_task_event_with_enqueue_timing(event, base_event, "emit_task_end");
      return;
    } catch (...) {
      event.status = "error";
      log_task_event_with_enqueue_timing(event, base_event, "emit_task_error");
      throw;
    }
  });
}

hpx::future<void> run_graph_task_impl(const TaskTemplateIR& tmpl,
                                      int32_t plan_id,
                                      int32_t stage_idx,
                                      int32_t tmpl_idx,
                                      int32_t group_idx,
                                      const RunMeta& meta,
                                      DataService& data) {
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

  const auto& params = prepared_graph_reduce(tmpl);
  const int32_t input_base = params.input_base;
  const bool debug_reduce = std::getenv("KANGAROO_DEBUG_REDUCE") != nullptr;

  const int32_t start = graph_reduce_group_start(params, group_idx);
  const int32_t end = graph_reduce_group_end(params, group_idx);
  if (start >= end) {
    return hpx::make_ready_future();
  }
  if (debug_reduce && !params.input_blocks.empty()) {
    std::ostringstream oss;
    oss << "[kangaroo] graph_reduce tmpl=" << tmpl.kernel << " group=" << group_idx
        << " blocks=";
    for (int32_t idx = start; idx < end; ++idx) {
      if (idx > start) {
        oss << ",";
      }
      oss << params.input_blocks.at(static_cast<std::size_t>(idx));
    }
    std::cout << oss.str() << std::endl;
  }
  const int32_t out_block = graph_reduce_output_block(params, group_idx);
  if (debug_dataflow_enabled()) {
    std::ostringstream oss;
    oss << "[kangaroo][dataflow] op=graph_reduce_inputs"
        << " locality=" << hpx::get_locality_id()
        << " name=" << tmpl.name
        << " kernel=" << tmpl.kernel
        << " group=" << group_idx
        << " out_block=" << out_block
        << " blocks=";
    for (int32_t idx = start; idx < end; ++idx) {
      if (idx > start) {
        oss << ",";
      }
      oss << (params.input_blocks.empty()
                  ? (input_base + idx)
                  : params.input_blocks.at(static_cast<std::size_t>(idx)));
    }
    std::cout << oss.str() << std::endl;
  }
  const bool log_enabled = has_event_log();
  TaskEvent base_event;
  if (log_enabled) {
    double start_ts = now_seconds();
    base_event = base_task_event(tmpl, plan_id, stage_idx, tmpl_idx, out_block);
    base_event.status = "start";
    base_event.ts = start_ts;
    base_event.start = start_ts;
    base_event.end = start_ts;
    log_task_event_with_enqueue_timing(base_event, base_event, "emit_task_start");
  }

  TaskEvent get_inputs_event;
  if (log_enabled) {
    get_inputs_event = start_span(base_event, "get_inputs");
  }
  std::vector<ChunkRef> input_refs;
  input_refs.reserve(static_cast<std::size_t>((end - start) * tmpl.inputs.size()));
  TaskEvent resolve_inputs_event;
  if (log_enabled) {
    resolve_inputs_event = start_span(base_event, "resolve_inputs");
  }
  for (const auto& input : tmpl.inputs) {
    for (int32_t idx = start; idx < end; ++idx) {
      const int32_t block_id = params.input_blocks.empty()
                                   ? (input_base + idx)
                                   : params.input_blocks.at(static_cast<std::size_t>(idx));
      int32_t step = tmpl.domain.step;
      int16_t level = tmpl.domain.level;
      if (input.domain.has_value()) {
        step = input.domain->step;
        level = input.domain->level;
      }
      input_refs.push_back(ChunkRef{step, level, input.field, input.version, block_id});
    }
  }
  if (log_enabled) {
    end_span(resolve_inputs_event);
  }

  std::vector<hpx::future<HostView>> input_futures;
  input_futures.reserve(input_refs.size());
  TaskEvent issue_inputs_event;
  if (log_enabled) {
    issue_inputs_event = start_span(base_event, "issue_get_inputs");
  }
  input_futures = data.get_hosts(input_refs);
  if (log_enabled) {
    end_span(issue_inputs_event);
  }
  if (log_enabled) {
    end_span(get_inputs_event);
  }

  TaskEvent wait_inputs_event;
  if (log_enabled) {
    wait_inputs_event = start_span(base_event, "wait_inputs");
  }
  auto fut = hpx::when_all(std::move(input_futures))
      .then([&meta,
             &data,
             plan_id,
             tmpl,
             out_block,
             log_enabled,
             base_event,
             wait_inputs_event](auto&& all) mutable -> hpx::future<void> {
        if (log_enabled) {
          end_span(wait_inputs_event);
        }
        TaskEvent collect_inputs_event;
        if (log_enabled) {
          collect_inputs_event = start_span(base_event, "collect_inputs");
        }
        auto input_pack = all.get();
        std::vector<HostView> inputs;
        inputs.reserve(input_pack.size());
        for (auto& f : input_pack) {
          inputs.push_back(f.get());
        }
        if (log_enabled) {
          end_span(collect_inputs_event);
        }
        if (debug_dataflow_enabled()) {
          std::ostringstream oss;
          oss << "[kangaroo][dataflow] op=graph_reduce_ready"
              << " locality=" << hpx::get_locality_id()
              << " name=" << tmpl.name
              << " kernel=" << tmpl.kernel
              << " out_block=" << out_block
              << " input_bytes=";
          for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (i > 0) {
              oss << ",";
            }
            oss << inputs[i].data.size();
          }
          std::cout << oss.str() << std::endl;
        }

        if (!tmpl.output_bytes.empty() && tmpl.output_bytes.size() != tmpl.outputs.size()) {
          throw std::runtime_error("output_bytes size must match outputs");
        }

        TaskEvent alloc_outputs_event;
        if (log_enabled) {
          alloc_outputs_event = start_span(base_event, "alloc_outputs");
        }
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
        if (log_enabled) {
          end_span(alloc_outputs_event);
        }

        NeighborViews nbrs;
        auto inputs_ptr = std::make_shared<std::vector<HostView>>(std::move(inputs));
        auto nbrs_ptr = std::make_shared<NeighborViews>(std::move(nbrs));
        auto outputs_ptr = std::make_shared<std::vector<HostView>>(std::move(outputs));
        auto params_ptr = std::make_shared<std::vector<std::uint8_t>>(tmpl.params_msgpack);

        const auto& level = meta.steps.at(tmpl.domain.step).levels.at(tmpl.domain.level);
        auto& fn = prepared_kernel(tmpl);

        TaskEvent kernel_event;
        if (log_enabled) {
          kernel_event = start_span(base_event, "kernel");
        }
        ScopedExecutionContext active_context(plan_id);
        ScopedPreparedParams active_params(tmpl.prepared_params_type, tmpl.prepared_params);
        return fn(level,
                  out_block,
                  *inputs_ptr,
                  *nbrs_ptr,
                  *outputs_ptr,
                  std::span<const std::uint8_t>(params_ptr->data(), params_ptr->size()))
            .then([&data,
                   tmpl,
                   out_block,
                   inputs_ptr = std::move(inputs_ptr),
                   nbrs_ptr = std::move(nbrs_ptr),
                   outputs_ptr = std::move(outputs_ptr),
                   params_ptr = std::move(params_ptr),
                   log_enabled,
                   base_event,
                   kernel_event](auto&&) mutable {
              if (log_enabled) {
                end_span(kernel_event);
              }
              TaskEvent put_outputs_event;
              if (log_enabled) {
                put_outputs_event = start_span(base_event, "put_outputs");
              }
              std::vector<hpx::future<void>> puts;
              puts.reserve(tmpl.outputs.size());
              for (std::size_t i = 0; i < tmpl.outputs.size(); ++i) {
                log_projection_output_summary(tmpl, out_block, i, (*outputs_ptr)[i]);
                const auto& out = tmpl.outputs[i];
                ChunkRef cref{tmpl.domain.step, tmpl.domain.level, out.field, out.version,
                              out_block};
                puts.push_back(data.put_host(cref, std::move((*outputs_ptr)[i])));
              }
              if (log_enabled) {
                end_span(put_outputs_event);
              }
              TaskEvent wait_outputs_event;
              if (log_enabled) {
                wait_outputs_event = start_span(base_event, "wait_outputs");
              }
              return hpx::when_all(puts).then(
                  [log_enabled, wait_outputs_event](auto&&) mutable {
                    if (log_enabled) {
                      end_span(wait_outputs_event);
                    }
                    return;
                  });
            });
      });
  if (!log_enabled) {
    return fut;
  }
  return fut.then([base_event](auto&& done) mutable {
    double end_ts = now_seconds();
    TaskEvent event = base_event;
    event.ts = end_ts;
    event.end = end_ts;
    try {
      done.get();
      event.status = "end";
      log_task_event_with_enqueue_timing(event, base_event, "emit_task_end");
      return;
    } catch (...) {
      event.status = "error";
      log_task_event_with_enqueue_timing(event, base_event, "emit_task_error");
      throw;
    }
  });
}

hpx::future<void> run_block_task_remote(int32_t plan_id, int32_t stage_idx, int32_t tmpl_idx,
                                        int32_t block) {
  auto ctx = execution_context_shared(plan_id);
  const auto& stage = ctx->plan.stages.at(stage_idx);
  const auto& tmpl = stage.templates.at(tmpl_idx);
  auto data = std::make_shared<DataServiceLocal>(plan_id);
  return run_block_task_impl(tmpl, plan_id, stage_idx, tmpl_idx, block, ctx->meta, *data,
                             *ctx->adjacency)
      .then([ctx = std::move(ctx), data = std::move(data)](auto&& done) mutable {
        done.get();
      });
}

hpx::future<void> run_graph_task_remote(int32_t plan_id, int32_t stage_idx, int32_t tmpl_idx,
                                        int32_t group_idx) {
  auto ctx = execution_context_shared(plan_id);
  const auto& stage = ctx->plan.stages.at(stage_idx);
  const auto& tmpl = stage.templates.at(tmpl_idx);
  auto data = std::make_shared<DataServiceLocal>(plan_id);
  return run_graph_task_impl(tmpl, plan_id, stage_idx, tmpl_idx, group_idx, ctx->meta, *data)
      .then([ctx = std::move(ctx), data = std::move(data)](auto&& done) mutable {
        done.get();
      });
}

}  // namespace

}  // namespace kangaroo

HPX_PLAIN_ACTION(kangaroo::run_block_task_remote, kangaroo_run_block_task_action)
HPX_PLAIN_ACTION(kangaroo::run_graph_task_remote, kangaroo_run_graph_task_action)

namespace kangaroo {

void prepare_plan(PlanIR& plan, KernelRegistry& kernels) {
  std::vector<std::shared_ptr<const CoveredBoxListIR>> shared_covered_boxes;
  shared_covered_boxes.reserve(plan.shared_covered_boxes.size());
  for (const auto& boxes : plan.shared_covered_boxes) {
    shared_covered_boxes.push_back(std::make_shared<CoveredBoxListIR>(boxes));
  }

  for (auto& stage : plan.stages) {
    for (auto& tmpl : stage.templates) {
      tmpl.kernel_fn = kernels.get_shared_by_name(tmpl.kernel);
      std::shared_ptr<const CoveredBoxListIR> covered_boxes;
      if (tmpl.covered_boxes_ref >= 0) {
        const auto ref_idx = static_cast<std::size_t>(tmpl.covered_boxes_ref);
        if (ref_idx >= shared_covered_boxes.size()) {
          throw std::runtime_error("covered_boxes_ref out of range while preparing plan");
        }
        covered_boxes = shared_covered_boxes[ref_idx];
      }
      auto prepared_params = kernels.prepare_params_by_name(
          tmpl.kernel,
          KernelParamContext{
              std::span<const std::uint8_t>(tmpl.params_msgpack.data(), tmpl.params_msgpack.size()),
              std::move(covered_boxes),
          });
      tmpl.prepared_params = std::move(prepared_params.value);
      tmpl.prepared_params_type = prepared_params.type;
      if (tmpl.plane == ExecPlane::Graph) {
        const auto& params = decode_params_cached<GraphReduceSpecIR>(
            std::span<const std::uint8_t>(tmpl.params_msgpack.data(), tmpl.params_msgpack.size()),
            parse_graph_reduce_params);
        tmpl.graph_reduce = params;
      } else {
        tmpl.graph_reduce.reset();
      }
    }
  }
}

Executor::Executor(int32_t plan_id, const RunMeta& meta, DataService& data, AdjacencyService& adj)
    : plan_id_(plan_id), meta_(meta), data_(data), adj_(adj) {}

hpx::future<void> Executor::run(const PlanIR& plan) {
  current_plan_ = &plan;
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
  if (nstages == 0) {
    current_plan_ = nullptr;
    return hpx::make_ready_future();
  }

  std::vector<hpx::shared_future<void>> stage_futures(nstages);
  for (int32_t stage_idx : order) {
    const auto& stage = plan.stages[stage_idx];
    if (stage.after.empty()) {
      stage_futures[stage_idx] = run_stage(stage_idx, stage).share();
      continue;
    }

    std::vector<hpx::shared_future<void>> deps;
    deps.reserve(stage.after.size());
    for (int32_t dep : stage.after) {
      deps.push_back(stage_futures.at(dep));
    }
    stage_futures[stage_idx] =
        hpx::when_all(deps).then([this, stage, stage_idx](auto&& ready_deps) {
          auto dep_results = ready_deps.get();
          for (auto& dep : dep_results) {
            dep.get();
          }
          return run_stage(stage_idx, stage);
        }).share();
  }

  return hpx::when_all(stage_futures).then([this](auto&& ready_stages) {
    try {
      auto stage_results = ready_stages.get();
      for (auto& stage : stage_results) {
        stage.get();
      }
    } catch (...) {
      current_plan_ = nullptr;
      throw;
    }
    current_plan_ = nullptr;
    return;
  });
}

hpx::future<void> Executor::run_stage(int32_t stage_idx, const StageIR& stage) {
  if (current_plan_ == nullptr) {
    throw std::runtime_error("executor run_stage requires active plan");
  }
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
      const auto& params = prepared_graph_reduce(tmpl);
      const int32_t n_groups = graph_reduce_group_count(params);
      for (int32_t group_idx = 0; group_idx < n_groups; ++group_idx) {
        futures.push_back(
            run_graph_task(tmpl, stage_idx, static_cast<int32_t>(tmpl_idx), group_idx));
      }
    } else {
      return hpx::make_exceptional_future<void>(
          std::runtime_error("unsupported execution plane"));
    }
  }

  if (futures.empty()) {
    return hpx::make_ready_future();
  }
  return hpx::when_all(futures).then([](auto&& ready_tasks) {
    auto task_results = ready_tasks.get();
    for (auto& task : task_results) {
      task.get();
    }
    return;
  });
}

hpx::future<void> Executor::run_block_task(const TaskTemplateIR& tmpl, int32_t stage_idx,
                                           int32_t tmpl_idx, int32_t block) {
  if (tmpl.plane != ExecPlane::Chunk) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("only chunk plane is implemented"));
  }

  if (tmpl.outputs.empty()) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("templates must specify outputs"));
  }
  const bool log_enabled = has_event_log();
  TaskEvent dispatch_event;
  if (log_enabled) {
    dispatch_event = base_task_event(tmpl, plan_id_, stage_idx, tmpl_idx, block);
  }
  TaskEvent resolve_target_event;
  if (log_enabled) {
    resolve_target_event = start_span(dispatch_event, "dispatch_resolve_target");
  }
  ChunkRef target_ref;
  if (tmpl.inputs.empty()) {
    target_ref = ChunkRef{tmpl.domain.step, tmpl.domain.level, tmpl.outputs.front().field,
                          tmpl.outputs.front().version, block};
  } else {
    const auto& first_input = tmpl.inputs.front();
    const auto loc = resolve_input_location(tmpl, first_input, block);
    target_ref = ChunkRef{loc.step, loc.level, first_input.field, first_input.version, loc.block};
  }
  int target = data_.home_rank(target_ref);
  if (log_enabled) {
    end_span(resolve_target_event);
  }
  int here = hpx::get_locality_id();
  if (target == here) {
    return hpx::unwrap(hpx::async([this, tmpl, stage_idx, tmpl_idx, block]() mutable {
      return run_block_task_impl(tmpl, plan_id_, stage_idx, tmpl_idx, block, meta_, data_, adj_);
    }));
  }

  auto localities = hpx::find_all_localities();
  TaskEvent launch_event;
  if (log_enabled) {
    launch_event = start_span(dispatch_event, "dispatch_remote_action");
  }
  auto fut = hpx::async<::kangaroo_run_block_task_action>(localities.at(target), plan_id_, stage_idx,
                                                          tmpl_idx, block);
  if (log_enabled) {
    end_span(launch_event);
  }
  return fut;
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

  const auto& params = prepared_graph_reduce(tmpl);
  const int32_t out_block = graph_reduce_output_block(params, group_idx);
  const bool log_enabled = has_event_log();
  TaskEvent dispatch_event;
  if (log_enabled) {
    dispatch_event = base_task_event(tmpl, plan_id_, stage_idx, tmpl_idx, out_block);
  }
  TaskEvent resolve_target_event;
  if (log_enabled) {
    resolve_target_event = start_span(dispatch_event, "dispatch_resolve_target");
  }
  const auto& out = tmpl.outputs.front();
  ChunkRef cref{tmpl.domain.step, tmpl.domain.level, out.field, out.version, out_block};

  int target = data_.home_rank(cref);
  if (log_enabled) {
    end_span(resolve_target_event);
  }
  int here = hpx::get_locality_id();
  if (target == here) {
    return hpx::unwrap(hpx::async([this, tmpl, stage_idx, tmpl_idx, group_idx]() mutable {
      return run_graph_task_impl(tmpl, plan_id_, stage_idx, tmpl_idx, group_idx, meta_, data_);
    }));
  }

  auto localities = hpx::find_all_localities();
  TaskEvent launch_event;
  if (log_enabled) {
    launch_event = start_span(dispatch_event, "dispatch_remote_action");
  }
  auto fut = hpx::async<::kangaroo_run_graph_task_action>(localities.at(target), plan_id_,
                                                          stage_idx, tmpl_idx, group_idx);
  if (log_enabled) {
    end_span(launch_event);
  }
  return fut;
}

}  // namespace kangaroo
