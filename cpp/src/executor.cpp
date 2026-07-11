#include "kangaroo/executor.hpp"

#include "kangaroo/data_service_local.hpp"
#include "kangaroo/param_decode.hpp"
#include "kangaroo/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/async_combinators/when_any.hpp>
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

ViewSummary summarize_view_f64(const ChunkBuffer& view) {
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
                                   const ChunkBuffer& view) {
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

std::size_t checked_positive_extent(int32_t lo, int32_t hi) {
  if (hi < lo) {
    return 0;
  }
  return static_cast<std::size_t>(hi - lo + 1);
}

std::size_t block_cell_count(const RunMeta& meta, int32_t step, int16_t level, int32_t block) {
  const auto& box = meta.steps.at(static_cast<std::size_t>(step))
                        .levels.at(static_cast<std::size_t>(level))
                        .boxes.at(static_cast<std::size_t>(block));
  const std::size_t nx = checked_positive_extent(box.lo.x, box.hi.x);
  const std::size_t ny = checked_positive_extent(box.lo.y, box.hi.y);
  const std::size_t nz = checked_positive_extent(box.lo.z, box.hi.z);
  return nx * ny * nz;
}

std::array<std::uint64_t, 3> block_extents(const RunMeta& meta,
                                           int32_t step,
                                           int16_t level,
                                           int32_t block) {
  const auto& box = meta.steps.at(static_cast<std::size_t>(step))
                        .levels.at(static_cast<std::size_t>(level))
                        .boxes.at(static_cast<std::size_t>(block));
  return {checked_positive_extent(box.lo.x, box.hi.x),
          checked_positive_extent(box.lo.y, box.hi.y),
          checked_positive_extent(box.lo.z, box.hi.z)};
}

std::optional<BufferDesc> resolve_static_output_desc(const BufferSpecIR& spec,
                                                     const RunMeta& meta,
                                                     int32_t step,
                                                     int16_t level,
                                                     int32_t block) {
  if (spec.shape_kind == ShapeRuleKind::kFixed) {
    return BufferDesc::contiguous(spec.scalar, spec.fixed_extents);
  }
  if (spec.shape_kind == ShapeRuleKind::kBlock) {
    const auto xyz = block_extents(meta, step, level, block);
    if (spec.block_components == 1) return BufferDesc::runtime_grid(spec.scalar, xyz);
    const std::array<std::uint64_t, 4> extents{
        xyz[0], xyz[1], xyz[2], spec.block_components};
    return BufferDesc::contiguous(spec.scalar, extents);
  }
  if (spec.shape_kind == ShapeRuleKind::kDynamic &&
      spec.dynamic_upper_bound.kind == DynamicUpperBoundKind::kLiteral) {
    const std::array<std::uint64_t, 1> extents{spec.dynamic_upper_bound.value};
    return BufferDesc::contiguous(spec.scalar, extents);
  }
  return std::nullopt;
}

std::size_t output_spec_bytes(const BufferSpecIR& spec,
                              const RunMeta& meta,
                              int32_t step,
                              int16_t level,
                              int32_t block) {
  try {
    const auto desc = resolve_static_output_desc(spec, meta, step, level, block);
    return desc.has_value() ? static_cast<std::size_t>(desc->required_bytes()) : 0;
  } catch (...) {
    return 0;
  }
}

ResolvedBufferSpec make_dynamic_output_spec(const BufferSpecIR& spec, std::uint64_t capacity) {
  ResolvedBufferSpec resolved;
  resolved.init = spec.init;
  const std::array<std::uint64_t, 1> capacity_extent{capacity};
  resolved.desc = BufferDesc::contiguous(spec.scalar, capacity_extent);
  resolved.desc.extents[0] = 0;
  resolved.dynamic_capacity_elements = capacity;
  return resolved;
}

std::uint64_t derive_backend_chunk_capacity(const BufferSpecIR& spec,
                                            const TaskTemplateIR& task,
                                            const DatasetHandle& dataset,
                                            int32_t block,
                                            std::span<const ChunkBuffer> inputs) {
  if (task.kernel == "particle_value_counts_reduce") {
    std::uint64_t input_bytes = 0;
    for (const auto& input : inputs) {
      input_bytes = checked_add(input_bytes, input.bytes());
    }
    const auto output_bytes = std::max<std::uint64_t>(sizeof(std::uint64_t), input_bytes);
    const auto width = scalar_size(spec.scalar);
    return output_bytes / width + static_cast<std::uint64_t>(output_bytes % width != 0);
  }

  if (task.kernel != "particle_load_field_chunk_f64" &&
      task.kernel != "particle_topk_modes_map") {
    throw BufferContractError(BufferContractReason::kInvalidExtent,
                              "backend-chunk bound is not defined for kernel " + task.kernel);
  }
  if (!dataset.backend) {
    throw BufferContractError(BufferContractReason::kInvalidExtent,
                              "backend-chunk bound requires an initialized dataset backend");
  }
  const auto& root = cached_params_root(task.params_msgpack);
  const auto* particle_type = find_msgpack_map_value(root, "particle_type");
  if (!particle_type || particle_type->type != msgpack::type::STR) {
    throw BufferContractError(BufferContractReason::kInvalidExtent,
                              "backend-chunk bound requires particle_type");
  }
  const auto records = dataset.backend->estimate_particle_chunk_records(
      particle_type->as<std::string>(), block);
  if (!records.has_value()) {
    throw BufferContractError(BufferContractReason::kInvalidExtent,
                              "dataset backend cannot estimate particle chunk records");
  }
  if (task.kernel == "particle_load_field_chunk_f64") {
    return *records;
  }
  return checked_add(
      sizeof(std::uint64_t),
      checked_multiply(*records, sizeof(double) + sizeof(std::int64_t)));
}

struct AmrSubboxPackParams {
  int32_t input_field = -1;
  int32_t input_version = 0;
  int32_t input_step = 0;
  int16_t input_level = 0;
  int32_t halo_cells = 1;
};

AmrSubboxPackParams parse_amr_subbox_pack_params(const TaskTemplateIR& task) {
  const auto& root = cached_params_root(task.params_msgpack);
  AmrSubboxPackParams params;
  if (const auto* value = find_msgpack_map_value(root, "input_field")) {
    params.input_field = value->as<int32_t>();
  }
  if (const auto* value = find_msgpack_map_value(root, "input_version")) {
    params.input_version = value->as<int32_t>();
  }
  if (const auto* value = find_msgpack_map_value(root, "input_step")) {
    params.input_step = value->as<int32_t>();
  }
  if (const auto* value = find_msgpack_map_value(root, "input_level")) {
    params.input_level = value->as<int16_t>();
  }
  if (const auto* value = find_msgpack_map_value(root, "halo_cells")) {
    params.halo_cells = value->as<int32_t>();
  }
  return params;
}

std::uint64_t derive_amr_subbox_pack_capacity(const TaskTemplateIR& task,
                                              const DatasetHandle& dataset,
                                              const RunMeta& meta,
                                              int32_t step,
                                              int16_t level,
                                              int32_t block) {
  // Covers the fixed MessagePack map, keys, rank-four descriptor metadata,
  // and container headers used by amr_subbox_fetch_pack; payload bytes are
  // added separately from the intersecting source chunks below.
  constexpr std::uint64_t kPackedRootMetadataMaxBytes = 64;
  constexpr std::uint64_t kPackedPatchMetadataMaxBytes = 512;
  const auto params = parse_amr_subbox_pack_params(task);
  if (params.input_field < 0) {
    return 0;
  }

  const auto& target_level = meta.steps.at(static_cast<std::size_t>(step))
                                 .levels.at(static_cast<std::size_t>(level));
  const auto& target_box = target_level.boxes.at(static_cast<std::size_t>(block));
  const auto box_lo = [](const BlockBox& box, int axis) {
    return axis == 0 ? box.lo.x : (axis == 1 ? box.lo.y : box.lo.z);
  };
  const auto box_hi = [](const BlockBox& box, int axis) {
    return axis == 0 ? box.hi.x : (axis == 1 ? box.hi.y : box.hi.z);
  };
  const auto cell_edge_at = [](const LevelGeom& geom, int axis, int32_t index) {
    return geom.x0[axis] +
        static_cast<double>(index - geom.index_origin[axis]) * geom.dx[axis];
  };
  const auto coord_to_index_at = [](const LevelGeom& geom, int axis, double coordinate) {
    if (!(geom.dx[axis] > 0.0)) {
      throw BufferContractError(BufferContractReason::kInvalidExtent,
                                "AMR subbox bound requires positive cell spacing");
    }
    return static_cast<int32_t>(std::floor(
               (coordinate - geom.x0[axis]) / geom.dx[axis])) +
        geom.index_origin[axis];
  };

  const int halo = std::max(1, params.halo_cells);
  double query_lo[3] = {0.0, 0.0, 0.0};
  double query_hi[3] = {0.0, 0.0, 0.0};
  for (int axis = 0; axis < 3; ++axis) {
    query_lo[axis] = cell_edge_at(target_level.geom, axis, box_lo(target_box, axis)) -
        static_cast<double>(halo) * target_level.geom.dx[axis];
    query_hi[axis] = cell_edge_at(target_level.geom, axis, box_hi(target_box, axis) + 1) +
        static_cast<double>(halo) * target_level.geom.dx[axis];
  }

  const auto& source_step = meta.steps.at(static_cast<std::size_t>(params.input_step));
  std::uint64_t capacity = kPackedRootMetadataMaxBytes;
  for (int16_t source_level_index = 0;
       source_level_index < static_cast<int16_t>(source_step.levels.size());
       ++source_level_index) {
    const auto& source_level = source_step.levels.at(
        static_cast<std::size_t>(source_level_index));
    int32_t request_lo[3] = {0, 0, 0};
    int32_t request_hi[3] = {-1, -1, -1};
    for (int axis = 0; axis < 3; ++axis) {
      request_lo[axis] = coord_to_index_at(source_level.geom, axis, query_lo[axis]);
      request_hi[axis] = coord_to_index_at(source_level.geom, axis, query_hi[axis]);
    }
    for (int32_t source_block = 0;
         source_block < static_cast<int32_t>(source_level.boxes.size());
         ++source_block) {
      if (source_level_index == params.input_level && source_block == block) {
        continue;
      }
      const auto& source_box = source_level.boxes.at(static_cast<std::size_t>(source_block));
      std::uint64_t requested_cells = 1;
      std::uint64_t source_cells = 1;
      bool intersects = true;
      for (int axis = 0; axis < 3; ++axis) {
        const int32_t intersection_lo = std::max(box_lo(source_box, axis), request_lo[axis]);
        const int32_t intersection_hi = std::min(box_hi(source_box, axis), request_hi[axis]);
        if (intersection_hi < intersection_lo) {
          intersects = false;
          break;
        }
        requested_cells = checked_multiply(
            requested_cells,
            static_cast<std::uint64_t>(intersection_hi - intersection_lo + 1));
        source_cells = checked_multiply(
            source_cells,
            static_cast<std::uint64_t>(box_hi(source_box, axis) - box_lo(source_box, axis) + 1));
      }
      if (!intersects) {
        continue;
      }

      const ChunkRef source_ref{params.input_step, source_level_index, params.input_field,
                                params.input_version, source_block};
      std::uint64_t source_bytes = dataset.estimate_chunk_bytes(source_ref);
      if (source_bytes == 0 && dataset.backend) {
        if (const auto desc = dataset.backend->describe_chunk(source_ref); desc.has_value()) {
          source_bytes = desc->required_bytes();
        }
      }
      if (source_bytes == 0 || source_cells == 0) {
        throw BufferContractError(BufferContractReason::kInvalidExtent,
                                  "dataset backend cannot estimate AMR source chunk bytes");
      }
      const auto bytes_per_cell = source_bytes / source_cells +
          static_cast<std::uint64_t>(source_bytes % source_cells != 0);
      const auto payload_bytes = checked_multiply(requested_cells, bytes_per_cell);
      capacity = checked_add(capacity, kPackedPatchMetadataMaxBytes);
      capacity = checked_add(capacity, payload_bytes);
    }
  }
  return capacity;
}

ResolvedBufferSpec resolve_output_spec(const BufferSpecIR& spec,
                                       const RunMeta& meta,
                                       int32_t step,
                                       int16_t level,
                                       int32_t block,
                                       std::span<const ChunkBuffer> inputs) {
  ResolvedBufferSpec resolved;
  resolved.init = spec.init;
  if (const auto desc = resolve_static_output_desc(spec, meta, step, level, block);
      desc.has_value() && spec.shape_kind != ShapeRuleKind::kDynamic) {
    resolved.desc = *desc;
    return resolved;
  }
  if (spec.shape_kind == ShapeRuleKind::kLikeInput) {
    const auto index = static_cast<std::size_t>(spec.like_input_index);
    if (index >= inputs.size()) {
      throw BufferContractError(BufferContractReason::kInvalidExtent,
                                "like-input output index is out of range");
    }
    const auto& source = inputs[index].desc();
    std::vector<std::uint64_t> extents(
        source.extents.begin(), source.extents.begin() + source.rank);
    resolved.desc = BufferDesc::contiguous(spec.scalar, extents);
    return resolved;
  }
  if (spec.shape_kind != ShapeRuleKind::kDynamic) {
    throw BufferContractError(BufferContractReason::kInvalidExtent,
                              "unable to resolve static output buffer specification");
  }

  std::uint64_t capacity = 0;
  switch (spec.dynamic_upper_bound.kind) {
    case DynamicUpperBoundKind::kLiteral:
      capacity = spec.dynamic_upper_bound.value;
      break;
    case DynamicUpperBoundKind::kLikeInput: {
      const auto index = static_cast<std::size_t>(spec.dynamic_upper_bound.input_index);
      if (index >= inputs.size()) {
        throw BufferContractError(BufferContractReason::kInvalidExtent,
                                  "dynamic like-input index is out of range");
      }
      capacity = inputs[index].desc().element_count();
      break;
    }
    case DynamicUpperBoundKind::kBackendChunk:
    case DynamicUpperBoundKind::kAmrSubboxPack:
      throw BufferContractError(BufferContractReason::kInvalidExtent,
                                "dynamic output bound requires task and dataset context");
  }
  return make_dynamic_output_spec(spec, capacity);
}

void finalize_output_buffer(ChunkBuffer& buffer, const BufferSpecIR& spec) {
  if (spec.shape_kind == ShapeRuleKind::kDynamic) {
    const auto width = scalar_size(spec.scalar);
    if (width == 0 || buffer.data.size() % width != 0) {
      throw BufferContractError(BufferContractReason::kDescriptorStorageMismatch,
                                "dynamic kernel output byte count is not scalar aligned");
    }
    buffer.commit_dynamic_extent(buffer.data.size() / width);
    return;
  }
  buffer.desc().validate(buffer.data.size());
}

std::size_t template_output_storage_bytes(const TaskTemplateIR& tmpl,
                                  const RunMeta& meta,
                                  int32_t block) {
  std::size_t bytes = 0;
  for (const auto& output : tmpl.outputs) {
    bytes += output_spec_bytes(
        output.buffer, meta, tmpl.domain.step, tmpl.domain.level, block);
  }
  return bytes;
}

using ChunkByteMap = std::unordered_map<ChunkRef, std::size_t, ChunkRefHash, ChunkRefEq>;

void add_known_output_storage_bytes(ChunkByteMap& known,
                            const ChunkRef& ref,
                            std::size_t bytes) {
  if (bytes == 0) {
    return;
  }
  known[ref] = bytes;
}

ChunkByteMap build_known_output_storage_bytes(const PlanIR& plan, const RunMeta& meta) {
  ChunkByteMap known;
  for (const auto& stage : plan.stages) {
    for (const auto& tmpl : stage.templates) {
      if (tmpl.outputs.empty()) {
        continue;
      }
      if (tmpl.plane == ExecPlane::Chunk) {
        const auto& level = meta.steps.at(static_cast<std::size_t>(tmpl.domain.step))
                                .levels.at(static_cast<std::size_t>(tmpl.domain.level));
        std::vector<int32_t> blocks;
        if (tmpl.domain.blocks.has_value()) {
          blocks.assign(tmpl.domain.blocks->begin(), tmpl.domain.blocks->end());
        } else {
          const int32_t nblocks = static_cast<int32_t>(level.boxes.size());
          blocks.reserve(static_cast<std::size_t>(nblocks));
          for (int32_t block = 0; block < nblocks; ++block) {
            blocks.push_back(block);
          }
        }
        for (int32_t block : blocks) {
          for (const auto& out : tmpl.outputs) {
            const auto bytes = output_spec_bytes(
                out.buffer, meta, tmpl.domain.step, tmpl.domain.level, block);
            add_known_output_storage_bytes(
                known,
                ChunkRef{tmpl.domain.step, tmpl.domain.level,
                         out.field.field, out.field.version, block},
                bytes);
          }
        }
      } else if (tmpl.plane == ExecPlane::Graph) {
        const auto& params = prepared_graph_reduce(tmpl);
        const int32_t n_groups = graph_reduce_group_count(params);
        for (int32_t group_idx = 0; group_idx < n_groups; ++group_idx) {
          const int32_t out_block = graph_reduce_output_block(params, group_idx);
          for (const auto& out : tmpl.outputs) {
            const auto bytes = output_spec_bytes(
                out.buffer, meta, tmpl.domain.step, tmpl.domain.level, out_block);
            add_known_output_storage_bytes(
                known,
                ChunkRef{tmpl.domain.step, tmpl.domain.level,
                         out.field.field, out.field.version, out_block},
                bytes);
          }
        }
      }
    }
  }
  return known;
}

std::size_t estimate_task_input_ref_bytes(const ChunkByteMap& known_outputs,
                                          const DataService& data,
                                          const ChunkRef& ref) {
  auto it = known_outputs.find(ref);
  if (it != known_outputs.end()) {
    return it->second;
  }
  return data.estimate_host_bytes(ref);
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

  std::vector<hpx::future<ChunkBuffer>> input_futures;
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
                        int16_t level) -> hpx::future<std::vector<ChunkBuffer>> {
    if (tmpl.deps.kind != "FaceNeighbors") {
      return hpx::make_ready_future(std::vector<ChunkBuffer>{});
    }
    const int32_t face_idx = static_cast<int32_t>(face);
    if (!tmpl.deps.faces[face_idx]) {
      return hpx::make_ready_future(std::vector<ChunkBuffer>{});
    }
    if (tmpl.deps.width <= 0) {
      return hpx::make_ready_future(std::vector<ChunkBuffer>{});
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
      return hpx::make_ready_future(std::vector<ChunkBuffer>{});
    }
    auto host_futures = data.get_hosts(refs);
    return hpx::when_all(host_futures).then([](auto&& all) {
      std::vector<ChunkBuffer> out;
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

  std::vector<hpx::future<std::vector<ChunkBuffer>>> neighbor_futures;
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

        std::vector<ChunkBuffer> inputs;
        inputs.reserve(input_pack.size());
        for (auto& f : input_pack) {
          inputs.push_back(f.get());
        }

        NeighborViews nbrs;
        nbrs.input_indices = halo_inputs;
        nbrs.inputs.resize(halo_inputs.size());

        auto assign_face = [](NeighborViews::FieldNeighbors& field, Face face,
                              std::vector<ChunkBuffer>&& views) {
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
        std::vector<ChunkBuffer> outputs;
        outputs.reserve(tmpl.outputs.size());
        for (std::size_t i = 0; i < tmpl.outputs.size(); ++i) {
          const auto& out = tmpl.outputs[i];
          ChunkRef cref{tmpl.domain.step, tmpl.domain.level,
                        out.field.field, out.field.version, block};
          outputs.push_back(data.alloc_host(
              cref,
              resolve_output_spec_for_task(
                  out.buffer, tmpl, execution_context(plan_id).dataset, meta,
                  tmpl.domain.step, tmpl.domain.level, block, inputs)));
        }
        if (log_enabled) {
          end_span(alloc_outputs_event);
        }

        auto inputs_ptr = std::make_shared<std::vector<ChunkBuffer>>(std::move(inputs));
        auto nbrs_ptr = std::make_shared<NeighborViews>(std::move(nbrs));
        auto outputs_ptr = std::make_shared<std::vector<ChunkBuffer>>(std::move(outputs));
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
                finalize_output_buffer((*outputs_ptr)[i], tmpl.outputs[i].buffer);
                log_projection_output_summary(tmpl, block, i, (*outputs_ptr)[i]);
                const auto& out = tmpl.outputs[i];
                ChunkRef cref{tmpl.domain.step, tmpl.domain.level,
                              out.field.field, out.field.version, block};
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

  std::vector<hpx::future<ChunkBuffer>> input_futures;
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
        std::vector<ChunkBuffer> inputs;
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

        TaskEvent alloc_outputs_event;
        if (log_enabled) {
          alloc_outputs_event = start_span(base_event, "alloc_outputs");
        }
        std::vector<ChunkBuffer> outputs;
        outputs.reserve(tmpl.outputs.size());
        for (std::size_t i = 0; i < tmpl.outputs.size(); ++i) {
          const auto& out = tmpl.outputs[i];
          ChunkRef cref{tmpl.domain.step, tmpl.domain.level,
                        out.field.field, out.field.version, out_block};
          outputs.push_back(data.alloc_host(
              cref,
              resolve_output_spec_for_task(
                  out.buffer, tmpl, execution_context(plan_id).dataset, meta,
                  tmpl.domain.step, tmpl.domain.level, out_block, inputs)));
        }
        if (log_enabled) {
          end_span(alloc_outputs_event);
        }

        NeighborViews nbrs;
        auto inputs_ptr = std::make_shared<std::vector<ChunkBuffer>>(std::move(inputs));
        auto nbrs_ptr = std::make_shared<NeighborViews>(std::move(nbrs));
        auto outputs_ptr = std::make_shared<std::vector<ChunkBuffer>>(std::move(outputs));
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
                finalize_output_buffer((*outputs_ptr)[i], tmpl.outputs[i].buffer);
                log_projection_output_summary(tmpl, out_block, i, (*outputs_ptr)[i]);
                const auto& out = tmpl.outputs[i];
                ChunkRef cref{tmpl.domain.step, tmpl.domain.level,
                              out.field.field, out.field.version,
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

struct StorageUnitKey {
  int32_t step = 0;
  int16_t level = 0;
  int32_t version = 0;
  int32_t block = 0;

  bool operator==(const StorageUnitKey& other) const {
    return step == other.step && level == other.level && version == other.version &&
           block == other.block;
  }
};

struct StorageUnitKeyHash {
  std::size_t operator()(const StorageUnitKey& key) const {
    std::size_t h = 0xcbf29ce484222325ull;
    auto mix = [&](auto v) {
      h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };
    mix(key.step);
    mix(key.level);
    mix(key.version);
    mix(key.block);
    return h;
  }
};

std::vector<StorageUnitKey> task_storage_units(const TaskInstance& task) {
  std::vector<StorageUnitKey> units;
  units.reserve(task.input_refs.size());
  for (const auto& ref : task.input_refs) {
    StorageUnitKey key{ref.step, ref.level, ref.version, ref.block};
    if (std::find(units.begin(), units.end(), key) == units.end()) {
      units.push_back(key);
    }
  }
  return units;
}

hpx::future<void> run_stage_partition_impl(int32_t plan_id,
                                           const PlanIR& plan,
                                           const RunMeta& meta,
                                           DataService& data,
                                           AdjacencyService& adj,
                                           std::vector<TaskInstance> tasks,
                                           std::size_t max_active_tasks,
                                           std::size_t max_active_storage_units,
                                           std::size_t max_input_bytes,
                                           std::size_t max_output_storage_bytes) {
  if (tasks.empty()) {
    return hpx::make_ready_future();
  }

  struct ActiveTask {
    hpx::future<void> future;
    std::vector<StorageUnitKey> storage_units;
    std::vector<ChunkRef> input_refs;
    std::size_t input_bytes = 0;
    std::size_t output_storage_bytes = 0;
  };

  struct Runner : std::enable_shared_from_this<Runner> {
    int32_t plan_id = 0;
    const PlanIR* plan = nullptr;
    const RunMeta* meta = nullptr;
    DataService* data = nullptr;
    AdjacencyService* adj = nullptr;
    std::vector<TaskInstance> pending;
    std::size_t max_active_tasks = 1;
    std::size_t max_active_storage_units = 0;
    std::size_t max_input_bytes = 0;
    std::size_t max_output_storage_bytes = 0;
    std::vector<ActiveTask> active;
    std::unordered_map<StorageUnitKey, int32_t, StorageUnitKeyHash> active_storage_units;
    std::size_t active_storage_unit_count = 0;
    std::size_t active_input_bytes = 0;
    std::size_t active_output_storage_bytes = 0;
    hpx::promise<void> done;

    void log_partition_state(const char* reason,
                             std::size_t admitted = 0,
                             std::size_t scanned = 0) const {
      if (!has_event_log()) {
        return;
      }
      std::size_t admissible = 0;
      std::size_t pending_input_bytes = 0;
      for (const auto& task : pending) {
        pending_input_bytes += task.estimated_input_bytes;
        auto units = task_storage_units(task);
        if (can_admit(task, units)) {
          ++admissible;
        }
      }
      const double ts = now_seconds();
      DataEvent event;
      event.op = std::string("executor_partition_state_") + reason;
      event.mode = "local";
      event.status = "end";
      event.ref = ChunkRef{0, 0, -1, 0, -1};
      event.locality = hpx::get_locality_id();
      event.target_locality = event.locality;
      event.worker = static_cast<int32_t>(hpx::get_worker_thread_num());
      event.bytes = admitted;
      event.estimated_bytes = pending_input_bytes;
      event.file_offset = static_cast<int64_t>(std::min<std::size_t>(
          scanned, static_cast<std::size_t>(std::numeric_limits<int64_t>::max())));
      event.comp_start = static_cast<int32_t>(
          std::min<std::size_t>(active_storage_unit_count,
                                static_cast<std::size_t>(std::numeric_limits<int32_t>::max())));
      event.comp_count = static_cast<int32_t>(
          std::min<std::size_t>(max_active_storage_units,
                                static_cast<std::size_t>(std::numeric_limits<int32_t>::max())));
      event.queue_depth = static_cast<int32_t>(
          std::min<std::size_t>(pending.size(),
                                static_cast<std::size_t>(std::numeric_limits<int32_t>::max())));
      event.in_flight = static_cast<int32_t>(
          std::min<std::size_t>(active.size(),
                                static_cast<std::size_t>(std::numeric_limits<int32_t>::max())));
      event.concurrency = static_cast<int32_t>(
          std::min<std::size_t>(max_active_tasks,
                                static_cast<std::size_t>(std::numeric_limits<int32_t>::max())));
      event.in_flight_bytes = active_input_bytes > static_cast<std::size_t>(
                                                     std::numeric_limits<int64_t>::max())
                                  ? std::numeric_limits<int64_t>::max()
                                  : static_cast<int64_t>(active_input_bytes);
      event.byte_limit = max_input_bytes > static_cast<std::size_t>(
                                             std::numeric_limits<int64_t>::max())
                             ? std::numeric_limits<int64_t>::max()
                             : static_cast<int64_t>(max_input_bytes);
      event.ts = ts;
      event.start = ts;
      event.end = ts;
      log_data_event(event);
    }

    hpx::future<void> start() {
      auto out = done.get_future();
      try {
        log_partition_state("start");
        refill();
        wait_next();
      } catch (...) {
        done.set_exception(std::current_exception());
      }
      return out;
    }

    hpx::future<void> launch(const TaskInstance& task) {
      const auto& stage = plan->stages.at(static_cast<std::size_t>(task.stage_idx));
      const auto tmpl = stage.templates.at(static_cast<std::size_t>(task.tmpl_idx));
      if (task.kind == TaskKind::Chunk) {
        return hpx::unwrap(hpx::async([tmpl,
                                       plan_id = plan_id,
                                       stage_idx = task.stage_idx,
                                       tmpl_idx = task.tmpl_idx,
                                       block = task.block_or_group,
                                       meta = meta,
                                       data = data,
                                       adj = adj]() mutable {
          return run_block_task_impl(tmpl, plan_id, stage_idx, tmpl_idx, block, *meta, *data, *adj);
        }));
      }
      return hpx::unwrap(hpx::async([tmpl,
                                     plan_id = plan_id,
                                     stage_idx = task.stage_idx,
                                     tmpl_idx = task.tmpl_idx,
                                     group_idx = task.block_or_group,
                                     meta = meta,
                                     data = data]() mutable {
        return run_graph_task_impl(tmpl, plan_id, stage_idx, tmpl_idx, group_idx, *meta, *data);
      }));
    }

    std::size_t new_storage_unit_count(const std::vector<StorageUnitKey>& units) const {
      std::size_t count = 0;
      for (const auto& unit : units) {
        if (active_storage_units.find(unit) == active_storage_units.end()) {
          ++count;
        }
      }
      return count;
    }

    bool can_admit(const TaskInstance& task, const std::vector<StorageUnitKey>& units) const {
      if (active.size() >= max_active_tasks) {
        return false;
      }
      if (max_active_storage_units == 0 || units.empty()) {
        return can_admit_bytes(task);
      }
      const std::size_t new_units = new_storage_unit_count(units);
      if (active.empty()) {
        return true;
      }
      return active_storage_unit_count + new_units <= max_active_storage_units &&
             can_admit_bytes(task);
    }

    bool can_admit_bytes(const TaskInstance& task) const {
      if (active.empty()) {
        return true;
      }
      if (max_input_bytes > 0 && task.estimated_input_bytes > 0 &&
          active_input_bytes + task.estimated_input_bytes > max_input_bytes) {
        return false;
      }
      if (max_output_storage_bytes > 0 && task.estimated_output_storage_bytes > 0 &&
          active_output_storage_bytes + task.estimated_output_storage_bytes > max_output_storage_bytes) {
        return false;
      }
      return true;
    }

    void add_task_bytes(const TaskInstance& task) {
      active_input_bytes += task.estimated_input_bytes;
      active_output_storage_bytes += task.estimated_output_storage_bytes;
    }

    void remove_task_bytes(const ActiveTask& task) {
      active_input_bytes = task.input_bytes > active_input_bytes
                               ? 0
                               : active_input_bytes - task.input_bytes;
      active_output_storage_bytes = task.output_storage_bytes > active_output_storage_bytes
                                ? 0
                                : active_output_storage_bytes - task.output_storage_bytes;
    }

    void add_storage_units(const std::vector<StorageUnitKey>& units) {
      for (const auto& unit : units) {
        auto [it, inserted] = active_storage_units.emplace(unit, 0);
        if (inserted) {
          ++active_storage_unit_count;
        }
        it->second += 1;
      }
    }

    void remove_storage_units(const std::vector<StorageUnitKey>& units) {
      for (const auto& unit : units) {
        auto it = active_storage_units.find(unit);
        if (it == active_storage_units.end()) {
          continue;
        }
        it->second -= 1;
        if (it->second <= 0) {
          active_storage_units.erase(it);
          if (active_storage_unit_count > 0) {
            --active_storage_unit_count;
          }
        }
      }
    }

    void release_consumed_inputs(const std::vector<ChunkRef>& refs) {
      if (refs.empty()) {
        return;
      }
      auto* local_data = dynamic_cast<DataServiceLocal*>(data);
      if (local_data == nullptr) {
        return;
      }
      local_data->release_consumed_inputs(refs).get();
    }

    void refill() {
      std::size_t admitted_total = 0;
      std::size_t scanned_total = 0;
      log_partition_state("refill_begin");
      while (!pending.empty() && active.size() < max_active_tasks) {
        bool admitted = false;
        for (std::size_t i = 0; i < pending.size(); ++i) {
          ++scanned_total;
          auto units = task_storage_units(pending[i]);
          if (!can_admit(pending[i], units)) {
            continue;
          }
          TaskInstance task = std::move(pending[i]);
          pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(i));
          std::vector<ChunkRef> input_refs = task.input_refs;
          add_storage_units(units);
          add_task_bytes(task);
          active.push_back(ActiveTask{launch(task),
                                      std::move(units),
                                      std::move(input_refs),
                                      task.estimated_input_bytes,
                                      task.estimated_output_storage_bytes});
          ++admitted_total;
          log_partition_state("launch", admitted_total, scanned_total);
          admitted = true;
          break;
        }
        if (!admitted) {
          break;
        }
      }
      log_partition_state("refill_end", admitted_total, scanned_total);
    }

    void wait_next() {
      if (active.empty()) {
        if (pending.empty()) {
          done.set_value();
          return;
        }
        refill();
        if (active.empty()) {
          done.set_exception(std::make_exception_ptr(
              std::runtime_error("streaming partition could not admit pending task")));
          return;
        }
      }

      std::vector<hpx::future<void>> waiting;
      waiting.reserve(active.size());
      for (auto& task : active) {
        waiting.push_back(std::move(task.future));
      }
      hpx::when_any(std::move(waiting))
          .then([self = this->shared_from_this()](auto&& ready) mutable {
            self->handle_ready(std::move(ready));
          });
    }

    void handle_ready(hpx::future<hpx::when_any_result<std::vector<hpx::future<void>>>> ready) {
      try {
        auto result = ready.get();
        if (result.index >= result.futures.size() || result.index >= active.size()) {
          throw std::runtime_error("streaming partition received invalid when_any index");
        }
        for (std::size_t i = 0; i < active.size(); ++i) {
          if (i != result.index) {
            active[i].future = std::move(result.futures[i]);
          }
        }
        result.futures[result.index].get();
        release_consumed_inputs(active[result.index].input_refs);
        remove_storage_units(active[result.index].storage_units);
        remove_task_bytes(active[result.index]);
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(result.index));
        log_partition_state("complete");
        refill();
        wait_next();
      } catch (...) {
        done.set_exception(std::current_exception());
      }
    }
  };

  auto runner = std::make_shared<Runner>();
  runner->plan_id = plan_id;
  runner->plan = &plan;
  runner->meta = &meta;
  runner->data = &data;
  runner->adj = &adj;
  runner->pending = std::move(tasks);
  runner->max_active_tasks = std::max<std::size_t>(1, max_active_tasks);
  runner->max_active_storage_units = max_active_storage_units;
  runner->max_input_bytes = max_input_bytes;
  runner->max_output_storage_bytes = max_output_storage_bytes;
  return runner->start();
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

hpx::future<void> run_stage_partition_remote(int32_t plan_id,
                                             std::vector<TaskInstance> tasks,
                                             int32_t max_active_tasks,
                                             int32_t max_active_storage_units,
                                             std::uint64_t max_input_bytes,
                                             std::uint64_t max_output_storage_bytes) {
  auto ctx = execution_context_shared(plan_id);
  auto data = std::make_shared<DataServiceLocal>(plan_id);
  const std::size_t task_limit =
      static_cast<std::size_t>(std::max<int32_t>(1, max_active_tasks));
  const std::size_t storage_unit_limit =
      static_cast<std::size_t>(std::max<int32_t>(0, max_active_storage_units));
  return run_stage_partition_impl(plan_id,
                                  ctx->plan,
                                  ctx->meta,
                                  *data,
                                  *ctx->adjacency,
                                  std::move(tasks),
                                  task_limit,
                                  storage_unit_limit,
                                  static_cast<std::size_t>(max_input_bytes),
                                  static_cast<std::size_t>(max_output_storage_bytes))
      .then([ctx = std::move(ctx), data = std::move(data)](auto&& done) mutable {
        done.get();
      });
}

}  // namespace

}  // namespace kangaroo

HPX_PLAIN_ACTION(kangaroo::run_block_task_remote, kangaroo_run_block_task_action)
HPX_PLAIN_ACTION(kangaroo::run_graph_task_remote, kangaroo_run_graph_task_action)
HPX_PLAIN_ACTION(kangaroo::run_stage_partition_remote, kangaroo_run_stage_partition_action)

namespace kangaroo {

ResolvedBufferSpec resolve_output_spec_for_task(
    const BufferSpecIR& spec,
    const TaskTemplateIR& task,
    const DatasetHandle& dataset,
    const RunMeta& meta,
    int32_t step,
    int16_t level,
    int32_t block,
    std::span<const ChunkBuffer> inputs) {
  if (spec.shape_kind == ShapeRuleKind::kDynamic) {
    if (spec.dynamic_upper_bound.kind == DynamicUpperBoundKind::kBackendChunk) {
      return make_dynamic_output_spec(
          spec, derive_backend_chunk_capacity(spec, task, dataset, block, inputs));
    }
    if (spec.dynamic_upper_bound.kind == DynamicUpperBoundKind::kAmrSubboxPack) {
      return make_dynamic_output_spec(
          spec, derive_amr_subbox_pack_capacity(task, dataset, meta, step, level, block));
    }
  }
  return resolve_output_spec(spec, meta, step, level, block, inputs);
}

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

ExecutorOptions executor_options_from_environment() {
  auto positive_env_int = [](const char* name, int32_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
      return fallback;
    }
    try {
      int value = std::stoi(raw);
      return value > 0 ? value : fallback;
    } catch (...) {
      return fallback;
    }
  };
  auto positive_env_size = [](const char* name, std::size_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
      return fallback;
    }
    try {
      std::string value(raw);
      char suffix = '\0';
      if (!value.empty()) {
        suffix = value.back();
      }
      std::size_t multiplier = 1;
      if (suffix == 'k' || suffix == 'K') {
        multiplier = 1024;
        value.pop_back();
      } else if (suffix == 'm' || suffix == 'M') {
        multiplier = 1024 * 1024;
        value.pop_back();
      } else if (suffix == 'g' || suffix == 'G') {
        multiplier = 1024ull * 1024ull * 1024ull;
        value.pop_back();
      }
      std::size_t parsed = static_cast<std::size_t>(std::stoull(value));
      return parsed > 0 ? parsed * multiplier : fallback;
    } catch (...) {
      return fallback;
    }
  };

  ExecutorOptions options;
  if (const char* mode = std::getenv("KANGAROO_EXECUTOR_MODE");
      mode != nullptr && *mode != '\0') {
    options.mode = mode;
  }
  options.max_active_tasks_per_locality = positive_env_int(
      "KANGAROO_EXECUTOR_MAX_ACTIVE_TASKS_PER_LOCALITY",
      options.max_active_tasks_per_locality);
  options.max_active_storage_units_per_locality = positive_env_int(
      "KANGAROO_EXECUTOR_MAX_ACTIVE_STORAGE_UNITS_PER_LOCALITY",
      options.max_active_storage_units_per_locality);
  options.max_active_tasks_per_stage = positive_env_int(
      "KANGAROO_EXECUTOR_MAX_ACTIVE_TASKS_PER_STAGE",
      options.max_active_tasks_per_stage);
  options.max_input_bytes_per_locality = positive_env_size(
      "KANGAROO_EXECUTOR_MAX_INPUT_BYTES_PER_LOCALITY",
      options.max_input_bytes_per_locality);
  options.max_output_storage_bytes_per_locality = positive_env_size(
      "KANGAROO_EXECUTOR_MAX_OUTPUT_BYTES_PER_LOCALITY",
      options.max_output_storage_bytes_per_locality);
  return options;
}

Executor::Executor(int32_t plan_id, const RunMeta& meta, DataService& data, AdjacencyService& adj)
    : Executor(plan_id, meta, data, adj, executor_options_from_environment()) {}

Executor::Executor(int32_t plan_id, const RunMeta& meta, DataService& data, AdjacencyService& adj,
                   ExecutorOptions options)
    : plan_id_(plan_id), meta_(meta), data_(data), adj_(adj), options_(std::move(options)) {}

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
  if (options_.mode == "streaming") {
    try {
      register_streaming_input_consumers(plan).get();
    } catch (...) {
      current_plan_ = nullptr;
      throw;
    }
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
  if (options_.mode == "streaming") {
    return run_stage_streaming(stage_idx, stage);
  }
  if (options_.mode != "eager") {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("unsupported KANGAROO_EXECUTOR_MODE: " + options_.mode));
  }
  return run_stage_eager(stage_idx, stage);
}

hpx::future<void> Executor::run_stage_eager(int32_t stage_idx, const StageIR& stage) {
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

std::size_t Executor::streaming_locality_task_window() const {
  std::size_t limit =
      static_cast<std::size_t>(std::max<int32_t>(1, options_.max_active_tasks_per_locality));
  if (options_.max_active_tasks_per_stage > 0) {
    const std::size_t localities = std::max<std::size_t>(1, hpx::find_all_localities().size());
    const std::size_t stage_limit = static_cast<std::size_t>(options_.max_active_tasks_per_stage);
    const std::size_t per_locality_stage_limit =
        std::max<std::size_t>(1, (stage_limit + localities - 1) / localities);
    limit = std::min(limit, per_locality_stage_limit);
  }
  return limit;
}

std::size_t Executor::streaming_locality_storage_unit_window() const {
  if (options_.max_active_storage_units_per_locality > 0) {
    return static_cast<std::size_t>(options_.max_active_storage_units_per_locality);
  }
  return streaming_locality_task_window();
}

std::size_t streaming_partition_action_limit(std::size_t value) {
  return std::min<std::size_t>(
      value,
      static_cast<std::size_t>(std::numeric_limits<int32_t>::max()));
}

std::vector<std::vector<TaskInstance>> partition_tasks_by_locality(
    std::vector<TaskInstance> tasks,
    std::size_t locality_count) {
  std::vector<std::vector<TaskInstance>> partitions(locality_count);
  for (auto& task : tasks) {
    if (task.target_locality < 0 ||
        static_cast<std::size_t>(task.target_locality) >= locality_count) {
      throw std::runtime_error("streaming task target locality out of range");
    }
    partitions[static_cast<std::size_t>(task.target_locality)].push_back(std::move(task));
  }
  return partitions;
}

std::vector<TaskInstance> Executor::expand_stage_tasks(int32_t stage_idx,
                                                       const StageIR& stage) const {
  if (current_plan_ == nullptr) {
    throw std::runtime_error("executor expand_stage_tasks requires active plan");
  }
  const ChunkByteMap known_outputs = build_known_output_storage_bytes(*current_plan_, meta_);
  std::vector<TaskInstance> tasks;
  for (std::size_t tmpl_idx_size = 0; tmpl_idx_size < stage.templates.size(); ++tmpl_idx_size) {
    const int32_t tmpl_idx = static_cast<int32_t>(tmpl_idx_size);
    const auto& tmpl = stage.templates[tmpl_idx_size];
    if (stage.plane == ExecPlane::Chunk) {
      if (tmpl.plane != ExecPlane::Chunk) {
        throw std::runtime_error("chunk stage requires chunk templates");
      }
      const auto& level = meta_.steps.at(tmpl.domain.step).levels.at(tmpl.domain.level);
      std::vector<int32_t> blocks;
      if (tmpl.domain.blocks.has_value()) {
        blocks.assign(tmpl.domain.blocks->begin(), tmpl.domain.blocks->end());
      } else {
        const int32_t nblocks = static_cast<int32_t>(level.boxes.size());
        blocks.reserve(static_cast<std::size_t>(nblocks));
        for (int32_t block = 0; block < nblocks; ++block) {
          blocks.push_back(block);
        }
      }

      for (int32_t block : blocks) {
        TaskInstance task;
        task.kind = TaskKind::Chunk;
        task.stage_idx = stage_idx;
        task.tmpl_idx = tmpl_idx;
        task.block_or_group = block;
        task.input_refs.reserve(tmpl.inputs.size());
        for (std::size_t input_idx = 0; input_idx < tmpl.inputs.size(); ++input_idx) {
          const auto& input = tmpl.inputs[input_idx];
          const auto loc = resolve_input_location(tmpl, input, block);
          ChunkRef ref{loc.step, loc.level, input.field, input.version, loc.block};
          task.input_refs.push_back(ref);
          task.estimated_input_bytes +=
              estimate_task_input_ref_bytes(known_outputs, data_, ref);
        }
        task.output_refs.reserve(tmpl.outputs.size());
        for (const auto& out : tmpl.outputs) {
          task.output_refs.push_back(
              ChunkRef{tmpl.domain.step, tmpl.domain.level,
                       out.field.field, out.field.version, block});
        }
        task.estimated_output_storage_bytes = template_output_storage_bytes(tmpl, meta_, block);
        ChunkRef target_ref;
        if (task.input_refs.empty()) {
          if (task.output_refs.empty()) {
            throw std::runtime_error("templates must specify outputs");
          }
          target_ref = task.output_refs.front();
        } else {
          target_ref = task.input_refs.front();
        }
        task.target_locality = data_.home_rank(target_ref);
        tasks.push_back(std::move(task));
      }
    } else if (stage.plane == ExecPlane::Graph) {
      if (tmpl.plane != ExecPlane::Graph) {
        throw std::runtime_error("graph stage requires graph templates");
      }
      const auto& params = prepared_graph_reduce(tmpl);
      const int32_t n_groups = graph_reduce_group_count(params);
      for (int32_t group_idx = 0; group_idx < n_groups; ++group_idx) {
        const int32_t start = graph_reduce_group_start(params, group_idx);
        const int32_t end = graph_reduce_group_end(params, group_idx);
        const int32_t out_block = graph_reduce_output_block(params, group_idx);
        TaskInstance task;
        task.kind = TaskKind::Graph;
        task.stage_idx = stage_idx;
        task.tmpl_idx = tmpl_idx;
        task.block_or_group = group_idx;
        task.input_refs.reserve(static_cast<std::size_t>(
            std::max<int32_t>(0, end - start) * static_cast<int32_t>(tmpl.inputs.size())));
        for (std::size_t input_idx = 0; input_idx < tmpl.inputs.size(); ++input_idx) {
          const auto& input = tmpl.inputs[input_idx];
          for (int32_t idx = start; idx < end; ++idx) {
            const int32_t block_id = params.input_blocks.empty()
                                         ? (params.input_base + idx)
                                         : params.input_blocks.at(static_cast<std::size_t>(idx));
            int32_t step = tmpl.domain.step;
            int16_t level = tmpl.domain.level;
            if (input.domain.has_value()) {
              step = input.domain->step;
              level = input.domain->level;
            }
            ChunkRef ref{step, level, input.field, input.version, block_id};
            task.input_refs.push_back(ref);
            task.estimated_input_bytes +=
                estimate_task_input_ref_bytes(known_outputs, data_, ref);
          }
        }
        task.output_refs.reserve(tmpl.outputs.size());
        for (const auto& out : tmpl.outputs) {
          task.output_refs.push_back(
              ChunkRef{tmpl.domain.step, tmpl.domain.level,
                       out.field.field, out.field.version, out_block});
        }
        task.estimated_output_storage_bytes = template_output_storage_bytes(tmpl, meta_, out_block);
        if (task.output_refs.empty()) {
          throw std::runtime_error("graph templates must specify outputs");
        }
        task.target_locality = data_.home_rank(task.output_refs.front());
        tasks.push_back(std::move(task));
      }
    } else {
      throw std::runtime_error("unsupported execution plane");
    }
  }
  return tasks;
}

hpx::future<void> Executor::register_streaming_input_consumers(const PlanIR& plan) const {
  auto* local_data = dynamic_cast<DataServiceLocal*>(&data_);
  if (local_data == nullptr) {
    return hpx::make_ready_future();
  }

  std::unordered_map<ChunkRef, std::int64_t, ChunkRefHash, ChunkRefEq> counts;
  for (std::size_t stage_idx = 0; stage_idx < plan.stages.size(); ++stage_idx) {
    auto tasks = expand_stage_tasks(static_cast<int32_t>(stage_idx), plan.stages[stage_idx]);
    for (const auto& task : tasks) {
      for (const auto& ref : task.input_refs) {
        counts[ref] += 1;
      }
    }
  }

  std::vector<ChunkConsumerCount> consumer_counts;
  consumer_counts.reserve(counts.size());
  for (const auto& [ref, count] : counts) {
    if (count > 0) {
      consumer_counts.push_back(ChunkConsumerCount{ref, count});
    }
  }
  return local_data->register_input_consumers(consumer_counts);
}

hpx::future<void> Executor::run_stage_streaming(int32_t stage_idx, const StageIR& stage) {
  auto tasks = expand_stage_tasks(stage_idx, stage);
  if (tasks.empty()) {
    return hpx::make_ready_future();
  }

  if (current_plan_ == nullptr) {
    throw std::runtime_error("executor run_stage_streaming requires active plan");
  }

  const auto localities = hpx::find_all_localities();
  const std::size_t locality_count = std::max<std::size_t>(1, localities.size());
  auto partitions = partition_tasks_by_locality(std::move(tasks), locality_count);
  const int here = hpx::get_locality_id();
  const std::size_t task_limit = streaming_locality_task_window();
  const std::size_t storage_unit_limit = streaming_locality_storage_unit_window();
  const int32_t action_task_limit =
      static_cast<int32_t>(streaming_partition_action_limit(task_limit));
  const int32_t action_storage_unit_limit =
      static_cast<int32_t>(streaming_partition_action_limit(storage_unit_limit));
  const std::uint64_t action_input_byte_limit =
      static_cast<std::uint64_t>(options_.max_input_bytes_per_locality);
  const std::uint64_t action_output_byte_limit =
      static_cast<std::uint64_t>(options_.max_output_storage_bytes_per_locality);

  std::vector<hpx::future<void>> partition_futures;
  partition_futures.reserve(locality_count);
  for (std::size_t target = 0; target < partitions.size(); ++target) {
    if (partitions[target].empty()) {
      continue;
    }
    if (static_cast<int>(target) == here) {
      partition_futures.push_back(run_stage_partition_impl(plan_id_,
                                                           *current_plan_,
                                                           meta_,
                                                           data_,
                                                           adj_,
                                                           std::move(partitions[target]),
                                                           task_limit,
                                                           storage_unit_limit,
                                                           options_.max_input_bytes_per_locality,
                                                           options_.max_output_storage_bytes_per_locality));
      continue;
    }
    partition_futures.push_back(
        hpx::async<::kangaroo_run_stage_partition_action>(localities.at(target),
                                                          plan_id_,
                                                          std::move(partitions[target]),
                                                          action_task_limit,
                                                          action_storage_unit_limit,
                                                          action_input_byte_limit,
                                                          action_output_byte_limit));
  }

  if (partition_futures.empty()) {
    return hpx::make_ready_future();
  }
  return hpx::when_all(partition_futures).then([](auto&& ready_partitions) {
    auto partition_results = ready_partitions.get();
    for (auto& partition : partition_results) {
      partition.get();
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
    target_ref = ChunkRef{tmpl.domain.step, tmpl.domain.level,
                          tmpl.outputs.front().field.field,
                          tmpl.outputs.front().field.version, block};
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
  ChunkRef cref{tmpl.domain.step, tmpl.domain.level,
                out.field.field, out.field.version, out_block};

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
