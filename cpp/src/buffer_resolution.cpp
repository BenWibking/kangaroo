#include "kangaroo/buffer_resolution.hpp"

#include "kangaroo/plan_ir.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace kangaroo {

namespace {

std::size_t checked_positive_extent(int32_t lo, int32_t hi) {
  if (hi < lo) return 0;
  return static_cast<std::size_t>(hi - lo + 1);
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

BufferResolution resolved_static(const BufferSpecIR& spec, BufferDesc desc) {
  BufferResolution out;
  out.allocation.desc = desc;
  out.allocation.init = spec.init;
  out.facts.payload_bytes = static_cast<std::size_t>(desc.required_bytes());
  out.facts.storage_bytes = out.facts.payload_bytes;
  out.facts.storage_known = true;
  out.facts.desc = std::move(desc);
  out.facts.element_capacity = out.facts.desc->element_count();
  return out;
}

BufferResolution resolved_dynamic(const BufferSpecIR& spec, std::uint64_t capacity) {
  const std::array<std::uint64_t, 1> capacity_extent{capacity};
  auto capacity_desc = BufferDesc::contiguous(spec.scalar, capacity_extent);
  BufferResolution out;
  out.allocation.init = spec.init;
  out.allocation.desc = capacity_desc;
  out.allocation.desc.extents[0] = 0;
  out.allocation.dynamic_capacity_elements = capacity;
  out.facts.payload_bytes = static_cast<std::size_t>(capacity_desc.required_bytes());
  out.facts.storage_bytes = out.facts.payload_bytes;
  out.facts.storage_known = true;
  out.facts.desc = capacity_desc;
  out.facts.element_capacity = capacity;
  return out;
}

}  // namespace

BufferFacts buffer_facts(const ChunkBuffer& buffer) {
  BufferFacts facts;
  facts.payload_bytes = buffer.bytes();
  facts.storage_bytes = buffer.resident_bytes();
  facts.storage_known = true;
  facts.desc = buffer.desc();
  facts.element_capacity = buffer.capacity_bytes() / scalar_size(buffer.desc().scalar);
  return facts;
}

std::optional<BufferResolution> try_resolve_buffer_spec(
    const BufferSpecIR& spec,
    const TaskTemplateIR& task,
    const DataService& data,
    const RunMeta& meta,
    int32_t step,
    int16_t level,
    int32_t block,
    std::size_t output_index,
    std::span<const BufferFacts> inputs) {
  if (spec.shape_kind == ShapeRuleKind::kFixed) {
    return resolved_static(spec, BufferDesc::contiguous(spec.scalar, spec.fixed_extents));
  }
  if (spec.shape_kind == ShapeRuleKind::kBlock) {
    const auto xyz = block_extents(meta, step, level, block);
    if (spec.block_components == 1) {
      return resolved_static(spec, BufferDesc::runtime_grid(spec.scalar, xyz));
    }
    const std::array<std::uint64_t, 4> extents{
        xyz[0], xyz[1], xyz[2], spec.block_components};
    return resolved_static(spec, BufferDesc::contiguous(spec.scalar, extents));
  }
  if (spec.shape_kind == ShapeRuleKind::kLikeInput) {
    const auto index = static_cast<std::size_t>(spec.like_input_index);
    if (index >= inputs.size() || !inputs[index].desc.has_value()) return std::nullopt;
    const auto& source = *inputs[index].desc;
    std::vector<std::uint64_t> extents(
        source.extents.begin(), source.extents.begin() + source.rank);
    return resolved_static(spec, BufferDesc::contiguous(spec.scalar, extents));
  }
  if (spec.shape_kind != ShapeRuleKind::kDynamic) return std::nullopt;

  std::optional<std::uint64_t> capacity;
  switch (spec.dynamic_upper_bound.kind) {
    case DynamicUpperBoundKind::kLiteral:
      capacity = spec.dynamic_upper_bound.value;
      break;
    case DynamicUpperBoundKind::kLikeInput: {
      const auto index = static_cast<std::size_t>(spec.dynamic_upper_bound.input_index);
      if (index >= inputs.size()) return std::nullopt;
      capacity = inputs[index].element_capacity;
      break;
    }
    case DynamicUpperBoundKind::kBackendChunk:
    case DynamicUpperBoundKind::kAmrSubboxPack:
      if (!task.dynamic_output_bound) return std::nullopt;
      capacity = (*task.dynamic_output_bound)(DynamicOutputBoundContext{
          spec.scalar, meta, data, step, level, block, output_index, inputs,
          task.prepared_params_type, task.prepared_params});
      break;
  }
  if (!capacity.has_value()) return std::nullopt;
  return resolved_dynamic(spec, *capacity);
}

ResolvedBufferSpec resolve_output_spec_for_task(
    const BufferSpecIR& spec,
    const TaskTemplateIR& task,
    const DataService& data,
    const RunMeta& meta,
    int32_t step,
    int16_t level,
    int32_t block,
    std::size_t output_index,
    std::span<const ChunkBuffer> inputs) {
  std::vector<BufferFacts> facts;
  facts.reserve(inputs.size());
  for (const auto& input : inputs) facts.push_back(buffer_facts(input));
  auto resolved = try_resolve_buffer_spec(
      spec, task, data, meta, step, level, block, output_index, facts);
  if (!resolved.has_value()) {
    throw BufferContractError(BufferContractReason::kInvalidExtent,
                              "unable to resolve output Buffer Specification");
  }
  return std::move(resolved->allocation);
}

std::optional<std::uint64_t> estimate_amr_subbox_pack_capacity(
    const DynamicOutputBoundContext& context,
    const AmrSubboxPackParams& params) {
  constexpr std::uint64_t kPackedRootMetadataMaxBytes = 64;
  constexpr std::uint64_t kPackedPatchMetadataMaxBytes = 512;
  if (params.input_field < 0) return 0;

  const auto& target_level = context.meta.steps.at(static_cast<std::size_t>(context.step))
                                 .levels.at(static_cast<std::size_t>(context.level));
  const auto& target_box = target_level.boxes.at(static_cast<std::size_t>(context.block));
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
    return static_cast<int32_t>(
               std::floor((coordinate - geom.x0[axis]) / geom.dx[axis])) +
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

  const auto& source_step = context.meta.steps.at(static_cast<std::size_t>(params.input_step));
  std::uint64_t capacity = kPackedRootMetadataMaxBytes;
  for (int16_t source_level_index = 0;
       source_level_index < static_cast<int16_t>(source_step.levels.size());
       ++source_level_index) {
    const auto& source_level = source_step.levels.at(static_cast<std::size_t>(source_level_index));
    int32_t request_lo[3] = {0, 0, 0};
    int32_t request_hi[3] = {-1, -1, -1};
    for (int axis = 0; axis < 3; ++axis) {
      request_lo[axis] = coord_to_index_at(source_level.geom, axis, query_lo[axis]);
      request_hi[axis] = coord_to_index_at(source_level.geom, axis, query_hi[axis]);
    }
    for (int32_t source_block = 0;
         source_block < static_cast<int32_t>(source_level.boxes.size());
         ++source_block) {
      if (source_level_index == params.input_level && source_block == context.block) continue;
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
      if (!intersects) continue;

      const ChunkRef source_ref{params.input_step, source_level_index, params.input_field,
                                params.input_version, source_block};
      std::uint64_t source_bytes = context.data.estimate_host_bytes(source_ref);
      if (source_bytes == 0) {
        const auto source_desc = context.data.describe_host(source_ref);
        if (source_desc.has_value()) source_bytes = source_desc->required_bytes();
      }
      if (source_bytes == 0 || source_cells == 0) {
        throw BufferContractError(
            BufferContractReason::kInvalidExtent,
            "cannot estimate AMR source Chunk Buffer for step=" +
                std::to_string(source_ref.step) + " level=" +
                std::to_string(source_ref.level) + " field=" +
                std::to_string(source_ref.field) + " block=" +
                std::to_string(source_ref.block) + " input_level=" +
                std::to_string(params.input_level) + " target_block=" +
                std::to_string(context.block));
      }
      const auto bytes_per_cell = source_bytes / source_cells +
          static_cast<std::uint64_t>(source_bytes % source_cells != 0);
      capacity = checked_add(capacity, kPackedPatchMetadataMaxBytes);
      capacity = checked_add(capacity, checked_multiply(requested_cells, bytes_per_cell));
    }
  }
  return capacity;
}

}  // namespace kangaroo
