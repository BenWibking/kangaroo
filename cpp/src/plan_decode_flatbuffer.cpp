#include "kangaroo/plan_decode.hpp"

#include "plan_generated.h"

#include <flatbuffers/verifier.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace kangaroo {
namespace {

namespace fb = ::analysis::fb;

template <typename To, typename From>
std::vector<To> copy_vector(const std::vector<From> &input) {
  return std::vector<To>(input.begin(), input.end());
}

template <typename To, std::size_t N, typename From>
std::array<To, N> copy_array(const std::vector<From> &input, const char *name) {
  if (input.size() != N) {
    throw std::runtime_error(std::string(name) + " must contain " +
                             std::to_string(N) + " values");
  }
  std::array<To, N> output{};
  for (std::size_t i = 0; i < N; ++i)
    output[i] = static_cast<To>(input[i]);
  return output;
}

ExecPlane decode_plane(fb::ExecPlane plane) {
  switch (plane) {
  case fb::ExecPlane::Chunk:
    return ExecPlane::Chunk;
  case fb::ExecPlane::Graph:
    return ExecPlane::Graph;
  case fb::ExecPlane::Mixed:
    return ExecPlane::Mixed;
  }
  throw std::runtime_error("unknown execution plane");
}

DomainIR decode_domain(const fb::DomainT &input) {
  DomainIR output;
  output.step = input.step;
  output.level = input.level;
  if (!input.all_blocks)
    output.blocks = input.blocks;
  return output;
}

FieldRefIR decode_field_ref(const fb::FieldRefT &input) {
  FieldRefIR output;
  output.field = input.field;
  output.version = input.version;
  if (input.domain)
    output.domain = decode_domain(*input.domain);
  return output;
}

ScalarType decode_scalar(fb::ScalarType scalar) {
  switch (scalar) {
  case fb::ScalarType::Opaque:
    return ScalarType::kOpaque;
  case fb::ScalarType::U8:
    return ScalarType::kU8;
  case fb::ScalarType::I64:
    return ScalarType::kI64;
  case fb::ScalarType::F32:
    return ScalarType::kF32;
  case fb::ScalarType::F64:
    return ScalarType::kF64;
  }
  throw std::runtime_error("unknown buffer scalar type");
}

BufferSpecIR decode_buffer_spec(const fb::BufferSpecT &input) {
  BufferSpecIR output;
  output.scalar = decode_scalar(input.scalar);
  output.init = input.init == fb::InitPolicy::Zero ? InitPolicy::kZero
                                                   : InitPolicy::kUninitialized;
  switch (input.shape_kind) {
  case fb::ShapeRuleKind::Block:
    output.shape_kind = ShapeRuleKind::kBlock;
    output.block_components = input.block_components;
    if (output.block_components == 0)
      throw std::runtime_error("block components must be positive");
    break;
  case fb::ShapeRuleKind::Fixed:
    output.shape_kind = ShapeRuleKind::kFixed;
    output.fixed_extents = input.fixed_extents;
    if (output.fixed_extents.empty() ||
        output.fixed_extents.size() > kMaxBufferRank) {
      throw std::runtime_error("fixed buffer rank must be between 1 and 4");
    }
    for (auto extent : output.fixed_extents) {
      if (extent == 0)
        throw std::runtime_error("fixed buffer extents must be positive");
    }
    break;
  case fb::ShapeRuleKind::LikeInput:
    output.shape_kind = ShapeRuleKind::kLikeInput;
    output.like_input_index = input.like_input_index;
    if (output.like_input_index < 0)
      throw std::runtime_error("like-input index must be non-negative");
    break;
  case fb::ShapeRuleKind::Dynamic: {
    output.shape_kind = ShapeRuleKind::kDynamic;
    if (!input.dynamic_upper_bound)
      throw std::runtime_error("dynamic buffer requires an upper bound");
    const auto &bound = *input.dynamic_upper_bound;
    output.dynamic_upper_bound.value = bound.value;
    output.dynamic_upper_bound.input_index = bound.input_index;
    switch (bound.kind) {
    case fb::DynamicUpperBoundKind::Literal:
      output.dynamic_upper_bound.kind = DynamicUpperBoundKind::kLiteral;
      break;
    case fb::DynamicUpperBoundKind::LikeInput:
      output.dynamic_upper_bound.kind = DynamicUpperBoundKind::kLikeInput;
      break;
    case fb::DynamicUpperBoundKind::BackendChunk:
      output.dynamic_upper_bound.kind = DynamicUpperBoundKind::kBackendChunk;
      break;
    case fb::DynamicUpperBoundKind::AmrSubboxPack:
      output.dynamic_upper_bound.kind = DynamicUpperBoundKind::kAmrSubboxPack;
      break;
    }
    if (output.dynamic_upper_bound.input_index < -1) {
      throw std::runtime_error(
          "dynamic upper-bound input index must be non-negative");
    }
    break;
  }
  }
  if (output.scalar == ScalarType::kOpaque &&
      output.shape_kind == ShapeRuleKind::kBlock) {
    throw std::runtime_error("opaque buffers cannot use block shape");
  }
  if (output.scalar == ScalarType::kOpaque &&
      output.shape_kind == ShapeRuleKind::kFixed &&
      output.fixed_extents.size() != 1) {
    throw std::runtime_error("opaque fixed buffers must have rank 1");
  }
  return output;
}

GraphReduceSpecIR decode_graph_reduce(const fb::GraphReduceSpecT &input) {
  GraphReduceSpecIR output;
  output.fan_in = input.fan_in;
  output.num_inputs = input.num_inputs;
  output.input_base = input.input_base;
  output.output_base = input.output_base;
  output.input_blocks = input.input_blocks;
  output.output_blocks = input.output_blocks;
  output.group_offsets = input.group_offsets;
  if (output.fan_in < 1 || output.num_inputs < 1) {
    throw std::runtime_error(
        "graph reduction fan_in and num_inputs must be positive");
  }
  if (!output.input_blocks.empty() &&
      output.input_blocks.size() !=
          static_cast<std::size_t>(output.num_inputs)) {
    throw std::runtime_error(
        "graph reduction input_blocks must match num_inputs");
  }
  std::size_t groups = static_cast<std::size_t>(
      (output.num_inputs + output.fan_in - 1) / output.fan_in);
  if (!output.group_offsets.empty()) {
    if (output.group_offsets.front() != 0 ||
        output.group_offsets.back() != output.num_inputs) {
      throw std::runtime_error(
          "graph reduction group_offsets must span num_inputs");
    }
    for (std::size_t i = 1; i < output.group_offsets.size(); ++i) {
      if (output.group_offsets[i] <= output.group_offsets[i - 1]) {
        throw std::runtime_error(
            "graph reduction group_offsets must be strictly increasing");
      }
    }
    groups = output.group_offsets.size() - 1;
  }
  if (!output.output_blocks.empty() && output.output_blocks.size() != groups) {
    throw std::runtime_error(
        "graph reduction output_blocks must match group count");
  }
  return output;
}

KernelParamsIR decode_params(const fb::KernelParamsUnion &input) {
  switch (input.type) {
  case fb::KernelParams::NONE:
    return NoKernelParamsIR{};
  case fb::KernelParams::AmrSubboxPackParams: {
    const auto &p = *input.AsAmrSubboxPackParams();
    return AmrSubboxPackParams{p.input_field, p.input_version, p.input_step,
                               p.input_level, p.halo_cells};
  }
  case fb::KernelParams::GradStencilParams: {
    const auto &p = *input.AsGradStencilParams();
    return GradStencilParams{p.input_field, p.input_version, p.input_step,
                             p.input_level, p.stencil_radius};
  }
  case fb::KernelParams::PlotfileLoadParams: {
    const auto &p = *input.AsPlotfileLoadParams();
    return PlotfileLoadParams{p.plotfile, p.level, p.comp};
  }
  case fb::KernelParams::FluxSurfaceParams: {
    const auto &p = *input.AsFluxSurfaceParams();
    FluxSurfaceParams out;
    out.radii = p.radii;
    out.radius_indices = p.radius_indices;
    out.temperature_bins = p.temperature_bins;
    out.num_radii = static_cast<std::size_t>(p.num_radii);
    out.gamma = p.gamma;
    return out;
  }
  case fb::KernelParams::CylindricalFluxParams: {
    const auto &p = *input.AsCylindricalFluxParams();
    CylindricalFluxParams out;
    out.radius = p.radius;
    out.heights = p.heights;
    out.height_indices = p.height_indices;
    out.temperature_bins = p.temperature_bins;
    out.num_heights = static_cast<std::size_t>(p.num_heights);
    out.gamma = p.gamma;
    return out;
  }
  case fb::KernelParams::UniformSliceCellParams: {
    const auto &p = *input.AsUniformSliceCellParams();
    UniformSliceCellParams out;
    out.axis = p.axis;
    out.coord = p.coord;
    out.plane_index = p.plane_index;
    out.has_plane_index = p.has_plane_index;
    out.rect = copy_array<double, 4>(p.rect, "slice rect");
    out.resolution = copy_array<int32_t, 2>(p.resolution, "slice resolution");
    return out;
  }
  case fb::KernelParams::UniformProjectionParams: {
    const auto &p = *input.AsUniformProjectionParams();
    UniformProjectionParams out;
    out.axis = p.axis;
    out.axis_bounds =
        copy_array<double, 2>(p.axis_bounds, "projection axis bounds");
    out.rect = copy_array<double, 4>(p.rect, "projection rect");
    out.resolution =
        copy_array<int32_t, 2>(p.resolution, "projection resolution");
    return out;
  }
  case fb::KernelParams::FieldExprParams: {
    const auto &p = *input.AsFieldExprParams();
    return FieldExprParams{p.expression, p.variables};
  }
  case fb::KernelParams::UniformSliceParams: {
    const auto &p = *input.AsUniformSliceParams();
    UniformSliceParams out;
    out.axis = p.axis;
    out.coord = p.coord;
    out.rect = copy_array<double, 4>(p.rect, "slice rect");
    out.resolution = copy_array<int32_t, 2>(p.resolution, "slice resolution");
    return out;
  }
  case fb::KernelParams::Histogram1DParams: {
    const auto &p = *input.AsHistogram1DParams();
    Histogram1DParams out;
    out.range = copy_array<double, 2>(p.bounds, "histogram range");
    out.bins = p.bins;
    return out;
  }
  case fb::KernelParams::Histogram2DParams: {
    const auto &p = *input.AsHistogram2DParams();
    Histogram2DParams out;
    out.x_range = copy_array<double, 2>(p.x_range, "histogram x range");
    out.y_range = copy_array<double, 2>(p.y_range, "histogram y range");
    out.bins = copy_array<int32_t, 2>(p.bins, "histogram bins");
    out.weight_mode = p.weight_mode;
    return out;
  }
  case fb::KernelParams::ParticleFieldParams: {
    const auto &p = *input.AsParticleFieldParams();
    return ParticleFieldParams{p.particle_type, p.field_name};
  }
  case fb::KernelParams::ParticleCicGridParams: {
    const auto &p = *input.AsParticleCicGridParams();
    ParticleCicGridParams out;
    out.particle_type = p.particle_type;
    out.level_index = p.level_index;
    out.axis = p.axis;
    out.axis_bounds =
        copy_array<double, 2>(p.axis_bounds, "particle CIC axis bounds");
    out.mass_max =
        p.has_mass_max ? p.mass_max : std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  case fb::KernelParams::ParticleCicProjectionParams: {
    const auto &p = *input.AsParticleCicProjectionParams();
    ParticleCicProjectionParams out;
    out.particle_type = p.particle_type;
    out.level_index = p.level_index;
    out.axis = p.axis;
    out.axis_bounds =
        copy_array<double, 2>(p.axis_bounds, "particle CIC axis bounds");
    out.rect = copy_array<double, 4>(p.rect, "particle CIC rect");
    out.resolution =
        copy_array<int32_t, 2>(p.resolution, "particle CIC resolution");
    out.mass_max =
        p.has_mass_max ? p.mass_max : std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  case fb::KernelParams::ScalarParams:
    return ScalarParams{input.AsScalarParams()->scalar};
  case fb::KernelParams::ValuesParams:
    return ValuesParams{input.AsValuesParams()->values};
  case fb::KernelParams::FiniteOnlyParams:
    return FiniteOnlyParams{input.AsFiniteOnlyParams()->finite_only};
  case fb::KernelParams::ParticleHistogramParams: {
    const auto &p = *input.AsParticleHistogramParams();
    return ParticleHistogramParams{p.edges, p.density};
  }
  case fb::KernelParams::TopKModesParams:
    return TopKModesParams{input.AsTopKModesParams()->k};
  case fb::KernelParams::SliceFinalizeParams:
    return SliceFinalizeParams{input.AsSliceFinalizeParams()->pixel_area};
  }
  throw std::runtime_error("unknown kernel parameter type");
}

} // namespace

PlanIR decode_plan_flatbuffer(std::span<const std::uint8_t> payload) {
  if (payload.empty())
    throw std::runtime_error("typed plan payload is empty");
  flatbuffers::Verifier verifier(payload.data(), payload.size());
  if (!fb::VerifyPlanBuffer(verifier))
    throw std::runtime_error("invalid typed plan FlatBuffer");
  std::unique_ptr<fb::PlanT> input(fb::GetPlan(payload.data())->UnPack());
  if (!input)
    throw std::runtime_error("could not unpack typed plan");

  PlanIR plan;
  plan.shared_covered_boxes.reserve(input->shared_covered_boxes.size());
  for (const auto &list : input->shared_covered_boxes) {
    CoveredBoxListIR boxes;
    if (list) {
      boxes.reserve(list->boxes.size());
      for (const auto &box : list->boxes) {
        if (!box)
          throw std::runtime_error("covered box is missing");
        boxes.push_back(
            CoveredBoxIR{copy_array<int32_t, 3>(box->lo, "covered box lo"),
                         copy_array<int32_t, 3>(box->hi, "covered box hi")});
      }
    }
    plan.shared_covered_boxes.push_back(std::move(boxes));
  }

  plan.stages.reserve(input->stages.size());
  for (const auto &stage_in : input->stages) {
    if (!stage_in)
      throw std::runtime_error("plan stage is missing");
    StageIR stage;
    stage.name = stage_in->name;
    stage.plane = decode_plane(stage_in->plane);
    stage.after = stage_in->after;
    stage.templates.reserve(stage_in->templates.size());
    for (const auto &task_in : stage_in->templates) {
      if (!task_in || !task_in->domain || !task_in->deps)
        throw std::runtime_error("task template is incomplete");
      TaskTemplateIR task;
      task.name = task_in->name;
      task.plane = decode_plane(task_in->plane);
      task.kernel = task_in->kernel;
      task.domain = decode_domain(*task_in->domain);
      for (const auto &ref : task_in->inputs) {
        if (!ref)
          throw std::runtime_error("input field reference is missing");
        task.inputs.push_back(decode_field_ref(*ref));
      }
      for (const auto &ref : task_in->outputs) {
        if (!ref || !ref->buffer)
          throw std::runtime_error("output reference is incomplete");
        task.outputs.push_back(
            OutputRefIR{FieldRefIR{ref->field, ref->version, std::nullopt},
                        decode_buffer_spec(*ref->buffer)});
      }
      task.deps.kind = task_in->deps->kind == fb::DepKind::FaceNeighbors
                           ? "FaceNeighbors"
                           : "None";
      task.deps.width = task_in->deps->width;
      if (task_in->deps->faces.size() != 6)
        throw std::runtime_error("faces must contain 6 values");
      for (std::size_t i = 0; i < 6; ++i)
        task.deps.faces[i] = task_in->deps->faces[i];
      task.deps.halo_inputs = task_in->deps->halo_inputs;
      task.covered_boxes_ref = task_in->covered_boxes_ref;
      if (task.covered_boxes_ref < -1 ||
          task.covered_boxes_ref >=
              static_cast<int32_t>(plan.shared_covered_boxes.size())) {
        throw std::runtime_error("covered_boxes_ref out of range");
      }
      task.params = decode_params(task_in->params);
      if (task_in->graph_reduce)
        task.graph_reduce = decode_graph_reduce(*task_in->graph_reduce);
      if (task.plane == ExecPlane::Graph && !task.graph_reduce)
        throw std::runtime_error(
            "graph template requires graph reduction topology");
      if (task.deps.kind == "FaceNeighbors") {
        if (task.inputs.empty())
          throw std::runtime_error(
              "FaceNeighbors deps require at least one input field");
        for (int32_t index : task.deps.halo_inputs) {
          if (index < 0 || index >= static_cast<int32_t>(task.inputs.size()))
            throw std::runtime_error("halo_inputs index out of range");
        }
      }
      stage.templates.push_back(std::move(task));
    }
    plan.stages.push_back(std::move(stage));
  }
  return plan;
}

} // namespace kangaroo
