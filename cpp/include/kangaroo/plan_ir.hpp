#pragma once

#include "kangaroo/kernel.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <typeindex>
#include <vector>

#include <hpx/datastructures/serialization/optional.hpp>

namespace kangaroo {

enum class ExecPlane : uint8_t { Chunk = 0, Graph = 1, Mixed = 2 };

struct DomainIR {
  int32_t step = 0;
  int16_t level = 0;
  std::optional<std::vector<int32_t>> blocks;  // nullopt => all blocks on level

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& step& level& blocks;
  }
};

struct FieldRefIR {
  int32_t field = 0;
  int32_t version = 0;
  std::optional<DomainIR> domain;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& field& version& domain;
  }
};

struct DepRuleIR {
  std::string kind{"None"};  // "None" or "FaceNeighbors"
  int32_t width = 0;
  bool faces[6] = {true, true, true, true, true, true};
  std::vector<int32_t> halo_inputs;  // input indices that require neighbor halos

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& kind& width& halo_inputs;
    for (auto& face : faces) {
      ar& face;
    }
  }
};

struct GraphReduceSpecIR {
  int32_t fan_in = 1;
  int32_t num_inputs = 0;
  int32_t input_base = 0;
  int32_t output_base = 0;
  std::vector<int32_t> input_blocks;
  std::vector<int32_t> output_blocks;
  std::vector<int32_t> group_offsets;
};

struct CoveredBoxIR {
  std::array<int32_t, 3> lo{0, 0, 0};
  std::array<int32_t, 3> hi{0, 0, 0};

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    for (auto& v : lo) {
      ar& v;
    }
    for (auto& v : hi) {
      ar& v;
    }
  }
};

using CoveredBoxListIR = std::vector<CoveredBoxIR>;

enum class ShapeRuleKind : std::uint8_t { kBlock = 0, kFixed = 1, kLikeInput = 2, kDynamic = 3 };
enum class DynamicUpperBoundKind : std::uint8_t {
  kLiteral = 0,
  kLikeInput = 1,
  kBackendChunk = 2,
  kAmrSubboxPack = 3,
};

struct DynamicUpperBoundIR {
  DynamicUpperBoundKind kind = DynamicUpperBoundKind::kLiteral;
  std::uint64_t value = 0;
  std::int32_t input_index = -1;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& kind& value& input_index;
  }
};

struct BufferSpecIR {
  ScalarType scalar = ScalarType::kOpaque;
  ShapeRuleKind shape_kind = ShapeRuleKind::kFixed;
  std::vector<std::uint64_t> fixed_extents;
  std::uint32_t block_components = 1;
  std::int32_t like_input_index = -1;
  DynamicUpperBoundIR dynamic_upper_bound;
  InitPolicy init = InitPolicy::kUninitialized;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& scalar& shape_kind& fixed_extents& block_components& like_input_index&
        dynamic_upper_bound& init;
  }
};

struct OutputRefIR {
  FieldRefIR field;
  BufferSpecIR buffer;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& field& buffer;
  }
};

struct TaskTemplateIR {
  std::string name;
  ExecPlane plane = ExecPlane::Chunk;
  std::string kernel;
  DomainIR domain;
  std::vector<FieldRefIR> inputs;
  std::vector<OutputRefIR> outputs;
  DepRuleIR deps;
  int32_t covered_boxes_ref = -1;
  std::vector<std::uint8_t> params_msgpack;
  std::shared_ptr<const KernelFn> kernel_fn;           // locality-local prepared metadata
  std::optional<GraphReduceSpecIR> graph_reduce;       // locality-local prepared metadata
  std::shared_ptr<const void> prepared_params;         // locality-local prepared metadata
  std::type_index prepared_params_type{typeid(void)};  // locality-local prepared metadata

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& name& plane& kernel& domain& inputs& outputs& deps& covered_boxes_ref&
        params_msgpack;
  }
};

struct StageIR {
  std::string name;
  ExecPlane plane = ExecPlane::Chunk;
  std::vector<int32_t> after;
  std::vector<TaskTemplateIR> templates;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& name& plane& after& templates;
  }
};

struct PlanIR {
  std::vector<CoveredBoxListIR> shared_covered_boxes;
  std::vector<StageIR> stages;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& shared_covered_boxes& stages;
  }
};

}  // namespace kangaroo
