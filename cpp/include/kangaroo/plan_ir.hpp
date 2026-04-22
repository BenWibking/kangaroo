#pragma once

#include <cstdint>
#include <optional>
#include <string>
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

struct TaskTemplateIR {
  std::string name;
  ExecPlane plane = ExecPlane::Chunk;
  std::string kernel;
  DomainIR domain;
  std::vector<FieldRefIR> inputs;
  std::vector<FieldRefIR> outputs;
  std::vector<int32_t> output_bytes;
  DepRuleIR deps;
  std::vector<std::uint8_t> params_msgpack;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& name& plane& kernel& domain& inputs& outputs& output_bytes& deps& params_msgpack;
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
  std::vector<StageIR> stages;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& stages;
  }
};

}  // namespace kangaroo
