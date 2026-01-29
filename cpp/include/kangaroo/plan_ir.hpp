#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kangaroo {

enum class ExecPlane : uint8_t { Chunk = 0, Graph = 1, Mixed = 2 };

struct FieldRefIR {
  int32_t field = 0;
  int32_t version = 0;
};

struct DomainIR {
  int32_t step = 0;
  int16_t level = 0;
  std::optional<std::vector<int32_t>> blocks;  // nullopt => all blocks on level
};

struct DepRuleIR {
  std::string kind{"None"};  // "None" or "FaceNeighbors"
  int32_t width = 0;
  bool faces[6] = {true, true, true, true, true, true};
};

struct TaskTemplateIR {
  std::string name;
  ExecPlane plane = ExecPlane::Chunk;
  std::string kernel;
  DomainIR domain;
  std::vector<FieldRefIR> inputs;
  std::vector<FieldRefIR> outputs;
  DepRuleIR deps;
  std::vector<std::uint8_t> params_msgpack;
};

struct StageIR {
  std::string name;
  ExecPlane plane = ExecPlane::Chunk;
  std::vector<int32_t> after;
  std::vector<TaskTemplateIR> templates;
};

struct PlanIR {
  std::vector<StageIR> stages;
};

}  // namespace kangaroo
