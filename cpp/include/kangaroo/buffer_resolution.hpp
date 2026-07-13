#pragma once

#include "kangaroo/data_service.hpp"
#include "kangaroo/runmeta.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <typeindex>
#include <utility>

namespace kangaroo {

struct BufferSpecIR;
struct TaskTemplateIR;

struct BufferFacts {
  std::size_t payload_bytes = 0;
  std::size_t storage_bytes = 0;
  bool storage_known = false;
  std::optional<BufferDesc> desc;
  std::optional<std::uint64_t> element_capacity;
};

struct DynamicOutputBoundContext {
  ScalarType scalar = ScalarType::kOpaque;
  const RunMeta& meta;
  const DataService& data;
  int32_t step = 0;
  int16_t level = 0;
  int32_t block = 0;
  std::size_t output_index = 0;
  std::span<const BufferFacts> inputs;
  std::type_index prepared_params_type{typeid(void)};
  std::shared_ptr<const void> prepared_params;

  template <typename Params>
  const Params& params() const {
    if (prepared_params_type != std::type_index(typeid(Params)) || !prepared_params) {
      throw std::runtime_error("dynamic output bound has incompatible prepared parameters");
    }
    return *static_cast<const Params*>(prepared_params.get());
  }
};

using DynamicOutputBoundFn =
    std::function<std::optional<std::uint64_t>(const DynamicOutputBoundContext&)>;

class DynamicOutputBoundEvaluator {
 public:
  explicit DynamicOutputBoundEvaluator(DynamicOutputBoundFn fn) : fn_(std::move(fn)) {}

  std::optional<std::uint64_t> operator()(const DynamicOutputBoundContext& context) const {
    return fn_(context);
  }

 private:
  DynamicOutputBoundFn fn_;
};

struct BufferResolution {
  ResolvedBufferSpec allocation;
  BufferFacts facts;
};

struct AmrSubboxPackParams {
  int32_t input_field = -1;
  int32_t input_version = 0;
  int32_t input_step = 0;
  int16_t input_level = 0;
  int32_t halo_cells = 1;
};

BufferFacts buffer_facts(const ChunkBuffer& buffer);

std::optional<BufferResolution> try_resolve_buffer_spec(
    const BufferSpecIR& spec,
    const TaskTemplateIR& task,
    const DataService& data,
    const RunMeta& meta,
    int32_t step,
    int16_t level,
    int32_t block,
    std::size_t output_index,
    std::span<const BufferFacts> inputs);

ResolvedBufferSpec resolve_output_spec_for_task(
    const BufferSpecIR& spec,
    const TaskTemplateIR& task,
    const DataService& data,
    const RunMeta& meta,
    int32_t step,
    int16_t level,
    int32_t block,
    std::size_t output_index,
    std::span<const ChunkBuffer> inputs);

std::optional<std::uint64_t> estimate_amr_subbox_pack_capacity(
    const DynamicOutputBoundContext& context,
    const AmrSubboxPackParams& params);

}  // namespace kangaroo
