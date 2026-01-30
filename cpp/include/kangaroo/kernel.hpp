#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include <hpx/future.hpp>

namespace kangaroo {

struct HostView {
  std::vector<std::uint8_t> data;

  void* ptr() { return data.data(); }
  const void* ptr() const { return data.data(); }
  std::size_t bytes() const { return data.size(); }

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& data;
  }
};

struct NeighborViews {
  struct FieldNeighbors {
    std::vector<HostView> xm, xp, ym, yp, zm, zp;
  };

  std::vector<int32_t> input_indices;
  std::vector<FieldNeighbors> inputs;
};

struct LevelMeta;  // forward

using KernelFn = std::function<hpx::future<void>(
    const LevelMeta& level,
    int32_t block_index,
    std::span<const HostView> self_inputs,
    const NeighborViews& nbr_inputs,
    std::span<HostView> outputs,
    std::span<const std::uint8_t> params_msgpack)>;

struct KernelDesc {
  std::string name;
  int32_t n_inputs = 0;
  int32_t n_outputs = 0;
  bool needs_neighbors = false;
  std::string param_schema_json;
};

}  // namespace kangaroo
