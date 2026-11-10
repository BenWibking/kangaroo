#pragma once

#include "kangaroo/chunk_buffer.hpp"
#include "kangaroo/kernel_params.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include <hpx/future.hpp>

namespace kangaroo {

struct NeighborViews {
  struct FieldNeighbors {
    std::vector<ChunkBuffer> xm, xp, ym, yp, zm, zp;
  };
  std::vector<int32_t> input_indices;
  std::vector<FieldNeighbors> inputs;
};

struct LevelMeta;

using KernelFn = std::function<hpx::future<void>(
    const LevelMeta& level,
    int32_t block_index,
    std::span<const ChunkBuffer> self_inputs,
    const NeighborViews& nbr_inputs,
    std::span<ChunkBuffer> outputs,
    const KernelParamsIR& params)>;

struct KernelDesc {
  std::string name;
  int32_t n_inputs = 0;
  int32_t n_outputs = 0;
  bool needs_neighbors = false;
  std::string param_schema_json;
};

}  // namespace kangaroo
