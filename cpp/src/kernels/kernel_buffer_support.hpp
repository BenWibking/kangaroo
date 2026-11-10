#pragma once

#include "kangaroo/buffer_resolution.hpp"
#include "kangaroo/chunk_buffer.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <span>

namespace kangaroo {

template <typename T>
T load_buffer_scalar(const std::uint8_t *data, std::size_t index) {
  T value;
  std::memcpy(&value, data + index * sizeof(T), sizeof(T));
  return value;
}

template <typename T>
void store_buffer_scalar(std::uint8_t *data, std::size_t index, T value) {
  std::memcpy(data + index * sizeof(T), &value, sizeof(T));
}

void reduce_matching_real_buffers(std::span<const ChunkBuffer> inputs,
                                  ChunkBuffer &output);

struct RealGridAccessor {
  const std::uint8_t *data = nullptr;
  std::array<std::int64_t, 3> strides{};
  double (*load)(const RealGridAccessor &, int, int, int) = nullptr;

  double operator()(int i, int j, int k) const { return load(*this, i, j, k); }
};

struct RealBufferAccessor {
  const std::uint8_t *data = nullptr;
  std::uint8_t rank = 0;
  std::array<std::uint64_t, kMaxBufferRank> extents{};
  std::array<std::int64_t, kMaxBufferRank> strides{};
  double (*load)(const RealBufferAccessor &, std::size_t) = nullptr;

  double operator()(std::size_t index) const { return load(*this, index); }
};

template <typename T>
RealGridAccessor make_real_grid_accessor(const TensorView<const T, 3> &grid) {
  RealGridAccessor accessor;
  accessor.data = grid.byte_data();
  accessor.strides = grid.strides_bytes();
  accessor.load = [](const RealGridAccessor &self, int i, int j, int k) {
    const auto offset = static_cast<std::uint64_t>(i) * self.strides[0] +
                        static_cast<std::uint64_t>(j) * self.strides[1] +
                        static_cast<std::uint64_t>(k) * self.strides[2];
    T value;
    std::memcpy(&value, self.data + offset, sizeof(T));
    return static_cast<double>(value);
  };
  return accessor;
}

RealGridAccessor make_real_grid_accessor(const ChunkBuffer &buffer);
RealBufferAccessor make_real_buffer_accessor(const ChunkBuffer &buffer);

} // namespace kangaroo
