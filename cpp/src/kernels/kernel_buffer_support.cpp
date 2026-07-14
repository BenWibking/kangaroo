#include "kernel_buffer_support.hpp"

#include <algorithm>

namespace kangaroo {

namespace {

bool descriptors_match(const BufferDesc &lhs, const BufferDesc &rhs) {
  if (lhs.scalar != rhs.scalar || lhs.rank != rhs.rank) {
    return false;
  }
  for (std::size_t axis = 0; axis < lhs.rank; ++axis) {
    if (lhs.extents[axis] != rhs.extents[axis] ||
        lhs.strides_bytes[axis] != rhs.strides_bytes[axis]) {
      return false;
    }
  }
  return true;
}

template <typename T>
void reduce_matching_buffers(std::span<const ChunkBuffer> inputs,
                             ChunkBuffer &output) {
  const auto count = static_cast<std::size_t>(output.desc().element_count());
  auto out = output.mutable_byte_view();
  std::fill(out.begin(), out.end(), std::uint8_t{0});
  for (const auto &input : inputs) {
    if (!descriptors_match(input.desc(), output.desc())) {
      throw BufferContractError(
          BufferContractReason::kDescriptorStorageMismatch,
          "generic reduction requires identical descriptors");
    }
    const auto in = input.byte_view();
    for (std::size_t index = 0; index < count; ++index) {
      const T sum = load_buffer_scalar<T>(out.data(), index) +
                    load_buffer_scalar<T>(in.data(), index);
      store_buffer_scalar(out.data(), index, sum);
    }
  }
}

} // namespace

void reduce_matching_real_buffers(std::span<const ChunkBuffer> inputs,
                                  ChunkBuffer &output) {
  if (output.desc().scalar == ScalarType::kF32) {
    reduce_matching_buffers<float>(inputs, output);
  } else if (output.desc().scalar == ScalarType::kF64) {
    reduce_matching_buffers<double>(inputs, output);
  } else {
    throw BufferContractError(
        BufferContractReason::kScalarMismatch,
        "generic real reduction requires f32 or f64 buffers");
  }
}

RealGridAccessor make_real_grid_accessor(const ChunkBuffer &buffer) {
  RealGridAccessor accessor;
  visit_real_buffers_exact<1>(
      std::span<const ChunkBuffer>(&buffer, 1), [&](auto view) {
        accessor = make_real_grid_accessor(view.template tensor<3>());
      });
  return accessor;
}

} // namespace kangaroo
