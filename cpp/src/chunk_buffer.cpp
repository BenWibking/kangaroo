#include "kangaroo/chunk_buffer.hpp"

namespace kangaroo {

namespace {

template <typename T>
T load_unaligned_scalar(const std::uint8_t* source) noexcept {
  T value;
  std::memcpy(&value, source, sizeof(T));
  return value;
}

template <typename Source>
void store_converted_scalar(std::uint8_t* destination, ScalarType target,
                            Source value) noexcept {
  switch (target) {
    case ScalarType::kU8: {
      const auto converted = static_cast<std::uint8_t>(value);
      std::memcpy(destination, &converted, sizeof(converted));
      return;
    }
    case ScalarType::kI64: {
      const auto converted = static_cast<std::int64_t>(value);
      std::memcpy(destination, &converted, sizeof(converted));
      return;
    }
    case ScalarType::kF32: {
      const auto converted = static_cast<float>(value);
      std::memcpy(destination, &converted, sizeof(converted));
      return;
    }
    case ScalarType::kF64: {
      const auto converted = static_cast<double>(value);
      std::memcpy(destination, &converted, sizeof(converted));
      return;
    }
    case ScalarType::kOpaque:
      return;
  }
}

}  // namespace

std::uint64_t BufferDesc::element_count() const {
  if (rank < 1 || rank > kMaxBufferRank) {
    throw BufferContractError(BufferContractReason::kInvalidRank,
                              "buffer rank must be between 1 and 4");
  }
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis < rank; ++axis) {
    if (extents[axis] == 0) return 0;
    count = checked_multiply(count, extents[axis]);
  }
  return count;
}

std::uint64_t BufferDesc::required_bytes() const {
  if (element_count() == 0) return 0;
  std::uint64_t last = 0;
  for (std::size_t axis = 0; axis < rank; ++axis) {
    if (strides_bytes[axis] <= 0) {
      throw BufferContractError(BufferContractReason::kInvalidExtent,
                                "buffer strides must be positive");
    }
    last = checked_add(last, checked_multiply(
                                 extents[axis] - 1,
                                 static_cast<std::uint64_t>(strides_bytes[axis])));
  }
  return checked_add(last, scalar_size(scalar));
}

void BufferDesc::validate(std::size_t visible_databytes) const {
  if (rank < 1 || rank > kMaxBufferRank) {
    throw BufferContractError(BufferContractReason::kInvalidRank,
                              "buffer rank must be between 1 and 4");
  }
  if (scalar == ScalarType::kOpaque && rank != 1) {
    throw BufferContractError(BufferContractReason::kInvalidRank,
                              "opaque buffers must have rank 1");
  }
  for (std::size_t axis = 0; axis < kMaxBufferRank; ++axis) {
    if (axis >= rank && (extents[axis] != 0 || strides_bytes[axis] != 0)) {
      throw BufferContractError(BufferContractReason::kInvalidExtent,
                                "unused buffer axes must be zero");
    }
    if (axis < rank && extents[axis] == 0 && visible_databytes != 0) {
      throw BufferContractError(BufferContractReason::kInvalidExtent,
                                "non-empty buffers must have positive extents");
    }
  }
  if (scalar == ScalarType::kOpaque && extents[0] != visible_databytes) {
    throw BufferContractError(BufferContractReason::kDescriptorStorageMismatch,
                              "opaque extent must equal the visible byte count");
  }
  const auto needed = required_bytes();
  if (needed != visible_databytes) {
    throw BufferContractError(BufferContractReason::kDescriptorStorageMismatch,
                              "buffer descriptor requires " + std::to_string(needed) +
                                  " bytes but storage exposes " +
                                  std::to_string(visible_databytes));
  }
  if (scalar != ScalarType::kOpaque && needed != 0) {
    std::array<std::size_t, kMaxBufferRank> order{};
    for (std::size_t axis = 0; axis < rank; ++axis) order[axis] = axis;
    std::sort(order.begin(), order.begin() + rank, [&](std::size_t lhs, std::size_t rhs) {
      return strides_bytes[lhs] < strides_bytes[rhs];
    });
    std::uint64_t expected = scalar_size(scalar);
    for (std::size_t pos = 0; pos < rank; ++pos) {
      const auto axis = order[pos];
      if (extents[axis] == 1) continue;
      if (strides_bytes[axis] != static_cast<std::int64_t>(expected)) {
        throw BufferContractError(BufferContractReason::kInvalidExtent,
                                  "buffer layout must be a dense stride permutation");
      }
      expected = checked_multiply(expected, extents[axis]);
    }
  }
}

BufferDesc BufferDesc::contiguous(ScalarType scalar,
                                  std::span<const std::uint64_t> extents) {
  if (extents.empty() || extents.size() > kMaxBufferRank) {
    throw BufferContractError(BufferContractReason::kInvalidRank,
                              "buffer rank must be between 1 and 4");
  }
  BufferDesc desc;
  desc.scalar = scalar;
  desc.rank = static_cast<std::uint8_t>(extents.size());
  std::uint64_t stride = scalar_size(scalar);
  for (std::size_t reverse = extents.size(); reverse > 0; --reverse) {
    const auto axis = reverse - 1;
    desc.extents[axis] = extents[axis];
    desc.strides_bytes[axis] = static_cast<std::int64_t>(stride);
    stride = checked_multiply(stride, extents[axis]);
  }
  return desc;
}

BufferDesc BufferDesc::runtime_grid(ScalarType scalar,
                                    std::array<std::uint64_t, 3> extents) {
  return contiguous(scalar, extents);
}

BufferDesc BufferDesc::plotfile_grid(ScalarType scalar,
                                     std::array<std::uint64_t, 3> extents) {
  BufferDesc desc;
  desc.scalar = scalar;
  desc.rank = 3;
  desc.extents[0] = extents[0];
  desc.extents[1] = extents[1];
  desc.extents[2] = extents[2];
  desc.strides_bytes[0] = static_cast<std::int64_t>(scalar_size(scalar));
  desc.strides_bytes[1] = static_cast<std::int64_t>(
      checked_multiply(extents[0], scalar_size(scalar)));
  desc.strides_bytes[2] = static_cast<std::int64_t>(checked_multiply(
      checked_multiply(extents[0], extents[1]), scalar_size(scalar)));
  return desc;
}

BufferDesc BufferDesc::component_major_grid(
    ScalarType scalar, std::array<std::uint64_t, 3> extents,
    std::uint64_t components) {
  if (components <= 1) return plotfile_grid(scalar, extents);
  BufferDesc desc;
  desc.scalar = scalar;
  desc.rank = 4;
  desc.extents = {extents[0], extents[1], extents[2], components};
  desc.strides_bytes[0] = static_cast<std::int64_t>(scalar_size(scalar));
  desc.strides_bytes[1] = static_cast<std::int64_t>(
      checked_multiply(extents[0], scalar_size(scalar)));
  desc.strides_bytes[2] = static_cast<std::int64_t>(checked_multiply(
      checked_multiply(extents[0], extents[1]), scalar_size(scalar)));
  desc.strides_bytes[3] = static_cast<std::int64_t>(checked_multiply(
      checked_multiply(checked_multiply(extents[0], extents[1]), extents[2]),
      scalar_size(scalar)));
  return desc;
}

ChunkBuffer ChunkBuffer::copy_to(BufferDesc target) const {
  synchronize_dynamic_extent();
  if (desc_.scalar == ScalarType::kOpaque || target.scalar == ScalarType::kOpaque ||
      desc_.rank != target.rank) {
    throw BufferContractError(BufferContractReason::kScalarMismatch,
                              "layout copy requires numeric descriptors of equal rank");
  }
  for (std::size_t axis = 0; axis < desc_.rank; ++axis) {
    if (desc_.extents[axis] != target.extents[axis]) {
      throw BufferContractError(BufferContractReason::kInvalidExtent,
                                "layout copy requires matching logical extents");
    }
  }
  target.validate(static_cast<std::size_t>(target.required_bytes()));
  auto output = allocate(target);
  if (desc_.scalar == target.scalar && desc_.strides_bytes == target.strides_bytes) {
    std::copy(byte_view().begin(), byte_view().end(), output.mutable_byte_view().begin());
    return output;
  }
  const auto source = byte_view();
  auto destination = output.mutable_byte_view();
  std::array<std::uint64_t, kMaxBufferRank> index{};
  for (std::uint64_t element = 0; element < desc_.element_count(); ++element) {
    std::uint64_t source_offset = 0;
    std::uint64_t destination_offset = 0;
    for (std::size_t axis = 0; axis < desc_.rank; ++axis) {
      source_offset = checked_add(source_offset, checked_multiply(
          index[axis], static_cast<std::uint64_t>(desc_.strides_bytes[axis])));
      destination_offset = checked_add(destination_offset, checked_multiply(
          index[axis], static_cast<std::uint64_t>(target.strides_bytes[axis])));
    }
    const auto* source_scalar = source.data() + source_offset;
    auto* destination_scalar = destination.data() + destination_offset;
    switch (desc_.scalar) {
      case ScalarType::kU8:
        store_converted_scalar(destination_scalar, target.scalar,
                               load_unaligned_scalar<std::uint8_t>(source_scalar));
        break;
      case ScalarType::kI64:
        store_converted_scalar(destination_scalar, target.scalar,
                               load_unaligned_scalar<std::int64_t>(source_scalar));
        break;
      case ScalarType::kF32:
        store_converted_scalar(destination_scalar, target.scalar,
                               load_unaligned_scalar<float>(source_scalar));
        break;
      case ScalarType::kF64:
        store_converted_scalar(destination_scalar, target.scalar,
                               load_unaligned_scalar<double>(source_scalar));
        break;
      case ScalarType::kOpaque:
        break;
    }
    for (std::size_t reverse = desc_.rank; reverse > 0; --reverse) {
      const auto axis = reverse - 1;
      if (++index[axis] < desc_.extents[axis]) break;
      index[axis] = 0;
    }
  }
  return output;
}

void ChunkBuffer::copy_from(const ChunkBuffer& source) {
  auto converted = source.copy_to(desc());
  const auto source_bytes = converted.byte_view();
  auto destination = mutable_byte_view();
  std::copy(source_bytes.begin(), source_bytes.end(), destination.begin());
}

ChunkBuffer ChunkBuffer::copy_grid_region(
    std::array<std::uint64_t, 3> origin,
    std::array<std::uint64_t, 3> extents) const {
  synchronize_dynamic_extent();
  if (desc_.scalar == ScalarType::kOpaque || desc_.rank != 3) {
    throw BufferContractError(BufferContractReason::kRankMismatch,
                              "grid-region copy requires a numeric rank-three buffer");
  }
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (origin[axis] > desc_.extents[axis] ||
        extents[axis] > desc_.extents[axis] - origin[axis]) {
      throw BufferContractError(BufferContractReason::kInvalidExtent,
                                "grid-region copy exceeds the source extents");
    }
  }
  auto output = allocate(BufferDesc::runtime_grid(desc_.scalar, extents));
  const auto source = byte_view();
  auto destination = output.mutable_byte_view();
  const auto width = scalar_size(desc_.scalar);
  const auto& output_desc = output.desc();
  for (std::uint64_t i = 0; i < extents[0]; ++i) {
    for (std::uint64_t j = 0; j < extents[1]; ++j) {
      for (std::uint64_t k = 0; k < extents[2]; ++k) {
        const std::array<std::uint64_t, 3> source_index{
            origin[0] + i, origin[1] + j, origin[2] + k};
        const std::array<std::uint64_t, 3> destination_index{i, j, k};
        std::uint64_t source_offset = 0;
        std::uint64_t destination_offset = 0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          source_offset += source_index[axis] * desc_.strides_bytes[axis];
          destination_offset += destination_index[axis] * output_desc.strides_bytes[axis];
        }
        std::memcpy(destination.data() + destination_offset,
                    source.data() + source_offset, width);
      }
    }
  }
  return output;
}

}  // namespace kangaroo
