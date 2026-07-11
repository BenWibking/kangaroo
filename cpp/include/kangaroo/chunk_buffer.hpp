#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hpx/serialization/array.hpp>
#include <hpx/datastructures/serialization/optional.hpp>
#include <hpx/serialization/vector.hpp>

namespace kangaroo {

inline constexpr std::size_t kMaxBufferRank = 4;

enum class ScalarType : std::uint8_t { kOpaque = 0, kU8 = 1, kI64 = 2, kF32 = 3, kF64 = 4 };
enum class InitPolicy : std::uint8_t { kUninitialized = 0, kZero = 1 };

enum class BufferContractReason : std::uint8_t {
  kInvalidRank,
  kInvalidExtent,
  kArithmeticOverflow,
  kDescriptorStorageMismatch,
  kScalarMismatch,
  kRankMismatch,
  kInvalidDynamicResize,
  kDynamicUpperBoundViolation,
  kUnsupportedRealVisitationDtype,
  kVisitorArityExceeded,
  kOpaqueNumericAccess,
};

class BufferContractError : public std::runtime_error {
 public:
  BufferContractError(BufferContractReason reason, std::string message)
      : std::runtime_error(std::move(message)), reason_(reason) {}
  BufferContractReason reason() const noexcept { return reason_; }

 private:
  BufferContractReason reason_;
};

inline constexpr std::size_t scalar_size(ScalarType scalar) {
  switch (scalar) {
    case ScalarType::kOpaque:
    case ScalarType::kU8:
      return 1;
    case ScalarType::kI64:
    case ScalarType::kF64:
      return 8;
    case ScalarType::kF32:
      return 4;
  }
  return 0;
}

inline constexpr const char* scalar_type_name(ScalarType scalar) {
  switch (scalar) {
    case ScalarType::kOpaque: return "opaque";
    case ScalarType::kU8: return "u8";
    case ScalarType::kI64: return "i64";
    case ScalarType::kF32: return "f32";
    case ScalarType::kF64: return "f64";
  }
  return "unknown";
}

template <typename T> struct scalar_type_for;
template <> struct scalar_type_for<std::uint8_t> { static constexpr ScalarType value = ScalarType::kU8; };
template <> struct scalar_type_for<std::int64_t> { static constexpr ScalarType value = ScalarType::kI64; };
template <> struct scalar_type_for<float> { static constexpr ScalarType value = ScalarType::kF32; };
template <> struct scalar_type_for<double> { static constexpr ScalarType value = ScalarType::kF64; };

inline std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
  if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
    throw BufferContractError(BufferContractReason::kArithmeticOverflow,
                              "chunk-buffer size multiplication overflow");
  }
  return lhs * rhs;
}

inline std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs) {
  if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    throw BufferContractError(BufferContractReason::kArithmeticOverflow,
                              "chunk-buffer size addition overflow");
  }
  return lhs + rhs;
}

struct BufferDesc {
  ScalarType scalar = ScalarType::kOpaque;
  std::uint8_t rank = 1;
  std::array<std::uint64_t, kMaxBufferRank> extents{};
  std::array<std::int64_t, kMaxBufferRank> strides_bytes{};

  std::uint64_t element_count() const {
    if (rank < 1 || rank > kMaxBufferRank) {
      throw BufferContractError(BufferContractReason::kInvalidRank,
                                "buffer rank must be between 1 and 4");
    }
    std::uint64_t count = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      if (extents[axis] == 0) {
        return 0;
      }
      count = checked_multiply(count, extents[axis]);
    }
    return count;
  }

  std::uint64_t required_bytes() const {
    const auto count = element_count();
    if (count == 0) {
      return 0;
    }
    std::uint64_t last = 0;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      if (strides_bytes[axis] <= 0) {
        throw BufferContractError(BufferContractReason::kInvalidExtent,
                                  "buffer strides must be positive");
      }
      last = checked_add(last,
                         checked_multiply(extents[axis] - 1,
                                          static_cast<std::uint64_t>(strides_bytes[axis])));
    }
    return checked_add(last, scalar_size(scalar));
  }

  void validate(std::size_t visible_databytes) const {
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

    // Supported numeric layouts are dense positive-stride permutations.
    if (scalar != ScalarType::kOpaque && needed != 0) {
      std::array<std::size_t, kMaxBufferRank> order{};
      for (std::size_t axis = 0; axis < rank; ++axis) order[axis] = axis;
      std::sort(order.begin(), order.begin() + rank, [&](std::size_t lhs, std::size_t rhs) {
        return strides_bytes[lhs] < strides_bytes[rhs];
      });
      std::uint64_t expected = scalar_size(scalar);
      for (std::size_t pos = 0; pos < rank; ++pos) {
        const auto axis = order[pos];
        if (strides_bytes[axis] != static_cast<std::int64_t>(expected)) {
          throw BufferContractError(BufferContractReason::kInvalidExtent,
                                    "buffer layout must be a dense stride permutation");
        }
        expected = checked_multiply(expected, extents[axis]);
      }
    }
  }

  static BufferDesc contiguous(ScalarType scalar, std::span<const std::uint64_t> extents) {
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

  static BufferDesc runtime_grid(ScalarType scalar,
                                 std::array<std::uint64_t, 3> extents) {
    return contiguous(scalar, extents);
  }

  static BufferDesc plotfile_grid(ScalarType scalar,
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

  static BufferDesc component_major_grid(ScalarType scalar,
                                         std::array<std::uint64_t, 3> extents,
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

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& scalar& rank& extents& strides_bytes;
  }
};

class SharedByteBuffer {
 public:
  using container_type = std::vector<std::uint8_t>;
  using value_type = container_type::value_type;
  using size_type = container_type::size_type;
  using iterator = container_type::iterator;
  using const_iterator = container_type::const_iterator;

  SharedByteBuffer() : data_(std::make_shared<container_type>()) {}
  explicit SharedByteBuffer(container_type data)
      : data_(std::make_shared<container_type>(std::move(data))) {}

  SharedByteBuffer& operator=(container_type data) {
    data_ = std::make_shared<container_type>(std::move(data));
    offset_ = 0;
    size_ = npos;
    return *this;
  }

  const std::uint8_t* data() const noexcept {
    return data_ ? data_->data() + start_offset() : nullptr;
  }
  std::uint8_t* data() { return mutable_data(); }
  std::uint8_t* mutable_data() { detach(); return data_->data(); }
  bool empty() const noexcept { return size() == 0; }
  size_type size() const noexcept { return visible_size(); }
  bool is_shared_or_sliced() const noexcept {
    return data_ && (data_.use_count() != 1 || start_offset() != 0 || visible_size() != data_->size());
  }

  SharedByteBuffer slice(size_type offset, size_type count) const {
    SharedByteBuffer out;
    out.data_ = data_;
    const auto visible = size();
    offset = std::min(offset, visible);
    count = std::min(count, visible - offset);
    out.offset_ = start_offset() + offset;
    out.size_ = count;
    return out;
  }

  void clear() { detach(); data_->clear(); }
  void resize(size_type count) { detach(); data_->resize(count); }
  void resize(size_type count, value_type value) { detach(); data_->resize(count, value); }
  void assign(size_type count, value_type value) { detach(); data_->assign(count, value); }
  template <class InputIt> void assign(InputIt first, InputIt last) {
    detach();
    data_->assign(first, last);
  }
  iterator begin() { detach(); return data_->begin(); }
  iterator end() { detach(); return data_->end(); }
  const_iterator begin() const {
    return data_ ? data_->begin() + static_cast<std::ptrdiff_t>(start_offset())
                 : empty_storage().begin();
  }
  const_iterator end() const { return begin() + static_cast<std::ptrdiff_t>(size()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  container_type& mutable_vector() { detach(); return *data_; }
  const container_type& vector() const { return data_ ? *data_ : empty_storage(); }

  void detach() {
    if (!data_) {
      data_ = std::make_shared<container_type>();
      offset_ = 0;
      size_ = npos;
      return;
    }
    if (is_shared_or_sliced()) {
      const auto* visible = std::as_const(*this).data();
      data_ = std::make_shared<container_type>(visible, visible + size());
      offset_ = 0;
      size_ = npos;
    }
  }

  template <typename Archive>
  void save(Archive& ar, unsigned) const {
    container_type visible(data(), data() + size());
    ar& visible;
  }
  template <typename Archive>
  void load(Archive& ar, unsigned) {
    container_type visible;
    ar& visible;
    data_ = std::make_shared<container_type>(std::move(visible));
    offset_ = 0;
    size_ = npos;
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()

 private:
  static const container_type& empty_storage() {
    static const container_type empty;
    return empty;
  }
  static constexpr size_type npos = std::numeric_limits<size_type>::max();
  size_type start_offset() const noexcept { return data_ ? std::min(offset_, data_->size()) : 0; }
  size_type visible_size() const noexcept {
    if (!data_) return 0;
    const auto available = data_->size() - start_offset();
    return size_ == npos ? available : std::min(size_, available);
  }

  std::shared_ptr<container_type> data_;
  size_type offset_ = 0;
  size_type size_ = npos;
};

template <typename T>
class ScalarProxy {
 public:
  explicit ScalarProxy(std::uint8_t* ptr) : ptr_(ptr) {}
  operator T() const noexcept { T value; std::memcpy(&value, ptr_, sizeof(T)); return value; }
  ScalarProxy& operator=(T value) noexcept { std::memcpy(ptr_, &value, sizeof(T)); return *this; }
  ScalarProxy& operator=(const ScalarProxy& other) noexcept { return *this = static_cast<T>(other); }

 private:
  std::uint8_t* ptr_;
};

template <typename T, std::size_t Rank>
class TensorView {
  static_assert(Rank >= 1 && Rank <= kMaxBufferRank);
  using value_type = std::remove_const_t<T>;

 public:
  TensorView(std::conditional_t<std::is_const_v<T>, const std::uint8_t*, std::uint8_t*> data,
             std::array<std::uint64_t, Rank> extents,
             std::array<std::int64_t, Rank> strides)
      : data_(data), extents_(extents), strides_(strides) {}

  std::uint64_t extent(std::size_t axis) const noexcept { return extents_[axis]; }
  const auto& extents() const noexcept { return extents_; }
  const auto& strides_bytes() const noexcept { return strides_; }
  auto byte_data() const noexcept { return data_; }

  template <typename... Index>
  auto operator()(Index... index) const noexcept {
    static_assert(sizeof...(Index) == Rank);
    return access(std::array<std::uint64_t, Rank>{static_cast<std::uint64_t>(index)...});
  }

  template <typename... Index>
  auto at(Index... index) const {
    static_assert(sizeof...(Index) == Rank);
    std::array<std::uint64_t, Rank> indices{static_cast<std::uint64_t>(index)...};
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      if (indices[axis] >= extents_[axis]) throw std::out_of_range("TensorView index out of range");
    }
    return access(indices);
  }

 private:
  auto access(const std::array<std::uint64_t, Rank>& indices) const noexcept {
    std::uint64_t offset = 0;
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      offset += indices[axis] * static_cast<std::uint64_t>(strides_[axis]);
    }
    if constexpr (std::is_const_v<T>) {
      value_type value;
      std::memcpy(&value, data_ + offset, sizeof(value_type));
      return value;
    } else {
      return ScalarProxy<value_type>(data_ + offset);
    }
  }

  std::conditional_t<std::is_const_v<T>, const std::uint8_t*, std::uint8_t*> data_;
  std::array<std::uint64_t, Rank> extents_;
  std::array<std::int64_t, Rank> strides_;
};

template <typename T> using ArrayView = TensorView<T, 1>;
template <typename T> using BlockGrid = TensorView<T, 3>;
template <typename T> using ComponentBlockGrid = TensorView<T, 4>;

class ChunkBuffer {
 public:
  ChunkBuffer() = default;

  // Transitional compatibility for kernels being migrated to typed views.
  // Backends and the executor must always attach a concrete descriptor.
  SharedByteBuffer data;

  void* ptr() { return data.data(); }
  const void* ptr() const { return data.data(); }

  static ChunkBuffer allocate(BufferDesc desc, InitPolicy init = InitPolicy::kUninitialized) {
    const auto bytes64 = desc.required_bytes();
    if (bytes64 > std::numeric_limits<std::size_t>::max()) {
      throw BufferContractError(BufferContractReason::kArithmeticOverflow,
                                "buffer does not fit in host address space");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(bytes64));
    if (init == InitPolicy::kZero) std::fill(bytes.begin(), bytes.end(), std::uint8_t{0});
    ChunkBuffer out;
    out.data = SharedByteBuffer(std::move(bytes));
    out.desc_ = desc;
    out.desc_.validate(out.data.size());
    return out;
  }

  static ChunkBuffer allocate_dynamic(ScalarType scalar, std::uint64_t capacity_elements,
                                      InitPolicy init = InitPolicy::kUninitialized) {
    const std::array<std::uint64_t, 1> capacity{capacity_elements};
    auto desc = BufferDesc::contiguous(scalar, capacity);
    auto out = allocate(desc, init);
    out.dynamic_capacity_elements_ = capacity_elements;
    out.desc_.extents[0] = 0;
    return out;
  }

  static ChunkBuffer wrap(SharedByteBuffer storage, BufferDesc desc) {
    desc.validate(storage.size());
    ChunkBuffer out;
    out.data = std::move(storage);
    out.desc_ = desc;
    return out;
  }

  static ChunkBuffer opaque(std::vector<std::uint8_t> bytes) {
    const std::array<std::uint64_t, 1> extents{bytes.size()};
    return wrap(SharedByteBuffer(std::move(bytes)), BufferDesc::contiguous(ScalarType::kOpaque, extents));
  }

  const BufferDesc& desc() const noexcept { return desc_; }
  void replace_desc(BufferDesc desc) {
    desc.validate(data.size());
    desc_ = desc;
  }
  std::size_t bytes() const noexcept {
    if (dynamic_capacity_elements_) {
      if (desc_.extents[0] == 0) return 0;
      return static_cast<std::size_t>(desc_.extents[0] * scalar_size(desc_.scalar));
    }
    return data.size();
  }
  std::size_t capacity_bytes() const noexcept { return data.size(); }
  std::span<const std::uint8_t> byte_view() const noexcept { return {data.data(), bytes()}; }
  std::span<std::uint8_t> mutable_byte_view() {
    data.detach();
    return {data.mutable_data(), bytes()};
  }

  ChunkBuffer slice(std::size_t offset, std::size_t count, BufferDesc desc) const {
    return wrap(data.slice(offset, count), desc);
  }

  template <typename T, std::size_t Rank>
  TensorView<const T, Rank> view() const {
    validate_typed_view<T, Rank>();
    std::array<std::uint64_t, Rank> extents{};
    std::array<std::int64_t, Rank> strides{};
    std::copy_n(desc_.extents.begin(), Rank, extents.begin());
    std::copy_n(desc_.strides_bytes.begin(), Rank, strides.begin());
    return {data.data(), extents, strides};
  }

  template <typename T, std::size_t Rank>
  TensorView<T, Rank> mutable_view() {
    validate_typed_view<T, Rank>();
    data.detach();
    std::array<std::uint64_t, Rank> extents{};
    std::array<std::int64_t, Rank> strides{};
    std::copy_n(desc_.extents.begin(), Rank, extents.begin());
    std::copy_n(desc_.strides_bytes.begin(), Rank, strides.begin());
    return {data.mutable_data(), extents, strides};
  }

  template <typename T> ArrayView<const T> array() const { return view<T, 1>(); }
  template <typename T> ArrayView<T> mutable_array() { return mutable_view<T, 1>(); }

  void commit_dynamic_extent(std::uint64_t elements) {
    if (!dynamic_capacity_elements_ || desc_.rank != 1 || desc_.extents[0] != 0) {
      throw BufferContractError(BufferContractReason::kInvalidDynamicResize,
                                "buffer is not awaiting a dynamic extent commit");
    }
    if (elements > *dynamic_capacity_elements_) {
      throw BufferContractError(BufferContractReason::kDynamicUpperBoundViolation,
                                "dynamic extent exceeds its declared upper bound");
    }
    desc_.extents[0] = elements;
  }

  template <typename Archive>
  void save(Archive& ar, unsigned) const {
    ar& data& desc_& dynamic_capacity_elements_;
  }

  template <typename Archive>
  void load(Archive& ar, unsigned) {
    ar& data& desc_& dynamic_capacity_elements_;
    if (dynamic_capacity_elements_) {
      if (desc_.rank != 1) {
        throw BufferContractError(BufferContractReason::kInvalidExtent,
                                  "dynamic buffer must be rank one after deserialization");
      }
      if (desc_.extents[0] > *dynamic_capacity_elements_) {
        throw BufferContractError(BufferContractReason::kDynamicUpperBoundViolation,
                                  "dynamic extent exceeds capacity after deserialization");
      }
      const auto expected = checked_multiply(desc_.extents[0], scalar_size(desc_.scalar));
      if (expected != data.size()) {
        throw BufferContractError(BufferContractReason::kDescriptorStorageMismatch,
                                  "dynamic buffer extent/storage mismatch after deserialization");
      }
    } else {
      desc_.validate(data.size());
    }
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()

 private:
  template <typename T, std::size_t Rank>
  void validate_typed_view() const {
    if (desc_.scalar == ScalarType::kOpaque) {
      throw BufferContractError(BufferContractReason::kOpaqueNumericAccess,
                                "opaque payload cannot be viewed as numeric data");
    }
    if (desc_.scalar != scalar_type_for<std::remove_cv_t<T>>::value) {
      throw BufferContractError(BufferContractReason::kScalarMismatch,
                                "requested scalar type does not match buffer descriptor");
    }
    if (desc_.rank != Rank) {
      throw BufferContractError(BufferContractReason::kRankMismatch,
                                "requested rank does not match buffer descriptor");
    }
    desc_.validate(bytes());
  }

  BufferDesc desc_;
  std::optional<std::uint64_t> dynamic_capacity_elements_;
};

template <typename T>
class RealBufferView {
 public:
  explicit RealBufferView(const ChunkBuffer& buffer) : buffer_(&buffer) {}
  const BufferDesc& desc() const noexcept { return buffer_->desc(); }
  template <std::size_t Rank> TensorView<const T, Rank> tensor() const {
    return buffer_->template view<T, Rank>();
  }
  ArrayView<const T> array() const { return tensor<1>(); }
  BlockGrid<const T> grid() const { return tensor<3>(); }

 private:
  const ChunkBuffer* buffer_;
};

namespace detail {
template <std::size_t MaxInputs, typename F, typename... Views>
decltype(auto) visit_real_buffers_impl(std::span<const ChunkBuffer> inputs,
                                       std::size_t index, F&& fn, Views... views) {
  if (index == inputs.size()) {
    return std::invoke(std::forward<F>(fn), views...);
  }
  if constexpr (sizeof...(Views) >= MaxInputs) {
    throw BufferContractError(BufferContractReason::kVisitorArityExceeded,
                              "real-buffer visitor arity exceeds its configured maximum");
  } else {
    switch (inputs[index].desc().scalar) {
      case ScalarType::kF32:
        return visit_real_buffers_impl<MaxInputs>(inputs, index + 1, std::forward<F>(fn),
                                                  views..., RealBufferView<float>(inputs[index]));
      case ScalarType::kF64:
        return visit_real_buffers_impl<MaxInputs>(inputs, index + 1, std::forward<F>(fn),
                                                  views..., RealBufferView<double>(inputs[index]));
      default:
        throw BufferContractError(BufferContractReason::kUnsupportedRealVisitationDtype,
                                  "real-buffer visitor accepts only f32 and f64 inputs");
    }
  }
}

template <std::size_t Remaining, typename F, typename... Views>
decltype(auto) visit_real_buffers_exact_impl(std::span<const ChunkBuffer> inputs,
                                             std::size_t index,
                                             F&& fn,
                                             Views... views) {
  if constexpr (Remaining == 0) {
    return std::invoke(std::forward<F>(fn), views...);
  } else {
    switch (inputs[index].desc().scalar) {
      case ScalarType::kF32:
        return visit_real_buffers_exact_impl<Remaining - 1>(
            inputs, index + 1, std::forward<F>(fn), views...,
            RealBufferView<float>(inputs[index]));
      case ScalarType::kF64:
        return visit_real_buffers_exact_impl<Remaining - 1>(
            inputs, index + 1, std::forward<F>(fn), views...,
            RealBufferView<double>(inputs[index]));
      default:
        throw BufferContractError(BufferContractReason::kUnsupportedRealVisitationDtype,
                                  "real-buffer visitor accepts only f32 and f64 inputs");
    }
  }
}
}  // namespace detail

template <std::size_t MaxInputs = 10, typename F>
decltype(auto) visit_real_buffers(std::span<const ChunkBuffer> inputs, F&& fn) {
  if (inputs.size() > MaxInputs) {
    throw BufferContractError(BufferContractReason::kVisitorArityExceeded,
                              "real-buffer visitor received too many inputs");
  }
  return detail::visit_real_buffers_impl<MaxInputs>(inputs, 0, std::forward<F>(fn));
}

template <std::size_t Inputs, typename F>
decltype(auto) visit_real_buffers_exact(std::span<const ChunkBuffer> inputs, F&& fn) {
  static_assert(Inputs <= 10, "real-buffer visitor supports at most 10 inputs");
  if (inputs.size() != Inputs) {
    throw BufferContractError(BufferContractReason::kInvalidExtent,
                              "real-buffer visitor input count does not match expected arity");
  }
  return detail::visit_real_buffers_exact_impl<Inputs>(
      inputs, 0, std::forward<F>(fn));
}

}  // namespace kangaroo
