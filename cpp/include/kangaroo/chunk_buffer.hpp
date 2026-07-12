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

  std::uint64_t element_count() const;
  std::uint64_t required_bytes() const;
  void validate(std::size_t visible_databytes) const;

  static BufferDesc contiguous(ScalarType scalar,
                               std::span<const std::uint64_t> extents);
  static BufferDesc runtime_grid(ScalarType scalar,
                                 std::array<std::uint64_t, 3> extents);
  static BufferDesc plotfile_grid(ScalarType scalar,
                                  std::array<std::uint64_t, 3> extents);
  static BufferDesc component_major_grid(ScalarType scalar,
                                         std::array<std::uint64_t, 3> extents,
                                         std::uint64_t components);

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& scalar& rank& extents& strides_bytes;
  }
};

class SharedByteBuffer {
 private:
  struct UninitializedStorageTag {};
  explicit SharedByteBuffer(UninitializedStorageTag) noexcept {}

 public:
  using container_type = std::vector<std::uint8_t>;
  using value_type = container_type::value_type;
  using size_type = container_type::size_type;
  using iterator = value_type*;
  using const_iterator = const value_type*;

  SharedByteBuffer() : vector_data_(std::make_shared<container_type>()) {}
  explicit SharedByteBuffer(container_type data)
      : vector_data_(std::make_shared<container_type>(std::move(data))) {}

  static SharedByteBuffer allocate_uninitialized(size_type count) {
    SharedByteBuffer out(UninitializedStorageTag{});
    if (count != 0) {
      out.raw_data_ = std::shared_ptr<std::uint8_t[]>(
          new std::uint8_t[count], std::default_delete<std::uint8_t[]>());
    }
    out.raw_size_ = count;
    out.raw_capacity_ = count;
    return out;
  }

  SharedByteBuffer& operator=(container_type data) {
    reset_vector(std::move(data));
    return *this;
  }

  const std::uint8_t* data() const noexcept {
    const auto* base = storage_data();
    return base ? base + start_offset() : nullptr;
  }
  std::uint8_t* data() { return mutable_data(); }
  std::uint8_t* mutable_data() {
    detach();
    return vector_data_ ? vector_data_->data() : raw_data_.get();
  }
  bool empty() const noexcept { return size() == 0; }
  size_type size() const noexcept { return visible_size(); }
  bool uses_raw_storage() const noexcept { return !vector_data_; }
  bool is_shared_or_sliced() const noexcept {
    return storage_use_count() != 1 || start_offset() != 0 ||
           visible_size() != storage_size();
  }

  SharedByteBuffer slice(size_type offset, size_type count) const {
    SharedByteBuffer out;
    out.vector_data_ = vector_data_;
    out.raw_data_ = raw_data_;
    out.raw_size_ = raw_size_;
    out.raw_capacity_ = raw_capacity_;
    const auto visible = size();
    offset = std::min(offset, visible);
    count = std::min(count, visible - offset);
    out.offset_ = start_offset() + offset;
    out.size_ = count;
    return out;
  }

  void clear() {
    detach();
    reset_vector({});
  }
  void resize(size_type count) {
    detach();
    if (vector_data_) {
      vector_data_->resize(count);
    } else {
      resize_raw(count, value_type{});
    }
  }
  void resize(size_type count, value_type value) {
    detach();
    if (vector_data_) {
      vector_data_->resize(count, value);
    } else {
      resize_raw(count, value);
    }
  }
  void assign(size_type count, value_type value) {
    detach();
    materialize_vector();
    vector_data_->assign(count, value);
  }
  template <class InputIt> void assign(InputIt first, InputIt last) {
    detach();
    materialize_vector();
    vector_data_->assign(first, last);
  }
  iterator begin() {
    detach();
    return mutable_data();
  }
  iterator end() {
    auto* first = begin();
    return size() == 0 ? first : first + size();
  }
  const_iterator begin() const { return data(); }
  const_iterator end() const {
    const auto* first = begin();
    return size() == 0 ? first : first + size();
  }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  container_type& mutable_vector() {
    detach();
    materialize_vector();
    return *vector_data_;
  }
  const container_type& vector() const {
    const_cast<SharedByteBuffer*>(this)->materialize_vector();
    return *vector_data_;
  }

  void detach() {
    if (is_shared_or_sliced()) {
      const auto* visible = std::as_const(*this).data();
      container_type copy;
      if (size() != 0) copy.assign(visible, visible + size());
      reset_vector(std::move(copy));
    }
  }

  template <typename Archive>
  void save(Archive& ar, unsigned) const {
    container_type visible;
    if (size() != 0) visible.assign(data(), data() + size());
    ar& visible;
  }
  template <typename Archive>
  void load(Archive& ar, unsigned) {
    container_type visible;
    ar& visible;
    reset_vector(std::move(visible));
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()

 private:
  void reset_vector(container_type data) {
    vector_data_ = std::make_shared<container_type>(std::move(data));
    raw_data_.reset();
    raw_size_ = 0;
    raw_capacity_ = 0;
    offset_ = 0;
    size_ = npos;
  }
  void materialize_vector() {
    if (vector_data_) return;
    container_type data;
    if (raw_size_ != 0) data.assign(raw_data_.get(), raw_data_.get() + raw_size_);
    vector_data_ = std::make_shared<container_type>(std::move(data));
    raw_data_.reset();
    raw_size_ = 0;
    raw_capacity_ = 0;
  }
  void resize_raw(size_type count, value_type value) {
    const auto old_size = raw_size_;
    if (count > raw_capacity_) {
      auto replacement = std::shared_ptr<std::uint8_t[]>(
          new std::uint8_t[count], std::default_delete<std::uint8_t[]>());
      if (old_size != 0) std::copy_n(raw_data_.get(), old_size, replacement.get());
      raw_data_ = std::move(replacement);
      raw_capacity_ = count;
    }
    if (count > old_size) {
      std::fill(raw_data_.get() + old_size, raw_data_.get() + count, value);
    }
    raw_size_ = count;
  }
  const std::uint8_t* storage_data() const noexcept {
    return vector_data_ ? vector_data_->data() : raw_data_.get();
  }
  size_type storage_size() const noexcept {
    return vector_data_ ? vector_data_->size() : raw_size_;
  }
  long storage_use_count() const noexcept {
    if (vector_data_) return vector_data_.use_count();
    return raw_data_ ? raw_data_.use_count() : 1;
  }
  static constexpr size_type npos = std::numeric_limits<size_type>::max();
  size_type start_offset() const noexcept { return std::min(offset_, storage_size()); }
  size_type visible_size() const noexcept {
    const auto available = storage_size() - start_offset();
    return size_ == npos ? available : std::min(size_, available);
  }

  std::shared_ptr<container_type> vector_data_;
  std::shared_ptr<std::uint8_t[]> raw_data_;
  size_type raw_size_ = 0;
  size_type raw_capacity_ = 0;
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

  static ChunkBuffer allocate(BufferDesc desc, InitPolicy init = InitPolicy::kUninitialized) {
    const auto bytes64 = desc.required_bytes();
    if (bytes64 > std::numeric_limits<std::size_t>::max()) {
      throw BufferContractError(BufferContractReason::kArithmeticOverflow,
                                "buffer does not fit in host address space");
    }
    ChunkBuffer out;
    const auto bytes = static_cast<std::size_t>(bytes64);
    if (init == InitPolicy::kZero) {
      out.storage_ = SharedByteBuffer(std::vector<std::uint8_t>(bytes, std::uint8_t{0}));
    } else {
      out.storage_ = SharedByteBuffer::allocate_uninitialized(bytes);
    }
    out.desc_ = desc;
    out.desc_.validate(out.storage_.size());
    return out;
  }

  static ChunkBuffer allocate_dynamic(ScalarType scalar, std::uint64_t capacity_elements,
                                      InitPolicy init = InitPolicy::kUninitialized) {
    const std::array<std::uint64_t, 1> capacity{capacity_elements};
    auto desc = BufferDesc::contiguous(scalar, capacity);
    auto out = allocate(desc, init);
    out.dynamic_capacity_elements_ = capacity_elements;
    out.dynamic_committed_elements_ =
        std::make_shared<std::optional<std::uint64_t>>(std::nullopt);
    out.desc_.extents[0] = 0;
    return out;
  }

  static ChunkBuffer wrap(SharedByteBuffer storage, BufferDesc desc) {
    desc.validate(storage.size());
    ChunkBuffer out;
    out.storage_ = std::move(storage);
    out.desc_ = desc;
    return out;
  }

  static ChunkBuffer opaque(std::vector<std::uint8_t> bytes) {
    const std::array<std::uint64_t, 1> extents{bytes.size()};
    return wrap(SharedByteBuffer(std::move(bytes)), BufferDesc::contiguous(ScalarType::kOpaque, extents));
  }

  const BufferDesc& desc() const noexcept {
    synchronize_dynamic_extent();
    return desc_;
  }
  void replace_desc(BufferDesc desc) {
    desc.validate(storage_.size());
    desc_ = desc;
  }
  std::size_t bytes() const noexcept {
    if (dynamic_capacity_elements_) {
      if (!dynamic_committed_elements_ || !dynamic_committed_elements_->has_value()) return 0;
      return static_cast<std::size_t>(dynamic_committed_elements_->value() *
                                      scalar_size(desc_.scalar));
    }
    return storage_.size();
  }
  std::size_t capacity_bytes() const noexcept { return storage_.size(); }
  bool empty() const noexcept { return bytes() == 0; }
  bool uses_uninitialized_storage() const noexcept { return storage_.uses_raw_storage(); }
  bool awaiting_dynamic_extent_commit() const noexcept {
    return dynamic_capacity_elements_.has_value() &&
           (!dynamic_committed_elements_ || !dynamic_committed_elements_->has_value());
  }
  std::span<const std::uint8_t> byte_view() const noexcept { return {storage_.data(), bytes()}; }
  std::span<std::uint8_t> mutable_byte_view() {
    storage_.detach();
    return {storage_.mutable_data(), bytes()};
  }

  std::span<std::uint8_t> mutable_capacity_byte_view() {
    if (!awaiting_dynamic_extent_commit()) {
      throw BufferContractError(BufferContractReason::kInvalidDynamicResize,
                                "capacity access requires an uncommitted dynamic buffer");
    }
    storage_.detach();
    return {storage_.mutable_data(), storage_.size()};
  }

  void assign_dynamic_bytes(std::span<const std::uint8_t> source) {
    if (!awaiting_dynamic_extent_commit()) {
      throw BufferContractError(BufferContractReason::kInvalidDynamicResize,
                                "byte assignment requires an uncommitted dynamic buffer");
    }
    const auto width = scalar_size(desc_.scalar);
    if (width == 0 || source.size() % width != 0 || source.size() > storage_.size()) {
      throw BufferContractError(BufferContractReason::kDescriptorStorageMismatch,
                                "dynamic byte assignment violates the buffer contract");
    }
    auto destination = mutable_capacity_byte_view();
    std::copy(source.begin(), source.end(), destination.begin());
    commit_dynamic_extent(source.size() / width);
  }

  void replace_dynamic_bytes(std::span<const std::uint8_t> source) {
    if (!dynamic_capacity_elements_) {
      throw BufferContractError(BufferContractReason::kInvalidDynamicResize,
                                "byte replacement requires a dynamic buffer");
    }
    const auto width = scalar_size(desc_.scalar);
    if (width == 0 || source.size() % width != 0 || source.size() > storage_.size()) {
      throw BufferContractError(BufferContractReason::kDescriptorStorageMismatch,
                                "dynamic byte replacement violates the buffer contract");
    }
    storage_.detach();
    std::copy(source.begin(), source.end(), storage_.mutable_data());
    desc_.extents[0] = source.size() / width;
    if (!dynamic_committed_elements_) {
      dynamic_committed_elements_ =
          std::make_shared<std::optional<std::uint64_t>>(std::nullopt);
    }
    *dynamic_committed_elements_ = desc_.extents[0];
    dynamic_extent_committed_ = true;
  }

  ChunkBuffer slice(std::size_t offset, std::size_t count, BufferDesc desc) const {
    return wrap(storage_.slice(offset, count), desc);
  }

  ChunkBuffer copy_to(BufferDesc target) const;
  void copy_from(const ChunkBuffer& source);
  ChunkBuffer copy_grid_region(std::array<std::uint64_t, 3> origin,
                               std::array<std::uint64_t, 3> extents) const;

  template <typename T, std::size_t Rank>
  TensorView<const T, Rank> view() const {
    validate_typed_view<T, Rank>();
    std::array<std::uint64_t, Rank> extents{};
    std::array<std::int64_t, Rank> strides{};
    std::copy_n(desc_.extents.begin(), Rank, extents.begin());
    std::copy_n(desc_.strides_bytes.begin(), Rank, strides.begin());
    return {storage_.data(), extents, strides};
  }

  template <typename T, std::size_t Rank>
  TensorView<T, Rank> mutable_view() {
    validate_typed_view<T, Rank>();
    storage_.detach();
    std::array<std::uint64_t, Rank> extents{};
    std::array<std::int64_t, Rank> strides{};
    std::copy_n(desc_.extents.begin(), Rank, extents.begin());
    std::copy_n(desc_.strides_bytes.begin(), Rank, strides.begin());
    return {storage_.mutable_data(), extents, strides};
  }

  template <typename T> ArrayView<const T> array() const { return view<T, 1>(); }
  template <typename T> ArrayView<T> mutable_array() { return mutable_view<T, 1>(); }

  template <typename T> ArrayView<T> mutable_dynamic_array() {
    if (!awaiting_dynamic_extent_commit() || desc_.rank != 1 ||
        desc_.scalar != scalar_type_for<std::remove_cv_t<T>>::value) {
      throw BufferContractError(BufferContractReason::kInvalidDynamicResize,
                                "dynamic array access does not match the buffer contract");
    }
    storage_.detach();
    const std::array<std::uint64_t, 1> extents{*dynamic_capacity_elements_};
    const std::array<std::int64_t, 1> strides{
        static_cast<std::int64_t>(scalar_size(desc_.scalar))};
    return {storage_.mutable_data(), extents, strides};
  }

  void commit_dynamic_extent(std::uint64_t elements) {
    if (!dynamic_capacity_elements_ || desc_.rank != 1 ||
        (dynamic_committed_elements_ && dynamic_committed_elements_->has_value())) {
      throw BufferContractError(BufferContractReason::kInvalidDynamicResize,
                                "buffer is not awaiting a dynamic extent commit");
    }
    if (elements > *dynamic_capacity_elements_) {
      throw BufferContractError(BufferContractReason::kDynamicUpperBoundViolation,
                                "dynamic extent exceeds its declared upper bound");
    }
    desc_.extents[0] = elements;
    if (!dynamic_committed_elements_) {
      dynamic_committed_elements_ =
          std::make_shared<std::optional<std::uint64_t>>(std::nullopt);
    }
    *dynamic_committed_elements_ = elements;
    dynamic_extent_committed_ = true;
  }

  template <typename Archive>
  void save(Archive& ar, unsigned) const {
    synchronize_dynamic_extent();
    const auto serialized_bytes = dynamic_extent_committed_ ? bytes() : storage_.size();
    auto visible_data = storage_.slice(0, serialized_bytes);
    ar& visible_data& desc_& dynamic_capacity_elements_& dynamic_extent_committed_;
  }

  template <typename Archive>
  void load(Archive& ar, unsigned) {
    ar& storage_& desc_& dynamic_capacity_elements_& dynamic_extent_committed_;
    if (dynamic_capacity_elements_) {
      dynamic_committed_elements_ = std::make_shared<std::optional<std::uint64_t>>(
          dynamic_extent_committed_ ? std::optional<std::uint64_t>(desc_.extents[0])
                                    : std::nullopt);
    } else {
      dynamic_committed_elements_.reset();
    }
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
      if (dynamic_extent_committed_ && expected != storage_.size()) {
        throw BufferContractError(BufferContractReason::kDescriptorStorageMismatch,
                                  "dynamic buffer extent/storage mismatch after deserialization");
      }
    } else {
      desc_.validate(storage_.size());
    }
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()

 private:
  void synchronize_dynamic_extent() const noexcept {
    if (dynamic_committed_elements_ && dynamic_committed_elements_->has_value()) {
      desc_.extents[0] = dynamic_committed_elements_->value();
      dynamic_extent_committed_ = true;
    }
  }

  template <typename T, std::size_t Rank>
  void validate_typed_view() const {
    synchronize_dynamic_extent();
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

  SharedByteBuffer storage_;
  mutable BufferDesc desc_;
  std::optional<std::uint64_t> dynamic_capacity_elements_;
  std::shared_ptr<std::optional<std::uint64_t>> dynamic_committed_elements_;
  mutable bool dynamic_extent_committed_ = false;
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
