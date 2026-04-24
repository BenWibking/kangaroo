#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <hpx/future.hpp>
#include <hpx/serialization/vector.hpp>

namespace kangaroo {

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
    return *this;
  }

  std::uint8_t* data() {
    detach();
    return data_->data();
  }

  const std::uint8_t* data() const { return data_ ? data_->data() : nullptr; }

  bool empty() const { return !data_ || data_->empty(); }
  size_type size() const { return data_ ? data_->size() : 0; }

  void clear() {
    detach();
    data_->clear();
  }

  void resize(size_type count) {
    detach();
    data_->resize(count);
  }

  void resize(size_type count, value_type value) {
    detach();
    data_->resize(count, value);
  }

  void assign(size_type count, value_type value) {
    detach();
    data_->assign(count, value);
  }

  template <class InputIt>
  void assign(InputIt first, InputIt last) {
    detach();
    data_->assign(first, last);
  }

  iterator begin() {
    detach();
    return data_->begin();
  }

  iterator end() {
    detach();
    return data_->end();
  }

  const_iterator begin() const { return storage().begin(); }
  const_iterator end() const { return storage().end(); }
  const_iterator cbegin() const { return storage().cbegin(); }
  const_iterator cend() const { return storage().cend(); }

  container_type& mutable_vector() {
    detach();
    return *data_;
  }

  const container_type& vector() const { return storage(); }

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    if (!data_) {
      data_ = std::make_shared<container_type>();
    }
    ar& *data_;
  }

 private:
  // Host inputs are copied through futures frequently; keep copies shallow until
  // a caller asks for mutable storage.
  void detach() {
    if (!data_) {
      data_ = std::make_shared<container_type>();
      return;
    }
    if (data_.use_count() != 1) {
      data_ = std::make_shared<container_type>(*data_);
    }
  }

  const container_type& storage() const {
    static const container_type empty;
    return data_ ? *data_ : empty;
  }

  std::shared_ptr<container_type> data_;
};

struct HostView {
  SharedByteBuffer data;

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
