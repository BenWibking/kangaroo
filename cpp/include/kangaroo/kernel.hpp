#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <hpx/future.hpp>
#include <hpx/serialization/vector.hpp>

namespace kangaroo {

enum class HostViewLayout : std::uint8_t {
  kRuntimeIJK = 0,
  kPlotfileKJI = 1,
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

  std::uint8_t* data() {
    detach();
    return data_->data();
  }

  const std::uint8_t* data() const {
    if (!data_) {
      return nullptr;
    }
    return data_->data() + start_offset();
  }

  bool empty() const { return size() == 0; }
  size_type size() const { return visible_size(); }

  SharedByteBuffer slice(size_type offset, size_type count) const {
    SharedByteBuffer out;
    out.data_ = data_;
    const size_type visible = size();
    offset = std::min(offset, visible);
    count = std::min(count, visible - offset);
    out.offset_ = start_offset() + offset;
    out.size_ = count;
    return out;
  }

  void clear() {
    detach();
    data_->clear();
  }

  void resize(size_type count) {
    detach();
    data_->resize(count);
    offset_ = 0;
    size_ = npos;
  }

  void resize(size_type count, value_type value) {
    detach();
    data_->resize(count, value);
    offset_ = 0;
    size_ = npos;
  }

  void assign(size_type count, value_type value) {
    detach();
    data_->assign(count, value);
    offset_ = 0;
    size_ = npos;
  }

  template <class InputIt>
  void assign(InputIt first, InputIt last) {
    detach();
    data_->assign(first, last);
    offset_ = 0;
    size_ = npos;
  }

  iterator begin() {
    detach();
    return data_->begin();
  }

  iterator end() {
    detach();
    return data_->end();
  }

  const_iterator begin() const {
    return storage().begin() + static_cast<std::ptrdiff_t>(start_offset());
  }
  const_iterator end() const { return begin() + static_cast<std::ptrdiff_t>(size()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  container_type& mutable_vector() {
    detach();
    return *data_;
  }

  const container_type& vector() const { return storage(); }

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    detach();
    ar& *data_;
    offset_ = 0;
    size_ = npos;
  }

 private:
  static constexpr size_type npos = std::numeric_limits<size_type>::max();

  // Host inputs are copied through futures frequently; keep copies shallow until
  // a caller asks for mutable storage.
  void detach() {
    if (!data_) {
      data_ = std::make_shared<container_type>();
      offset_ = 0;
      size_ = npos;
      return;
    }
    const bool is_full_view = start_offset() == 0 && visible_size() == data_->size();
    if (data_.use_count() != 1 || !is_full_view) {
      data_ = std::make_shared<container_type>(cbegin(), cend());
      offset_ = 0;
      size_ = npos;
    }
  }

  size_type start_offset() const {
    if (!data_) {
      return 0;
    }
    return std::min(offset_, data_->size());
  }

  size_type visible_size() const {
    if (!data_) {
      return 0;
    }
    const size_type start = start_offset();
    const size_type available = data_->size() - start;
    return size_ == npos ? available : std::min(size_, available);
  }

  const container_type& storage() const {
    static const container_type empty;
    return data_ ? *data_ : empty;
  }

  std::shared_ptr<container_type> data_;
  size_type offset_ = 0;
  size_type size_ = npos;
};

struct HostView {
  SharedByteBuffer data;
  HostViewLayout layout = HostViewLayout::kRuntimeIJK;

  void* ptr() { return data.data(); }
  const void* ptr() const { return data.data(); }
  std::size_t bytes() const { return data.size(); }

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& data& layout;
  }
};

class HostGridView3D {
 public:
  HostGridView3D(const HostView& view, int nx, int ny, int nz, int bytes_per_value)
      : view_(view), nx_(nx), ny_(ny), nz_(nz), bytes_per_value_(bytes_per_value) {}

  std::size_t logical_cell_count() const {
    if (has_box()) {
      return static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) *
             static_cast<std::size_t>(nz_);
    }
    if (bytes_per_value_ <= 0) {
      return 0;
    }
    return view_.data.size() / static_cast<std::size_t>(bytes_per_value_);
  }

  std::size_t logical_index(int i, int j, int k) const {
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny_) +
            static_cast<std::size_t>(j)) *
               static_cast<std::size_t>(nz_) +
           static_cast<std::size_t>(k);
  }

  bool get_double(int i, int j, int k, double& out) const {
    if (!contains(i, j, k)) {
      return false;
    }
    return read_element(physical_index(i, j, k), out);
  }

  double get_double_or(int i, int j, int k, double fallback = 0.0) const {
    double value = fallback;
    return get_double(i, j, k, value) ? value : fallback;
  }

  double get_logical_or(std::size_t logical_idx, double fallback = 0.0) const {
    double value = fallback;
    if (has_box()) {
      const std::size_t plane = static_cast<std::size_t>(ny_) * static_cast<std::size_t>(nz_);
      if (plane == 0) {
        return fallback;
      }
      const int i = static_cast<int>(logical_idx / plane);
      const std::size_t rem = logical_idx % plane;
      const int j = static_cast<int>(rem / static_cast<std::size_t>(nz_));
      const int k = static_cast<int>(rem % static_cast<std::size_t>(nz_));
      return get_double(i, j, k, value) ? value : fallback;
    }
    return read_element(logical_idx, value) ? value : fallback;
  }

  bool copy_value_bytes(int i, int j, int k, std::uint8_t* out) const {
    if (!contains(i, j, k) || bytes_per_value_ <= 0 || out == nullptr) {
      return false;
    }
    const std::size_t pos = physical_index(i, j, k) * static_cast<std::size_t>(bytes_per_value_);
    if (pos + static_cast<std::size_t>(bytes_per_value_) > view_.data.size()) {
      return false;
    }
    std::memcpy(out, view_.data.data() + pos, static_cast<std::size_t>(bytes_per_value_));
    return true;
  }

 private:
  bool has_box() const { return nx_ > 0 && ny_ > 0 && nz_ > 0; }

  bool contains(int i, int j, int k) const {
    return has_box() && i >= 0 && i < nx_ && j >= 0 && j < ny_ && k >= 0 && k < nz_;
  }

  std::size_t physical_index(int i, int j, int k) const {
    if (view_.layout == HostViewLayout::kPlotfileKJI) {
      return (static_cast<std::size_t>(k) * static_cast<std::size_t>(ny_) +
              static_cast<std::size_t>(j)) *
                 static_cast<std::size_t>(nx_) +
             static_cast<std::size_t>(i);
    }
    return logical_index(i, j, k);
  }

  bool read_element(std::size_t element_idx, double& out) const {
    if (bytes_per_value_ == 4) {
      const std::size_t pos = element_idx * sizeof(float);
      if (pos + sizeof(float) > view_.data.size()) {
        return false;
      }
      out = static_cast<double>(reinterpret_cast<const float*>(view_.data.data())[element_idx]);
      return true;
    }
    if (bytes_per_value_ == 8) {
      const std::size_t pos = element_idx * sizeof(double);
      if (pos + sizeof(double) > view_.data.size()) {
        return false;
      }
      out = reinterpret_cast<const double*>(view_.data.data())[element_idx];
      return true;
    }
    return false;
  }

  const HostView& view_;
  int nx_ = 0;
  int ny_ = 0;
  int nz_ = 0;
  int bytes_per_value_ = 0;
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
