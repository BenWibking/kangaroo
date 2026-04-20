#include "internal.hpp"

#include <msgpack.hpp>

#include <algorithm>
#include <limits>

namespace kangaroo {
namespace {

int unpack_bytes_per_value(std::span<const std::uint8_t> params_msgpack, int default_value) {
  int bytes_per_value = default_value;
  if (params_msgpack.empty()) {
    return bytes_per_value;
  }
  auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()), params_msgpack.size());
  auto root = handle.get();
  if (root.type != msgpack::type::MAP) {
    return bytes_per_value;
  }
  for (uint32_t i = 0; i < root.via.map.size; ++i) {
    const auto& k = root.via.map.ptr[i].key;
    if (k.type == msgpack::type::STR && k.as<std::string>() == "bytes_per_value") {
      const auto& v = root.via.map.ptr[i].val;
      if (v.type == msgpack::type::POSITIVE_INTEGER || v.type == msgpack::type::NEGATIVE_INTEGER) {
        bytes_per_value = v.as<int>();
      }
    }
  }
  return bytes_per_value;
}

double unpack_pixel_area(std::span<const std::uint8_t> params_msgpack, double default_value) {
  double pixel_area = default_value;
  if (params_msgpack.empty()) {
    return pixel_area;
  }
  auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()), params_msgpack.size());
  auto root = handle.get();
  if (root.type != msgpack::type::MAP) {
    return pixel_area;
  }
  for (uint32_t i = 0; i < root.via.map.size; ++i) {
    const auto& k = root.via.map.ptr[i].key;
    if (k.type == msgpack::type::STR && k.as<std::string>() == "pixel_area") {
      const auto& v = root.via.map.ptr[i].val;
      if (v.type == msgpack::type::FLOAT || v.type == msgpack::type::POSITIVE_INTEGER ||
          v.type == msgpack::type::NEGATIVE_INTEGER) {
        pixel_area = v.as<double>();
      }
    }
  }
  return pixel_area;
}

}  // namespace

namespace runtime_kernels {

KANGAROO_KERNEL(uniform_slice_add) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  const int bytes_per_value = unpack_bytes_per_value(params_msgpack, 8);
  if (outputs.empty() || self_inputs.size() < 2) {
    return hpx::make_ready_future();
  }
  auto& out = outputs[0].data;
  if (out.empty()) {
    return hpx::make_ready_future();
  }

  if (bytes_per_value == 8) {
    const std::size_t n = out.size() / sizeof(double);
    auto* out_d = reinterpret_cast<double*>(out.data());
    std::fill(out_d, out_d + n, 0.0);
    for (const auto& in_view : self_inputs) {
      const auto& in = in_view.data;
      if (in.empty()) {
        continue;
      }
      const std::size_t n_in = std::min(n, in.size() / sizeof(double));
      const auto* in_d = reinterpret_cast<const double*>(in.data());
      for (std::size_t i = 0; i < n_in; ++i) {
        out_d[i] += in_d[i];
      }
    }
  } else if (bytes_per_value == 4) {
    const std::size_t n = out.size() / sizeof(float);
    auto* out_f = reinterpret_cast<float*>(out.data());
    std::fill(out_f, out_f + n, 0.0f);
    for (const auto& in_view : self_inputs) {
      const auto& in = in_view.data;
      if (in.empty()) {
        continue;
      }
      const std::size_t n_in = std::min(n, in.size() / sizeof(float));
      const auto* in_f = reinterpret_cast<const float*>(in.data());
      for (std::size_t i = 0; i < n_in; ++i) {
        out_f[i] += in_f[i];
      }
    }
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(uniform_slice_finalize) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  const int bytes_per_value = unpack_bytes_per_value(params_msgpack, 4);
  const double pixel_area = unpack_pixel_area(params_msgpack, 1.0);
  if (outputs.empty() || self_inputs.size() < 2 || pixel_area == 0.0) {
    return hpx::make_ready_future();
  }

  const auto& sum = self_inputs[0].data;
  const auto& area = self_inputs[1].data;
  auto& out = outputs[0].data;
  if (out.empty()) {
    return hpx::make_ready_future();
  }
  const std::size_t n = std::min(sum.size(), area.size()) / sizeof(double);
  const auto* sum_d = reinterpret_cast<const double*>(sum.data());
  const auto* area_d = reinterpret_cast<const double*>(area.data());

  if (bytes_per_value == 8) {
    auto* out_d = reinterpret_cast<double*>(out.data());
    for (std::size_t i = 0; i < n; ++i) {
      if (area_d[i] == 0.0) {
        out_d[i] = std::numeric_limits<double>::quiet_NaN();
      } else {
        out_d[i] = sum_d[i] / pixel_area;
      }
    }
  } else if (bytes_per_value == 4) {
    auto* out_f = reinterpret_cast<float*>(out.data());
    for (std::size_t i = 0; i < n; ++i) {
      if (area_d[i] == 0.0) {
        out_f[i] = std::numeric_limits<float>::quiet_NaN();
      } else {
        out_f[i] = static_cast<float>(sum_d[i] / pixel_area);
      }
    }
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(uniform_slice_reduce) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  const int bytes_per_value = unpack_bytes_per_value(params_msgpack, 4);
  if (outputs.empty() || self_inputs.empty()) {
    return hpx::make_ready_future();
  }

  auto& out = outputs[0].data;
  if (out.empty()) {
    return hpx::make_ready_future();
  }
  std::fill(out.begin(), out.end(), 0);

  if (bytes_per_value == 4) {
    const std::size_t n = out.size() / sizeof(float);
    auto* out_f = reinterpret_cast<float*>(out.data());
    for (const auto& in_view : self_inputs) {
      const auto& in = in_view.data;
      if (in.empty()) {
        continue;
      }
      const std::size_t n_in = std::min(n, in.size() / sizeof(float));
      const auto* in_f = reinterpret_cast<const float*>(in.data());
      for (std::size_t i = 0; i < n_in; ++i) {
        out_f[i] += in_f[i];
      }
    }
  } else if (bytes_per_value == 8) {
    const std::size_t n = out.size() / sizeof(double);
    auto* out_d = reinterpret_cast<double*>(out.data());
    for (const auto& in_view : self_inputs) {
      const auto& in = in_view.data;
      if (in.empty()) {
        continue;
      }
      const std::size_t n_in = std::min(n, in.size() / sizeof(double));
      const auto* in_d = reinterpret_cast<const double*>(in.data());
      for (std::size_t i = 0; i < n_in; ++i) {
        out_d[i] += in_d[i];
      }
    }
  }
  return hpx::make_ready_future();
}

}  // namespace runtime_kernels
}  // namespace kangaroo
