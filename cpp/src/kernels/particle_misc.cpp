#include "internal.hpp"

#include <msgpack.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace kangaroo {
namespace {

const msgpack::object* find_param(const msgpack::object& root, const char* key) {
  if (root.type != msgpack::type::MAP) {
    return nullptr;
  }
  for (uint32_t i = 0; i < root.via.map.size; ++i) {
    const auto& k = root.via.map.ptr[i].key;
    if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
      return &root.via.map.ptr[i].val;
    }
  }
  return nullptr;
}

double unpack_scalar_param(std::span<const std::uint8_t> params_msgpack) {
  double scalar = 0.0;
  if (params_msgpack.empty()) {
    return scalar;
  }
  auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()), params_msgpack.size());
  auto root = handle.get();
  if (const auto* value = find_param(root, "scalar")) {
    scalar = value->as<double>();
  }
  return scalar;
}

std::vector<double> unpack_values_param(std::span<const std::uint8_t> params_msgpack) {
  std::vector<double> values;
  if (params_msgpack.empty()) {
    return values;
  }
  auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()), params_msgpack.size());
  auto root = handle.get();
  const auto* value = find_param(root, "values");
  if (value == nullptr || value->type != msgpack::type::ARRAY) {
    return values;
  }
  values.reserve(value->via.array.size);
  for (uint32_t i = 0; i < value->via.array.size; ++i) {
    values.push_back(value->via.array.ptr[i].as<double>());
  }
  return values;
}

bool unpack_finite_only(std::span<const std::uint8_t> params_msgpack) {
  bool finite_only = true;
  if (params_msgpack.empty()) {
    return finite_only;
  }
  auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()), params_msgpack.size());
  auto root = handle.get();
  if (const auto* value = find_param(root, "finite_only")) {
    finite_only = value->as<bool>();
  }
  return finite_only;
}

}  // namespace

namespace runtime_kernels {

KANGAROO_KERNEL(particle_eq_mask) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  const double scalar = unpack_scalar_param(params_msgpack);
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& in = self_inputs[0].data;
  const std::size_t n = in.size() / sizeof(double);
  outputs[0].data.resize(n);
  const auto* in_d = reinterpret_cast<const double*>(in.data());
  auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = (in_d[i] == scalar) ? 1 : 0;
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_isin_mask) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  const auto values = unpack_values_param(params_msgpack);
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& in = self_inputs[0].data;
  const std::size_t n = in.size() / sizeof(double);
  outputs[0].data.resize(n);
  const auto* in_d = reinterpret_cast<const double*>(in.data());
  auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
  for (std::size_t i = 0; i < n; ++i) {
    bool found = false;
    for (double value : values) {
      if (in_d[i] == value) {
        found = true;
        break;
      }
    }
    out[i] = found ? 1 : 0;
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_isfinite_mask) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  (void)params_msgpack;
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& in = self_inputs[0].data;
  const std::size_t n = in.size() / sizeof(double);
  outputs[0].data.resize(n);
  const auto* in_d = reinterpret_cast<const double*>(in.data());
  auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = std::isfinite(in_d[i]) ? 1 : 0;
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_abs_lt_mask) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  const double scalar = unpack_scalar_param(params_msgpack);
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& in = self_inputs[0].data;
  const std::size_t n = in.size() / sizeof(double);
  outputs[0].data.resize(n);
  const auto* in_d = reinterpret_cast<const double*>(in.data());
  auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = (std::abs(in_d[i]) < scalar) ? 1 : 0;
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_le_mask) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  const double scalar = unpack_scalar_param(params_msgpack);
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& in = self_inputs[0].data;
  const std::size_t n = in.size() / sizeof(double);
  outputs[0].data.resize(n);
  const auto* in_d = reinterpret_cast<const double*>(in.data());
  auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = (in_d[i] <= scalar) ? 1 : 0;
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_gt_mask) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  const double scalar = unpack_scalar_param(params_msgpack);
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& in = self_inputs[0].data;
  const std::size_t n = in.size() / sizeof(double);
  outputs[0].data.resize(n);
  const auto* in_d = reinterpret_cast<const double*>(in.data());
  auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = (in_d[i] > scalar) ? 1 : 0;
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_and_mask) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  (void)params_msgpack;
  if (self_inputs.size() < 2 || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& a = self_inputs[0].data;
  const auto& b = self_inputs[1].data;
  const std::size_t n = std::min(a.size(), b.size());
  outputs[0].data.resize(n);
  auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
  const auto* a_u = reinterpret_cast<const std::uint8_t*>(a.data());
  const auto* b_u = reinterpret_cast<const std::uint8_t*>(b.data());
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = (a_u[i] != 0 && b_u[i] != 0) ? 1 : 0;
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_filter) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  (void)params_msgpack;
  if (self_inputs.size() < 2 || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& values = self_inputs[0].data;
  const auto& mask = self_inputs[1].data;
  const std::size_t n = std::min(values.size() / sizeof(double), mask.size());
  const auto* in_d = reinterpret_cast<const double*>(values.data());
  const auto* m_u = reinterpret_cast<const std::uint8_t*>(mask.data());
  std::size_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (m_u[i] != 0) {
      ++count;
    }
  }
  outputs[0].data.resize(count * sizeof(double));
  auto* out_d = reinterpret_cast<double*>(outputs[0].data.data());
  std::size_t out_idx = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (m_u[i] != 0) {
      out_d[out_idx++] = in_d[i];
    }
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_subtract) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  (void)params_msgpack;
  if (self_inputs.size() < 2 || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& a = self_inputs[0].data;
  const auto& b = self_inputs[1].data;
  const std::size_t n = std::min(a.size(), b.size()) / sizeof(double);
  outputs[0].data.resize(n * sizeof(double));
  const auto* a_d = reinterpret_cast<const double*>(a.data());
  const auto* b_d = reinterpret_cast<const double*>(b.data());
  auto* out_d = reinterpret_cast<double*>(outputs[0].data.data());
  for (std::size_t i = 0; i < n; ++i) {
    out_d[i] = a_d[i] - b_d[i];
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_distance3) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  (void)params_msgpack;
  if (self_inputs.size() < 6 || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const std::size_t n = std::min({self_inputs[0].data.size(), self_inputs[1].data.size(), self_inputs[2].data.size(),
                                  self_inputs[3].data.size(), self_inputs[4].data.size(), self_inputs[5].data.size()}) /
                        sizeof(double);
  outputs[0].data.resize(n * sizeof(double));
  const auto* ax = reinterpret_cast<const double*>(self_inputs[0].data.data());
  const auto* ay = reinterpret_cast<const double*>(self_inputs[1].data.data());
  const auto* az = reinterpret_cast<const double*>(self_inputs[2].data.data());
  const auto* bx = reinterpret_cast<const double*>(self_inputs[3].data.data());
  const auto* by = reinterpret_cast<const double*>(self_inputs[4].data.data());
  const auto* bz = reinterpret_cast<const double*>(self_inputs[5].data.data());
  auto* out = reinterpret_cast<double*>(outputs[0].data.data());
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = ax[i] - bx[i];
    const double dy = ay[i] - by[i];
    const double dz = az[i] - bz[i];
    out[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_sum) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  (void)params_msgpack;
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& in = self_inputs[0].data;
  const std::size_t n = in.size() / sizeof(double);
  const auto* in_d = reinterpret_cast<const double*>(in.data());
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += in_d[i];
  }
  outputs[0].data.resize(sizeof(double));
  *reinterpret_cast<double*>(outputs[0].data.data()) = sum;
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_count) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  (void)params_msgpack;
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& in = self_inputs[0].data;
  const std::size_t n = in.size();
  const auto* in_u = reinterpret_cast<const std::uint8_t*>(in.data());
  int64_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (in_u[i] != 0) {
      ++count;
    }
  }
  outputs[0].data.resize(sizeof(int64_t));
  *reinterpret_cast<int64_t*>(outputs[0].data.data()) = count;
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_len_f64) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  (void)params_msgpack;
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const int64_t n = static_cast<int64_t>(self_inputs[0].data.size() / sizeof(double));
  outputs[0].data.resize(sizeof(int64_t));
  *reinterpret_cast<int64_t*>(outputs[0].data.data()) = n;
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_min) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  const bool finite_only = unpack_finite_only(params_msgpack);
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& in = self_inputs[0].data;
  const std::size_t n = in.size() / sizeof(double);
  const auto* in_d = reinterpret_cast<const double*>(in.data());
  double out_v = std::numeric_limits<double>::infinity();
  bool any = false;
  for (std::size_t i = 0; i < n; ++i) {
    const double value = in_d[i];
    if (finite_only && !std::isfinite(value)) {
      continue;
    }
    if (!any || value < out_v) {
      out_v = value;
      any = true;
    }
  }
  outputs[0].data.resize(sizeof(double));
  *reinterpret_cast<double*>(outputs[0].data.data()) = any ? out_v : std::numeric_limits<double>::infinity();
  return hpx::make_ready_future();
}

KANGAROO_KERNEL(particle_max) {
  (void)level;
  (void)block_index;
  (void)nbr_inputs;
  const bool finite_only = unpack_finite_only(params_msgpack);
  if (self_inputs.empty() || outputs.empty()) {
    return hpx::make_ready_future();
  }
  const auto& in = self_inputs[0].data;
  const std::size_t n = in.size() / sizeof(double);
  const auto* in_d = reinterpret_cast<const double*>(in.data());
  double out_v = -std::numeric_limits<double>::infinity();
  bool any = false;
  for (std::size_t i = 0; i < n; ++i) {
    const double value = in_d[i];
    if (finite_only && !std::isfinite(value)) {
      continue;
    }
    if (!any || value > out_v) {
      out_v = value;
      any = true;
    }
  }
  outputs[0].data.resize(sizeof(double));
  *reinterpret_cast<double*>(outputs[0].data.data()) = any ? out_v : -std::numeric_limits<double>::infinity();
  return hpx::make_ready_future();
}

}  // namespace runtime_kernels
}  // namespace kangaroo
