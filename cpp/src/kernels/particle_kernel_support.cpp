#include "particle_kernel_support.hpp"

#include "kernel_buffer_support.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace kangaroo {

void append_particle_values_as_f64(const ParticleFieldChunk &data,
                                   const std::string &name,
                                   const std::string &context,
                                   std::vector<double> &out_vals) {
  const std::size_t n =
      static_cast<std::size_t>(std::max<int64_t>(0, data.count));
  const std::size_t start = out_vals.size();
  out_vals.resize(start + n, 0.0);
  if (data.dtype == "float64") {
    if (data.bytes.size() < n * sizeof(double)) {
      throw std::runtime_error(context + ": short float64 payload for " + name);
    }
    for (std::size_t i = 0; i < n; ++i) {
      out_vals[start + i] = load_buffer_scalar<double>(data.bytes.data(), i);
    }
  } else if (data.dtype == "float32") {
    if (data.bytes.size() < n * sizeof(float)) {
      throw std::runtime_error(context + ": short float32 payload for " + name);
    }
    for (std::size_t i = 0; i < n; ++i) {
      out_vals[start + i] =
          static_cast<double>(load_buffer_scalar<float>(data.bytes.data(), i));
    }
  } else if (data.dtype == "int64") {
    if (data.bytes.size() < n * sizeof(int64_t)) {
      throw std::runtime_error(context + ": short int64 payload for " + name);
    }
    for (std::size_t i = 0; i < n; ++i) {
      out_vals[start + i] = static_cast<double>(
          load_buffer_scalar<int64_t>(data.bytes.data(), i));
    }
  } else {
    throw std::runtime_error(context + ": unsupported dtype '" + data.dtype +
                             "' for " + name);
  }
}

std::unordered_map<double, int64_t>
decode_particle_value_counts(std::span<const std::uint8_t> bytes) {
  std::unordered_map<double, int64_t> counts;
  if (bytes.size() < sizeof(uint64_t)) {
    return counts;
  }
  uint64_t n = 0;
  std::memcpy(&n, bytes.data(), sizeof(uint64_t));
  const std::size_t expected =
      sizeof(uint64_t) +
      static_cast<std::size_t>(n) * (sizeof(double) + sizeof(int64_t));
  if (bytes.size() < expected) {
    return counts;
  }
  counts.reserve(static_cast<std::size_t>(n));
  const auto *ptr = bytes.data() + sizeof(uint64_t);
  for (uint64_t i = 0; i < n; ++i) {
    double value = 0.0;
    int64_t count = 0;
    std::memcpy(&value, ptr, sizeof(double));
    ptr += sizeof(double);
    std::memcpy(&count, ptr, sizeof(int64_t));
    ptr += sizeof(int64_t);
    counts[value] += count;
  }
  return counts;
}

void encode_particle_value_counts(
    const std::unordered_map<double, int64_t> &counts,
    std::vector<std::uint8_t> &out) {
  const uint64_t n = static_cast<uint64_t>(counts.size());
  out.resize(sizeof(uint64_t) +
             static_cast<std::size_t>(n) * (sizeof(double) + sizeof(int64_t)));
  auto *ptr = out.data();
  std::memcpy(ptr, &n, sizeof(uint64_t));
  ptr += sizeof(uint64_t);
  for (const auto &[value, count] : counts) {
    std::memcpy(ptr, &value, sizeof(double));
    ptr += sizeof(double);
    std::memcpy(ptr, &count, sizeof(int64_t));
    ptr += sizeof(int64_t);
  }
}

} // namespace kangaroo
