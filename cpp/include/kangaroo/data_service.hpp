#pragma once

#include "kangaroo/kernel.hpp"

#include <cstdint>

#include <hpx/future.hpp>

namespace kangaroo {

struct ChunkRef {
  int32_t step = 0;
  int16_t level = 0;
  int32_t field = 0;
  int32_t version = 0;
  int32_t block = 0;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& step& level& field& version& block;
  }
};

struct ChunkRefHash {
  std::size_t operator()(const ChunkRef& ref) const {
    std::size_t h = 0xcbf29ce484222325ull;
    auto mix = [&](auto v) {
      h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };
    mix(ref.step);
    mix(ref.level);
    mix(ref.field);
    mix(ref.version);
    mix(ref.block);
    return h;
  }
};

struct ChunkRefEq {
  bool operator()(const ChunkRef& a, const ChunkRef& b) const {
    return a.step == b.step && a.level == b.level && a.field == b.field && a.version == b.version &&
           a.block == b.block;
  }
};

class DataService {
 public:
  virtual ~DataService() = default;
  virtual int home_rank(const ChunkRef&) const = 0;
  virtual HostView alloc_host(const ChunkRef&, std::size_t bytes) = 0;
  virtual hpx::future<HostView> get_host(const ChunkRef&) = 0;
  virtual hpx::future<void> put_host(const ChunkRef&, HostView) = 0;
};

}  // namespace kangaroo
