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

class DataService {
 public:
  virtual ~DataService() = default;
  virtual int home_rank(const ChunkRef&) const = 0;
  virtual HostView alloc_host(const ChunkRef&, std::size_t bytes) = 0;
  virtual hpx::future<HostView> get_host(const ChunkRef&) = 0;
  virtual hpx::future<void> put_host(const ChunkRef&, HostView) = 0;
};

}  // namespace kangaroo
