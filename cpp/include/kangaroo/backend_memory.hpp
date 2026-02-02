#pragma once

#include "kangaroo/dataset_backend.hpp"

#include <mutex>
#include <unordered_map>

namespace kangaroo {

class MemoryBackend : public DatasetBackend {
 public:
  MemoryBackend() = default;

  std::optional<HostView> get_chunk(const ChunkRef& ref) override;
  bool has_chunk(const ChunkRef& ref) const override;
  DatasetMetadata get_metadata() const override;

  void set_chunk(const ChunkRef& ref, HostView view);

  const std::unordered_map<ChunkRef, HostView, ChunkRefHash, ChunkRefEq>& data() const {
    return data_;
  }
  
  void set_data(std::unordered_map<ChunkRef, HostView, ChunkRefHash, ChunkRefEq> data) {
    data_ = std::move(data);
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<ChunkRef, HostView, ChunkRefHash, ChunkRefEq> data_;
};

}  // namespace kangaroo
