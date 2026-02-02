#include "kangaroo/backend_memory.hpp"

namespace kangaroo {

std::optional<HostView> MemoryBackend::get_chunk(const ChunkRef& ref) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(ref);
  if (it != data_.end()) {
    return it->second;
  }
  return std::nullopt;
}

bool MemoryBackend::has_chunk(const ChunkRef& ref) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return data_.find(ref) != data_.end();
}

DatasetMetadata MemoryBackend::get_metadata() const {
  // TODO: Implement metadata storage for MemoryBackend
  return DatasetMetadata{};
}

void MemoryBackend::set_chunk(const ChunkRef& ref, HostView view) {
  std::lock_guard<std::mutex> lock(mutex_);
  data_[ref] = std::move(view);
}

}  // namespace kangaroo
