#include "kangaroo/backend_memory.hpp"

namespace kangaroo {

std::optional<ChunkBuffer> MemoryBackend::get_chunk(const ChunkRef& ref) {
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

std::optional<BufferDesc> MemoryBackend::describe_chunk(const ChunkRef& ref) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(ref);
  return it == data_.end() ? std::nullopt : std::optional<BufferDesc>(it->second.desc());
}

std::size_t MemoryBackend::estimate_chunk_bytes(const ChunkRef& ref) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(ref);
  if (it == data_.end()) {
    return 0;
  }
  return it->second.bytes();
}

DatasetMetadata MemoryBackend::metadata(int32_t) const {
  // TODO: Implement metadata storage for MemoryBackend
  return DatasetMetadata{};
}

void MemoryBackend::set_chunk(const ChunkRef& ref, ChunkBuffer view) {
  std::lock_guard<std::mutex> lock(mutex_);
  data_[ref] = std::move(view);
}

DatasetBackendSnapshot MemoryBackend::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return DatasetBackendSnapshot{.kind = kind(), .memory_chunks = data_};
}

}  // namespace kangaroo
