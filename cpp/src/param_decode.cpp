#include "kangaroo/param_decode.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kangaroo {

namespace {

struct ParamCacheEntry {
  explicit ParamCacheEntry(std::vector<std::uint8_t> bytes_in) : bytes(std::move(bytes_in)) {}

  std::vector<std::uint8_t> bytes;
  std::mutex mutex;
  bool root_ready = false;
  msgpack::object_handle root_handle;
  std::unordered_map<std::type_index, std::shared_ptr<const void>> decoded;
};

std::mutex& cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::size_t, std::vector<std::shared_ptr<ParamCacheEntry>>>& cache_entries() {
  static std::unordered_map<std::size_t, std::vector<std::shared_ptr<ParamCacheEntry>>> entries;
  return entries;
}

std::size_t hash_params(std::span<const std::uint8_t> bytes) {
  std::size_t hash = bytes.size();
  for (auto byte : bytes) {
    hash ^= static_cast<std::size_t>(byte) + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
  }
  return hash;
}

std::shared_ptr<ParamCacheEntry> get_or_create_entry(std::span<const std::uint8_t> params_msgpack) {
  const std::size_t hash = hash_params(params_msgpack);
  std::lock_guard<std::mutex> lock(cache_mutex());
  auto& bucket = cache_entries()[hash];
  for (const auto& entry : bucket) {
    if (entry->bytes.size() != params_msgpack.size()) {
      continue;
    }
    if (std::equal(entry->bytes.begin(), entry->bytes.end(), params_msgpack.begin())) {
      return entry;
    }
  }

  auto entry = std::make_shared<ParamCacheEntry>(
      std::vector<std::uint8_t>(params_msgpack.begin(), params_msgpack.end()));
  bucket.push_back(entry);
  return entry;
}

thread_local std::type_index g_prepared_params_type = std::type_index(typeid(void));
thread_local std::shared_ptr<const void> g_prepared_params;

}  // namespace

const msgpack::object* find_msgpack_map_value(const msgpack::object& obj, const char* key) {
  if (obj.type != msgpack::type::MAP) {
    return nullptr;
  }
  for (uint32_t i = 0; i < obj.via.map.size; ++i) {
    const auto& k = obj.via.map.ptr[i].key;
    if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
      return &obj.via.map.ptr[i].val;
    }
  }
  return nullptr;
}

const msgpack::object& cached_params_root(std::span<const std::uint8_t> params_msgpack) {
  static msgpack::object_handle empty_handle = []() {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> packer(buffer);
    packer.pack_nil();
    return msgpack::unpack(buffer.data(), buffer.size());
  }();
  if (params_msgpack.empty()) {
    return empty_handle.get();
  }

  auto entry = get_or_create_entry(params_msgpack);
  std::lock_guard<std::mutex> lock(entry->mutex);
  if (!entry->root_ready) {
    entry->root_handle =
        msgpack::unpack(reinterpret_cast<const char*>(entry->bytes.data()), entry->bytes.size());
    entry->root_ready = true;
  }
  return entry->root_handle.get();
}

ScopedPreparedParams::ScopedPreparedParams(std::type_index type, std::shared_ptr<const void> decoded)
    : previous_type_(g_prepared_params_type), previous_decoded_(std::move(g_prepared_params)) {
  g_prepared_params_type = type;
  g_prepared_params = std::move(decoded);
}

ScopedPreparedParams::~ScopedPreparedParams() {
  g_prepared_params_type = previous_type_;
  g_prepared_params = std::move(previous_decoded_);
}

std::shared_ptr<const void> detail::current_prepared_params(std::type_index type) {
  if (g_prepared_params && g_prepared_params_type == type) {
    return g_prepared_params;
  }
  return {};
}

std::shared_ptr<const void> detail::find_cached_params_decode(
    std::span<const std::uint8_t> params_msgpack,
    std::type_index type) {
  auto entry = get_or_create_entry(params_msgpack);
  std::lock_guard<std::mutex> lock(entry->mutex);
  auto it = entry->decoded.find(type);
  if (it != entry->decoded.end()) {
    return it->second;
  }
  return {};
}

void detail::store_cached_params_decode(std::span<const std::uint8_t> params_msgpack,
                                        std::type_index type,
                                        std::shared_ptr<const void> decoded) {
  auto entry = get_or_create_entry(params_msgpack);
  std::lock_guard<std::mutex> lock(entry->mutex);
  entry->decoded.emplace(type, std::move(decoded));
}

}  // namespace kangaroo
