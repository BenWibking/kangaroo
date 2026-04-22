#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <typeindex>

#include <msgpack.hpp>

namespace kangaroo {

const msgpack::object* find_msgpack_map_value(const msgpack::object& obj, const char* key);

const msgpack::object& cached_params_root(std::span<const std::uint8_t> params_msgpack);

template <typename T, typename DecodeFn>
const T& decode_params_cached(std::span<const std::uint8_t> params_msgpack, DecodeFn&& decode_fn);

namespace detail {

std::shared_ptr<const void> find_cached_params_decode(std::span<const std::uint8_t> params_msgpack,
                                                      std::type_index type);

void store_cached_params_decode(std::span<const std::uint8_t> params_msgpack,
                                std::type_index type,
                                std::shared_ptr<const void> decoded);

}  // namespace detail

template <typename T, typename DecodeFn>
const T& decode_params_cached(std::span<const std::uint8_t> params_msgpack, DecodeFn&& decode_fn) {
  const auto type = std::type_index(typeid(T));
  if (auto decoded = detail::find_cached_params_decode(params_msgpack, type)) {
    return *static_cast<const T*>(decoded.get());
  }

  auto decoded = std::shared_ptr<const void>(
      new T(decode_fn(cached_params_root(params_msgpack))),
      [](const void* ptr) { delete static_cast<const T*>(ptr); });
  detail::store_cached_params_decode(params_msgpack, type, decoded);
  return *static_cast<const T*>(decoded.get());
}

}  // namespace kangaroo
