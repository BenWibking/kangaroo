#pragma once

#include "kangaroo/kernel_registry.hpp"
#include "kangaroo/param_decode.hpp"

#include <memory>
#include <typeindex>
#include <utility>

#include <msgpack.hpp>

namespace kangaroo {

template <typename Params, typename DecodeFn>
KernelRegistry::KernelParamsPrepareFn
make_kernel_params_preparer(DecodeFn decode_fn) {
  return [decode_fn = std::move(decode_fn)](
             const KernelParamContext &context) -> KernelRegistry::PreparedParams {
    if (context.params_msgpack.empty()) {
      return {};
    }
    auto decoded = decode_fn(cached_params_root(context.params_msgpack));
    auto prepared = std::make_shared<const Params>(std::move(decoded));
    return KernelRegistry::PreparedParams{std::type_index(typeid(Params)),
                                          std::move(prepared)};
  };
}

template <typename Params, typename DecodeFn>
KernelRegistry::KernelParamsPrepareFn
make_covered_box_params_preparer(DecodeFn decode_fn) {
  return [decode_fn = std::move(decode_fn)](
             const KernelParamContext &context) -> KernelRegistry::PreparedParams {
    if (context.params_msgpack.empty() && !context.covered_boxes) {
      return {};
    }
    Params decoded;
    if (!context.params_msgpack.empty()) {
      decoded = decode_fn(cached_params_root(context.params_msgpack));
    }
    if (context.covered_boxes) {
      decoded.covered_boxes = context.covered_boxes;
    }
    auto prepared = std::make_shared<const Params>(std::move(decoded));
    return KernelRegistry::PreparedParams{std::type_index(typeid(Params)),
                                          std::move(prepared)};
  };
}

std::shared_ptr<const CoveredBoxListIR>
parse_covered_boxes_param(const msgpack::object &root);
std::size_t
covered_box_count(const std::shared_ptr<const CoveredBoxListIR> &boxes);
bool covered_box_contains(const CoveredBoxIR &box, int i, int j, int k);

} // namespace kangaroo
