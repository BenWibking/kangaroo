#pragma once

#include "kangaroo/kernel.hpp"
#include "kangaroo/plan_ir.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace kangaroo {

struct KernelParamContext {
  std::span<const std::uint8_t> params_msgpack;
  std::shared_ptr<const CoveredBoxListIR> covered_boxes;
};

class KernelRegistry {
 public:
  struct PreparedParams {
    std::type_index type{typeid(void)};
    std::shared_ptr<const void> value;

    explicit operator bool() const { return value != nullptr; }
  };

  using KernelParamsPrepareFn =
      std::function<PreparedParams(const KernelParamContext& context)>;

  void register_kernel(const KernelDesc& desc,
                       KernelFn fn,
                       KernelParamsPrepareFn prepare_params = {});
  void register_kernel_params_preparer(const std::string& name, KernelParamsPrepareFn prepare_params);
  PreparedParams prepare_params_by_name(const std::string& name,
                                        const KernelParamContext& context) const;
  std::shared_ptr<const KernelFn> get_shared_by_name(const std::string& name) const;
  const KernelFn& get_by_name(const std::string& name) const;
  std::vector<KernelDesc> list_kernel_descs() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<const KernelFn>> kernels_;
  std::unordered_map<std::string, KernelParamsPrepareFn> params_prepare_;
  std::unordered_map<std::string, KernelDesc> descs_;
};

}  // namespace kangaroo
