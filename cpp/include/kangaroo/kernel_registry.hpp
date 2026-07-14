#pragma once

#include "kangaroo/buffer_resolution.hpp"
#include "kangaroo/kernel.hpp"
#include "kangaroo/plan_ir.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <type_traits>
#include <utility>
#include <vector>

namespace kangaroo {

class KernelRegistry {
 public:
  void register_kernel(const KernelDesc& desc,
                       KernelFn fn,
                       DynamicOutputBoundFn dynamic_output_bound = {});

  template <typename Params>
  void register_typed_kernel(
      const KernelDesc& desc,
      KernelFn fn,
      DynamicOutputBoundFn dynamic_output_bound = {}) {
    register_kernel_impl(
        desc, std::move(fn), std::move(dynamic_output_bound),
        [](const KernelParamsIR& params) {
          return std::holds_alternative<std::decay_t<Params>>(params);
        });
  }

  std::shared_ptr<const KernelFn> get_shared_by_name(const std::string& name) const;
  std::shared_ptr<const DynamicOutputBoundEvaluator> get_dynamic_output_bound_by_name(
      const std::string& name) const;
  const KernelFn& get_by_name(const std::string& name) const;
  void validate_params_by_name(const std::string& name,
                               const KernelParamsIR& params) const;
  std::vector<KernelDesc> list_kernel_descs() const;

 private:
  using KernelParamsValidator = std::function<bool(const KernelParamsIR&)>;

  void register_kernel_impl(
      const KernelDesc& desc,
      KernelFn fn,
      DynamicOutputBoundFn dynamic_output_bound,
      KernelParamsValidator params_validator);

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<const KernelFn>> kernels_;
  std::unordered_map<std::string, std::shared_ptr<const KernelParamsValidator>>
      params_validators_;
  std::unordered_map<std::string, std::shared_ptr<const DynamicOutputBoundEvaluator>>
      dynamic_output_bounds_;
  std::unordered_map<std::string, KernelDesc> descs_;
};

}  // namespace kangaroo
