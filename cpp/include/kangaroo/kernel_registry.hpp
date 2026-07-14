#pragma once

#include "kangaroo/buffer_resolution.hpp"
#include "kangaroo/kernel.hpp"
#include "kangaroo/plan_ir.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace kangaroo {

class KernelRegistry {
 public:
  void register_kernel(const KernelDesc& desc,
                       KernelFn fn,
                       DynamicOutputBoundFn dynamic_output_bound = {});
  std::shared_ptr<const KernelFn> get_shared_by_name(const std::string& name) const;
  std::shared_ptr<const DynamicOutputBoundEvaluator> get_dynamic_output_bound_by_name(
      const std::string& name) const;
  const KernelFn& get_by_name(const std::string& name) const;
  std::vector<KernelDesc> list_kernel_descs() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<const KernelFn>> kernels_;
  std::unordered_map<std::string, std::shared_ptr<const DynamicOutputBoundEvaluator>>
      dynamic_output_bounds_;
  std::unordered_map<std::string, KernelDesc> descs_;
};

}  // namespace kangaroo
