#pragma once

#include "kangaroo/kernel.hpp"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace kangaroo {

class KernelRegistry {
 public:
  void register_kernel(const KernelDesc& desc, KernelFn fn);
  const KernelFn& get_by_name(const std::string& name) const;
  std::vector<KernelDesc> list_kernel_descs() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, KernelFn> kernels_;
  std::unordered_map<std::string, KernelDesc> descs_;
};

}  // namespace kangaroo
