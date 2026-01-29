#include "kangaroo/kernel_registry.hpp"

#include <stdexcept>

namespace kangaroo {

void KernelRegistry::register_kernel(const KernelDesc& desc, KernelFn fn) {
  std::lock_guard<std::mutex> lock(mutex_);
  kernels_[desc.name] = std::move(fn);
  descs_[desc.name] = desc;
}

const KernelFn& KernelRegistry::get_by_name(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    throw std::runtime_error("kernel not found: " + name);
  }
  return it->second;
}

std::vector<KernelDesc> KernelRegistry::list_kernel_descs() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KernelDesc> out;
  out.reserve(descs_.size());
  for (const auto& [name, desc] : descs_) {
    (void)name;
    out.push_back(desc);
  }
  return out;
}

}  // namespace kangaroo
