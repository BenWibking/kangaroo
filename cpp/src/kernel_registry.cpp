#include "kangaroo/kernel_registry.hpp"

#include <stdexcept>

namespace kangaroo {

void KernelRegistry::register_kernel(const KernelDesc& desc,
                                    KernelFn fn,
                                    DynamicOutputBoundFn dynamic_output_bound) {
  std::lock_guard<std::mutex> lock(mutex_);
  kernels_[desc.name] = std::make_shared<KernelFn>(std::move(fn));
  if (dynamic_output_bound) {
    dynamic_output_bounds_[desc.name] =
        std::make_shared<const DynamicOutputBoundEvaluator>(std::move(dynamic_output_bound));
  } else {
    dynamic_output_bounds_.erase(desc.name);
  }
  descs_[desc.name] = desc;
}

std::shared_ptr<const DynamicOutputBoundEvaluator> KernelRegistry::get_dynamic_output_bound_by_name(
    const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = dynamic_output_bounds_.find(name);
  return it == dynamic_output_bounds_.end() ? nullptr : it->second;
}

std::shared_ptr<const KernelFn> KernelRegistry::get_shared_by_name(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    throw std::runtime_error("kernel not found: " + name);
  }
  return it->second;
}

const KernelFn& KernelRegistry::get_by_name(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    throw std::runtime_error("kernel not found: " + name);
  }
  return *it->second;
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
