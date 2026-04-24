#include "kangaroo/kernel_registry.hpp"

#include <stdexcept>

namespace kangaroo {

void KernelRegistry::register_kernel(const KernelDesc& desc,
                                    KernelFn fn,
                                    KernelParamsPrepareFn prepare_params) {
  std::lock_guard<std::mutex> lock(mutex_);
  kernels_[desc.name] = std::make_shared<KernelFn>(std::move(fn));
  if (prepare_params) {
    params_prepare_[desc.name] = std::move(prepare_params);
  } else {
    params_prepare_.erase(desc.name);
  }
  descs_[desc.name] = desc;
}

void KernelRegistry::register_kernel_params_preparer(const std::string& name,
                                                     KernelParamsPrepareFn prepare_params) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (kernels_.find(name) == kernels_.end()) {
    throw std::runtime_error("kernel not found: " + name);
  }
  if (prepare_params) {
    params_prepare_[name] = std::move(prepare_params);
  } else {
    params_prepare_.erase(name);
  }
}

KernelRegistry::PreparedParams KernelRegistry::prepare_params_by_name(
    const std::string& name,
    const KernelParamContext& context) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = params_prepare_.find(name);
  if (it == params_prepare_.end()) {
    return {};
  }
  return it->second(context);
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
