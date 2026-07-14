#include "kangaroo/default_kernels.hpp"

#include "default_kernel_families.hpp"

namespace kangaroo {

void register_default_kernels(KernelRegistry &registry) {
  register_core_kernels(registry);
  register_grid_kernels(registry);
  register_particle_kernels(registry);
  register_histogram_kernels(registry);
  register_flux_kernels(registry);
  register_reduction_kernels(registry);
  register_toomre_kernels(registry);
}

} // namespace kangaroo
