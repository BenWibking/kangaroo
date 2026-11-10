#pragma once

namespace kangaroo {

class KernelRegistry;

void register_core_kernels(KernelRegistry &registry);
void register_grid_kernels(KernelRegistry &registry);
void register_particle_kernels(KernelRegistry &registry);
void register_histogram_kernels(KernelRegistry &registry);
void register_flux_kernels(KernelRegistry &registry);
void register_reduction_kernels(KernelRegistry &registry);
void register_toomre_kernels(KernelRegistry &registry);

} // namespace kangaroo
