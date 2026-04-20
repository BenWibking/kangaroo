#include "internal.hpp"

namespace kangaroo {

void register_default_kernels(KernelRegistry& registry) {
  KANGAROO_REGISTER_KERNEL(registry, plotfile_load, 0, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_eq_mask, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_isin_mask, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_isfinite_mask, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_abs_lt_mask, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_le_mask, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_gt_mask, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_and_mask, 2, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_filter, 2, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_subtract, 2, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_distance3, 6, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_sum, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_count, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_len_f64, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_min, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, particle_max, 1, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, uniform_slice_add, 2, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, uniform_slice_finalize, 2, 1, false);
  KANGAROO_REGISTER_KERNEL(registry, uniform_slice_reduce, 1, 1, false);
  register_legacy_kernels(registry);
}

}  // namespace kangaroo
