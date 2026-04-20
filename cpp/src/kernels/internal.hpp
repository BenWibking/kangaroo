#pragma once

#include "kangaroo/kernel.hpp"
#include "kangaroo/kernel_registry.hpp"
#include "kangaroo/runmeta.hpp"

namespace kangaroo {

void register_default_kernels(KernelRegistry& registry);
void register_legacy_kernels(KernelRegistry& registry);

namespace runtime_kernels {

#define KANGAROO_KERNEL(NAME) \
  hpx::future<void> NAME(\
      const LevelMeta& level, \
      int32_t block_index, \
      std::span<const HostView> self_inputs, \
      const NeighborViews& nbr_inputs, \
      std::span<HostView> outputs, \
      std::span<const std::uint8_t> params_msgpack)

#define KANGAROO_REGISTER_KERNEL_WITH_SCHEMA(\
    REGISTRY, NAME, N_INPUTS, N_OUTPUTS, NEEDS_NEIGHBORS, PARAM_SCHEMA_JSON) \
  REGISTRY.register_kernel(\
      KernelDesc{\
          .name = #NAME, \
          .n_inputs = N_INPUTS, \
          .n_outputs = N_OUTPUTS, \
          .needs_neighbors = NEEDS_NEIGHBORS, \
          .param_schema_json = PARAM_SCHEMA_JSON}, \
      &::kangaroo::runtime_kernels::NAME)

#define KANGAROO_REGISTER_KERNEL(REGISTRY, NAME, N_INPUTS, N_OUTPUTS, NEEDS_NEIGHBORS) \
  KANGAROO_REGISTER_KERNEL_WITH_SCHEMA(REGISTRY, NAME, N_INPUTS, N_OUTPUTS, NEEDS_NEIGHBORS, "")

KANGAROO_KERNEL(plotfile_load);
KANGAROO_KERNEL(particle_eq_mask);
KANGAROO_KERNEL(particle_isin_mask);
KANGAROO_KERNEL(particle_isfinite_mask);
KANGAROO_KERNEL(particle_abs_lt_mask);
KANGAROO_KERNEL(particle_le_mask);
KANGAROO_KERNEL(particle_gt_mask);
KANGAROO_KERNEL(particle_and_mask);
KANGAROO_KERNEL(particle_filter);
KANGAROO_KERNEL(particle_subtract);
KANGAROO_KERNEL(particle_distance3);
KANGAROO_KERNEL(particle_sum);
KANGAROO_KERNEL(particle_count);
KANGAROO_KERNEL(particle_len_f64);
KANGAROO_KERNEL(particle_min);
KANGAROO_KERNEL(particle_max);
KANGAROO_KERNEL(uniform_slice_add);
KANGAROO_KERNEL(uniform_slice_finalize);
KANGAROO_KERNEL(uniform_slice_reduce);

}  // namespace runtime_kernels

}  // namespace kangaroo
