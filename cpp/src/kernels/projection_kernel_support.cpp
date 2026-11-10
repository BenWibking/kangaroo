#include "projection_kernel_support.hpp"

#include <cstdlib>
#include <iostream>

#include <hpx/runtime_local/get_locality_id.hpp>

namespace kangaroo {

namespace {

bool debug_projection_enabled() {
  static const bool enabled =
      std::getenv("KANGAROO_DEBUG_PROJECTION") != nullptr;
  return enabled;
}

} // namespace

void log_projection_kernel_summary(const char *kernel, int32_t level_index,
                                   int32_t block,
                                   std::size_t covered_boxes_count,
                                   std::size_t candidates,
                                   std::size_t covered_skips,
                                   std::size_t bounds_skips,
                                   std::size_t deposited, double out_sum) {
  if (!debug_projection_enabled()) {
    return;
  }
  std::cout << "[kangaroo][projection] kernel=" << kernel
            << " locality=" << hpx::get_locality_id()
            << " level=" << level_index << " block=" << block
            << " covered_boxes=" << covered_boxes_count
            << " candidates=" << candidates
            << " covered_skips=" << covered_skips
            << " bounds_skips=" << bounds_skips << " deposited=" << deposited
            << " output_sum=" << out_sum << std::endl;
}

} // namespace kangaroo
