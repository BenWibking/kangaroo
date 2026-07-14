#pragma once

#include <cstddef>
#include <cstdint>

namespace kangaroo {

void log_projection_kernel_summary(const char *kernel, int32_t level_index,
                                   int32_t block,
                                   std::size_t covered_boxes_count,
                                   std::size_t candidates,
                                   std::size_t covered_skips,
                                   std::size_t bounds_skips,
                                   std::size_t deposited, double out_sum);

} // namespace kangaroo
