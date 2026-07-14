#include "kernel_param_support.hpp"

namespace kangaroo {

std::size_t
covered_box_count(const std::shared_ptr<const CoveredBoxListIR> &boxes) {
  return boxes ? boxes->size() : 0;
}

bool covered_box_contains(const CoveredBoxIR &box, int i, int j, int k) {
  return i >= box.lo[0] && i <= box.hi[0] && j >= box.lo[1] && j <= box.hi[1] &&
         k >= box.lo[2] && k <= box.hi[2];
}

} // namespace kangaroo
