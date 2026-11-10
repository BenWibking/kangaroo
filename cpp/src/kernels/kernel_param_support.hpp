#pragma once

#include "kangaroo/kernel_params.hpp"
#include "kangaroo/kernel_registry.hpp"

#include <memory>

namespace kangaroo {

std::size_t
covered_box_count(const std::shared_ptr<const CoveredBoxListIR> &boxes);
bool covered_box_contains(const CoveredBoxIR &box, int i, int j, int k);

} // namespace kangaroo
