#pragma once

#include "kangaroo/plan_ir.hpp"

#include <span>
#include <stdexcept>

namespace kangaroo {

PlanIR decode_plan_flatbuffer(std::span<const std::uint8_t> payload);

} // namespace kangaroo
