#include "kangaroo/chunk_buffer.hpp"

#include <cstddef>
#include <span>
#include <vector>

#ifndef KANGAROO_VISIT_ARITY
#define KANGAROO_VISIT_ARITY 1
#endif

namespace {

double instantiate_dtype_pack(std::span<const kangaroo::ChunkBuffer> inputs) {
#ifdef KANGAROO_VISIT_BASELINE
  double sum = 0.0;
  for (const auto& input : inputs) sum += input.array<double>()(0);
  return sum;
#else
  return kangaroo::visit_real_buffers_exact<KANGAROO_VISIT_ARITY>(
      inputs, [](auto... views) {
        double sum = 0.0;
        ((sum += static_cast<double>(views.array()(0))), ...);
        return sum;
      });
#endif
}

}  // namespace

int main() {
  std::vector<kangaroo::ChunkBuffer> inputs;
  inputs.reserve(KANGAROO_VISIT_ARITY);
  for (std::size_t i = 0; i < KANGAROO_VISIT_ARITY; ++i) {
    inputs.push_back(kangaroo::ChunkBuffer::allocate(
        kangaroo::BufferDesc::contiguous(
            kangaroo::ScalarType::kF64, std::array<std::uint64_t, 1>{1}),
        kangaroo::InitPolicy::kZero));
  }
  return instantiate_dtype_pack(inputs) == 0.0 ? 0 : 1;
}
