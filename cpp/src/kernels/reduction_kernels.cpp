#include "default_kernel_families.hpp"

#include "kernel_buffer_support.hpp"
#include "kernel_param_support.hpp"

namespace kangaroo {

void register_reduction_kernels(KernelRegistry &registry) {
  {
    /**
     * @brief Adds two matching partial uniform-slice buffers elementwise.
     * @par Chunk inputs `inputs[0]` and `inputs[1]` are matching f32 or f64
     * arrays.
     * @par Typed parameters None.
     * @par Chunk outputs `outputs[0]` is their elementwise sum with matching
     * shape.
     */
    registry.register_kernel(
        KernelDesc{.name = "uniform_slice_add",
                   .n_inputs = 2,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &) {
          if (outputs.empty() || inputs.size() < 2) {
            return hpx::make_ready_future();
          }
          reduce_matching_real_buffers(inputs, outputs[0]);
          return hpx::make_ready_future();
        });
  }
  {
    using Params = SliceFinalizeParams;

    /**
     * @brief Converts accumulated slice sums and areas into finalized pixel
     * values.
     * @par Chunk inputs `inputs[0]` and `inputs[1]` are matching f64 value-sum
     * and sampled-area images.
     * @par Typed parameters `pixel_area` is the physical area of one
     * output pixel.
     * @par Chunk outputs `outputs[0]` is an f32 or f64 image of value sum
     * divided by `pixel_area`, with NaN where sampled area is zero.
     */
    registry.register_kernel(
        KernelDesc{.name = "uniform_slice_finalize",
                   .n_inputs = 2,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &kernel_params) {
          const auto &params = require_kernel_params<Params>(
              kernel_params, "uniform_slice_finalize");

          if (outputs.empty() || inputs.size() < 2 ||
              params.pixel_area == 0.0) {
            return hpx::make_ready_future();
          }

          if (inputs[0].desc().scalar != ScalarType::kF64 ||
              inputs[1].desc().scalar != ScalarType::kF64 ||
              inputs[0].desc().element_count() !=
                  outputs[0].desc().element_count() ||
              inputs[1].desc().element_count() !=
                  outputs[0].desc().element_count()) {
            throw BufferContractError(
                BufferContractReason::kDescriptorStorageMismatch,
                "slice finalize descriptor mismatch");
          }
          const auto count =
              static_cast<std::size_t>(outputs[0].desc().element_count());
          const auto sum = inputs[0].byte_view();
          const auto area = inputs[1].byte_view();
          auto out = outputs[0].mutable_byte_view();
          auto evaluate = [&](std::size_t index) {
            const double denominator =
                load_buffer_scalar<double>(area.data(), index);
            return denominator == 0.0
                       ? std::numeric_limits<double>::quiet_NaN()
                       : load_buffer_scalar<double>(sum.data(), index) /
                             params.pixel_area;
          };
          if (outputs[0].desc().scalar == ScalarType::kF64) {
            for (std::size_t index = 0; index < count; ++index)
              store_buffer_scalar(out.data(), index, evaluate(index));
          } else if (outputs[0].desc().scalar == ScalarType::kF32) {
            for (std::size_t index = 0; index < count; ++index)
              store_buffer_scalar(out.data(), index,
                                  static_cast<float>(evaluate(index)));
          } else {
            throw BufferContractError(
                BufferContractReason::kScalarMismatch,
                "slice finalize output must be f32 or f64");
          }
          return hpx::make_ready_future();
        });
  }
  {
    /**
     * @brief Reduces matching partial uniform-slice buffers by elementwise
     * addition.
     * @par Chunk inputs `inputs[0..N)` are matching f32 or f64 partial arrays.
     * @par Typed parameters None.
     * @par Chunk outputs `outputs[0]` is their elementwise sum with matching
     * shape.
     */
    registry.register_kernel(
        KernelDesc{.name = "uniform_slice_reduce",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           const KernelParamsIR &) {
          if (outputs.empty() || inputs.empty()) {
            return hpx::make_ready_future();
          }
          reduce_matching_real_buffers(inputs, outputs[0]);
          return hpx::make_ready_future();
        });
  }
}

} // namespace kangaroo
