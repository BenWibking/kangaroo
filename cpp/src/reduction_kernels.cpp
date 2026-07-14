#include "default_kernel_families.hpp"

#include "default_kernel_support.hpp"

namespace kangaroo {

void register_reduction_kernels(KernelRegistry &registry) {
  {
    registry.register_kernel(
        KernelDesc{.name = "uniform_slice_add",
                   .n_inputs = 2,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           std::span<const std::uint8_t>) {
          if (outputs.empty() || inputs.size() < 2) {
            return hpx::make_ready_future();
          }
          reduce_matching_real_buffers(inputs, outputs[0]);
          return hpx::make_ready_future();
        });
  }
  {
    struct Params {
      double pixel_area = 1.0;
    };

    auto decode_params = [](const msgpack::object &root) {
      Params params;
      if (const auto *area = find_msgpack_map_value(root, "pixel_area");
          area && (area->type == msgpack::type::FLOAT ||
                   area->type == msgpack::type::POSITIVE_INTEGER ||
                   area->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.pixel_area = area->as<double>();
      }
      return params;
    };
    registry.register_kernel(
        KernelDesc{.name = "uniform_slice_finalize",
                   .n_inputs = 2,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta &, int32_t,
                        std::span<const ChunkBuffer> inputs,
                        const NeighborViews &, std::span<ChunkBuffer> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto &params =
              decode_params_cached<Params>(params_msgpack, decode_params);

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
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    registry.register_kernel(
        KernelDesc{.name = "uniform_slice_reduce",
                   .n_inputs = 1,
                   .n_outputs = 1,
                   .needs_neighbors = false},
        [](const LevelMeta &, int32_t, std::span<const ChunkBuffer> inputs,
           const NeighborViews &, std::span<ChunkBuffer> outputs,
           std::span<const std::uint8_t>) {
          if (outputs.empty() || inputs.empty()) {
            return hpx::make_ready_future();
          }
          reduce_matching_real_buffers(inputs, outputs[0]);
          return hpx::make_ready_future();
        });
  }
}

} // namespace kangaroo
