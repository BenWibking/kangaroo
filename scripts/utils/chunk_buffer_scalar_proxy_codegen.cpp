#include "kangaroo/chunk_buffer.hpp"

#include <cstddef>

extern "C" void chunk_buffer_proxy_axpy(kangaroo::TensorView<double, 1> out,
                                        kangaroo::TensorView<const double, 1> x,
                                        kangaroo::TensorView<const double, 1> y,
                                        double alpha,
                                        std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) out(i) = alpha * x(i) + y(i);
}

extern "C" void direct_pointer_axpy(double* out, const double* x, const double* y,
                                    double alpha, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) out[i] = alpha * x[i] + y[i];
}
