# Chunk Buffer performance fixtures

Run the compile-time dtype-pack and scalar-proxy fixtures in the HPX development environment:

```bash
pixi run -e pixi-hpx bash scripts/utils/run_chunk_buffer_compile_benchmark.sh
```

The script reports compiler identity, wall time, and object size for exact real-buffer
visitor arities 1 through 10. It also writes optimized assembly containing both the
`TensorView` scalar-proxy AXPY loop and a direct-pointer baseline.

The required scalar-proxy code-generation gates are:

- scalar values lower to native loads, arithmetic, and stores;
- no out-of-line `memcpy` call remains in the inner loop;
- stride interpretation stays inside the proxy implementation rather than leaking into
  scientific kernels.

Vectorization is not required for a runtime-strided `TensorView`. Its strides are known
only when a runtime plan executes, so the compiler cannot generally prove unit-stride,
contiguous, non-aliasing access. The direct-pointer loop remains a comparison baseline,
not an equivalence requirement. A future named-layout fast path may dispatch once to a
precompiled contiguous specialization where direct-pointer-equivalent vectorization is
expected.

Reference environment: Clang 19, C++20, `-O3`, Apple ARM64. The accepted local envelopes
are 20 seconds for arity 9 and 30 seconds for arity 10. Wall-clock thresholds are manual
performance gates and are not portable CI assertions.
