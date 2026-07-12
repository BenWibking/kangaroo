# Chunk Buffer performance fixtures

Run the compile-time dtype-pack and scalar-proxy fixtures in the HPX development environment:

```bash
pixi run -e pixi-hpx bash scripts/utils/run_chunk_buffer_compile_benchmark.sh
```

The script reports compiler identity, wall time, and object size for exact real-buffer
visitor arities 1 through 10. It also writes optimized assembly containing both the
`TensorView` scalar-proxy AXPY loop and a direct-pointer baseline. Inspect the two loop
bodies for native loads/stores and vectorization; no out-of-line `memcpy` call may remain.

Reference environment: Clang 19, C++20, `-O3`, Apple ARM64. The accepted local envelopes
are 20 seconds for arity 9 and 30 seconds for arity 10. Wall-clock thresholds are manual
performance gates and are not portable CI assertions.
