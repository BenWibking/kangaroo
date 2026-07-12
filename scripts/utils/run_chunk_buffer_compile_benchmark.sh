#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
output_dir=${1:-/tmp/kangaroo-chunk-buffer-benchmark}
cxx=${CXX:-clang++}
mkdir -p "$output_dir"

"$cxx" --version | head -n 1
for arity in $(seq 1 10); do
  baseline_object="$output_dir/baseline-$arity.o"
  object="$output_dir/dtype-pack-$arity.o"
  baseline_start=$SECONDS
  "$cxx" -std=c++20 -O3 -DNDEBUG \
    -DKANGAROO_VISIT_ARITY="$arity" -DKANGAROO_VISIT_BASELINE=1 \
    -I"$root_dir/cpp/include" -I"${CONDA_PREFIX:?}/include" \
    -c "$root_dir/scripts/utils/chunk_buffer_dtype_pack_benchmark.cpp" \
    -o "$baseline_object"
  baseline_elapsed=$((SECONDS - baseline_start))
  baseline_size=$(wc -c < "$baseline_object" | tr -d ' ')
  start=$SECONDS
  "$cxx" -std=c++20 -O3 -DNDEBUG \
    -DKANGAROO_VISIT_ARITY="$arity" \
    -I"$root_dir/cpp/include" -I"${CONDA_PREFIX:?}/include" \
    -c "$root_dir/scripts/utils/chunk_buffer_dtype_pack_benchmark.cpp" \
    -o "$object"
  elapsed=$((SECONDS - start))
  size=$(wc -c < "$object" | tr -d ' ')
  printf 'arity=%d baseline_seconds=%d baseline_bytes=%s visitor_seconds=%d visitor_bytes=%s\n' \
    "$arity" "$baseline_elapsed" "$baseline_size" "$elapsed" "$size"
done

"$cxx" -std=c++20 -O3 -S \
  -I"$root_dir/cpp/include" -I"${CONDA_PREFIX:?}/include" \
  "$root_dir/scripts/utils/chunk_buffer_scalar_proxy_codegen.cpp" \
  -o "$output_dir/scalar-proxy.s"
printf 'assembly=%s\n' "$output_dir/scalar-proxy.s"
