#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIXI_ENV="$REPO_ROOT/.pixi/envs/default"
BUILD_DIR="${KANGAROO_BUILD_DIR:-$REPO_ROOT/.pixi/build-external-hpx}"

if [[ ! -x "$PIXI_ENV/bin/python" ]]; then
  echo "Pixi environment not found at $PIXI_ENV. Run 'pixi install' first." >&2
  exit 1
fi

if [[ -z "${HPX_DIR:-}" && -n "${OLCF_HPX_ROOT:-}" ]]; then
  HPX_DIR="$OLCF_HPX_ROOT/lib64/cmake/HPX"
fi

if [[ -z "${HPX_DIR:-}" ]]; then
  echo "HPX_DIR is not set. Load an external HPX module and export HPX_DIR first." >&2
  exit 1
fi

if [[ ! -d "$HPX_DIR" ]]; then
  echo "HPX_DIR does not exist: $HPX_DIR" >&2
  exit 1
fi

cc="${CC:-$(command -v gcc)}"
cxx="${CXX:-$(command -v g++)}"

if [[ -z "${cc:-}" || -z "${cxx:-}" ]]; then
  echo "Unable to resolve gcc/g++; set CC and CXX explicitly." >&2
  exit 1
fi

cmake -S "$REPO_ROOT/cpp" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$cc" \
  -DCMAKE_CXX_COMPILER="$cxx" \
  -DHPX_DIR="$HPX_DIR" \
  -DCMAKE_PREFIX_PATH="$PIXI_ENV" \
  -Dnanobind_DIR="$("$PIXI_ENV/bin/python" -m nanobind --cmake_dir)" \
  -DCMAKE_INSTALL_PREFIX="$PIXI_ENV" \
  -DPython_EXECUTABLE="$PIXI_ENV/bin/python"
