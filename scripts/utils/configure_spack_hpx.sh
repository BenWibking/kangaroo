#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${KANGAROO_BUILD_DIR:-$REPO_ROOT/.pixi/build-spack-hpx-mpi}"
PIXI_ENV="${CONDA_PREFIX:-$REPO_ROOT/.pixi/envs/spack-hpx}"

find_hpx_dir_from_prefix() {
  local prefix="$1"
  local cand
  for cand in "$prefix/lib64/cmake/HPX" "$prefix/lib/cmake/HPX" "$prefix/share/HPX/cmake" "$prefix/share/HPX"; do
    if [[ -d "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

resolve_spack_hpx_prefix() {
  if [[ -n "${SPACK_HPX_PREFIX:-}" ]]; then
    echo "$SPACK_HPX_PREFIX"
    return 0
  fi

  if [[ -n "${HPX_DIR:-}" ]]; then
    local cmake_dir
    cmake_dir="$(cd "$HPX_DIR/../../.." && pwd)"
    echo "$cmake_dir"
    return 0
  fi

  if command -v spack >/dev/null 2>&1; then
    local spec prefix
    for spec in "hpx networking=mpi" "hpx+mpi" "hpx"; do
      prefix="$(spack location -i "$spec" 2>/dev/null || true)"
      if [[ -n "$prefix" && -d "$prefix" ]]; then
        echo "$prefix"
        return 0
      fi
    done
  fi

  return 1
}


resolve_spack_mpi_prefix() {
  if [[ -n "${SPACK_MPI_PREFIX:-}" ]]; then
    echo "$SPACK_MPI_PREFIX"
    return 0
  fi

  if command -v spack >/dev/null 2>&1; then
    local spec prefix
    for spec in "openmpi%gcc@14.2.0" "openmpi" "mpi"; do
      prefix="$(spack location -i "$spec" 2>/dev/null || true)"
      if [[ -n "$prefix" && -d "$prefix" ]]; then
        echo "$prefix"
        return 0
      fi
    done
  fi

  return 1
}

require_mpi_parcelport() {
  local hpx_dir="$1"
  local hpx_prefix="$2"
  local cache_vars="$hpx_dir/HPXCacheVariables.cmake"

  if [[ -f "$cache_vars" ]] && grep -Eq 'set\(HPX_WITH_PARCELPORT_MPI[[:space:]]+ON\)' "$cache_vars"; then
    return 0
  fi

  if compgen -G "$hpx_prefix/lib64/libhpx_parcelport_mpi*" >/dev/null ||
     compgen -G "$hpx_prefix/lib/libhpx_parcelport_mpi*" >/dev/null; then
    return 0
  fi

  echo "Selected Spack HPX does not appear to include the MPI parcelport:" >&2
  echo "  SPACK_HPX_PREFIX=$hpx_prefix" >&2
  echo "Expected HPX_WITH_PARCELPORT_MPI=ON or libhpx_parcelport_mpi*." >&2
  echo "Set SPACK_HPX_PREFIX or HPX_DIR to an HPX build configured with networking=mpi." >&2
  return 1
}

if [[ ! -x "$PIXI_ENV/bin/python" ]]; then
  echo "Pixi environment not found at $PIXI_ENV." >&2
  echo "Run 'pixi install -e spack-hpx' first." >&2
  exit 1
fi

SPACK_HPX_PREFIX="$(resolve_spack_hpx_prefix)" || {
  echo "Could not resolve a Spack HPX prefix. Set SPACK_HPX_PREFIX or HPX_DIR." >&2
  exit 1
}
HPX_DIR="${HPX_DIR:-$(find_hpx_dir_from_prefix "$SPACK_HPX_PREFIX")}" 

if [[ ! -d "$HPX_DIR" ]]; then
  echo "HPX_DIR does not exist: $HPX_DIR" >&2
  exit 1
fi

require_mpi_parcelport "$HPX_DIR" "$SPACK_HPX_PREFIX"

SPACK_MPI_PREFIX="$(resolve_spack_mpi_prefix)" || {
  echo "Could not resolve a Spack MPI prefix. Set SPACK_MPI_PREFIX to the MPI used to build HPX." >&2
  exit 1
}

mpi_cc="${MPI_C_COMPILER:-$SPACK_MPI_PREFIX/bin/mpicc}"
mpi_cxx="${MPI_CXX_COMPILER:-$SPACK_MPI_PREFIX/bin/mpicxx}"
mpi_exec="${MPIEXEC_EXECUTABLE:-$SPACK_MPI_PREFIX/bin/mpiexec}"

if [[ ! -x "$mpi_cxx" ]]; then
  echo "MPI C++ wrapper not found: $mpi_cxx" >&2
  exit 1
fi
if [[ ! -x "$mpi_cc" ]]; then
  echo "MPI C wrapper not found: $mpi_cc" >&2
  exit 1
fi

export PATH="$SPACK_MPI_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SPACK_HPX_PREFIX/lib64:$SPACK_HPX_PREFIX/lib:$SPACK_MPI_PREFIX/lib64:$SPACK_MPI_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cc="${CC:-}"
cxx="${CXX:-}"
if [[ -z "$cc" || "$cc" == "$PIXI_ENV/"* ]]; then
  if [[ -x /sw/andes/gcc/14.2.0/bin/gcc ]]; then
    cc=/sw/andes/gcc/14.2.0/bin/gcc
  else
    cc="$(command -v gcc || true)"
  fi
fi
if [[ -z "$cxx" || "$cxx" == "$PIXI_ENV/"* ]]; then
  if [[ -x /sw/andes/gcc/14.2.0/bin/g++ ]]; then
    cxx=/sw/andes/gcc/14.2.0/bin/g++
  else
    cxx="$(command -v g++ || true)"
  fi
fi

if [[ -z "$cc" || -z "$cxx" ]]; then
  echo "Unable to resolve gcc/g++; set CC and CXX explicitly." >&2
  exit 1
fi

cmake -S "$REPO_ROOT/cpp" -B "$BUILD_DIR"   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_C_COMPILER="$cc"   -DCMAKE_CXX_COMPILER="$cxx"   -DMPI_C_COMPILER="$mpi_cc"   -DMPI_CXX_COMPILER="$mpi_cxx"   -DMPIEXEC_EXECUTABLE="$mpi_exec"   -DHPX_DIR="$HPX_DIR"   -DCMAKE_PREFIX_PATH="$PIXI_ENV;$SPACK_HPX_PREFIX;$SPACK_MPI_PREFIX"   -DHDF5_ROOT="$PIXI_ENV"   -DHDF5_C_COMPILER_EXECUTABLE="$PIXI_ENV/bin/h5cc"   -Dnanobind_DIR="$("$PIXI_ENV/bin/python" -m nanobind --cmake_dir)"   -DCMAKE_INSTALL_PREFIX="$PIXI_ENV"   -DPython_EXECUTABLE="$PIXI_ENV/bin/python"
