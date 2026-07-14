#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIXI_PY="${CONDA_PREFIX:-$REPO_ROOT/.pixi/envs/spack-hpx}/bin/python"

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

if [[ -z "${SPACK_HPX_PREFIX:-}" ]]; then
  if [[ -n "${HPX_DIR:-}" ]]; then
    SPACK_HPX_PREFIX="$(cd "$HPX_DIR/../../.." && pwd)"
  elif command -v spack >/dev/null 2>&1; then
    SPACK_HPX_PREFIX="$(spack location -i "hpx networking=mpi" 2>/dev/null || true)"
  fi
fi

if [[ -z "${SPACK_HPX_PREFIX:-}" || ! -d "$SPACK_HPX_PREFIX" ]]; then
  echo "Set SPACK_HPX_PREFIX to the Spack HPX install with MPI parcelport enabled." >&2
  exit 1
fi

HPX_DIR="${HPX_DIR:-$(find_hpx_dir_from_prefix "$SPACK_HPX_PREFIX")}"
if [[ -f "$HPX_DIR/HPXCacheVariables.cmake" ]] &&
   ! grep -Eq 'set\(HPX_WITH_PARCELPORT_MPI[[:space:]]+ON\)' "$HPX_DIR/HPXCacheVariables.cmake"; then
  echo "Selected Spack HPX does not advertise HPX_WITH_PARCELPORT_MPI=ON: $SPACK_HPX_PREFIX" >&2
  exit 1
fi

if [[ ! -x "$PIXI_PY" ]]; then
  echo "pixi python not found at $PIXI_PY" >&2
  exit 1
fi
if [[ ! -f "$SPACK_HPX_PREFIX/bin/hpxrun.py" ]]; then
  echo "hpxrun.py not found at $SPACK_HPX_PREFIX/bin/hpxrun.py" >&2
  exit 1
fi

SPACK_MPI_PREFIX="$(resolve_spack_mpi_prefix)" || {
  echo "Could not resolve a Spack MPI prefix. Set SPACK_MPI_PREFIX to the MPI used to build HPX." >&2
  exit 1
}

export PATH="$SPACK_MPI_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SPACK_HPX_PREFIX/lib64:$SPACK_HPX_PREFIX/lib:$SPACK_MPI_PREFIX/lib64:$SPACK_MPI_PREFIX/lib:${LD_LIBRARY_PATH:-}"

if [[ $# -eq 0 ]]; then
  exec "$PIXI_PY" "$SPACK_HPX_PREFIX/bin/hpxrun.py"
fi

if [[ "$1" == -* || -x "$1" ]]; then
  exec "$PIXI_PY" "$SPACK_HPX_PREFIX/bin/hpxrun.py" "$@"
fi

exec "$PIXI_PY" "$SPACK_HPX_PREFIX/bin/hpxrun.py" "$PIXI_PY" "$@"
