#!/usr/bin/env bash
# Shared setup for make_*.submit; source after resolving REPO_ROOT.

kangaroo_setup_runtime() {
  REPO_ROOT=$(cd "$REPO_ROOT" && pwd -P)
  export KANGAROO_REPO_ROOT="$REPO_ROOT"
  export KANGAROO_HPX_BACKEND="${KANGAROO_HPX_BACKEND:-pixi-hpx}"
  NNODES=${SLURM_NNODES:-1}
  CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-32}
  if [[ "${SLURM_TRES_PER_TASK:-}" =~ (^|,)cpu=([0-9]+)(,|$) ]]; then
    CPUS_PER_TASK=${BASH_REMATCH[2]}
    export SLURM_CPUS_PER_TASK="$CPUS_PER_TASK"
  fi

  case "$KANGAROO_HPX_BACKEND" in
    pixi-hpx)
      if [[ "$NNODES" != 1 || "${SLURM_NTASKS:-1}" != 1 ]]; then
        echo 'Pixi HPX has networking disabled; request one node and one task. Use KANGAROO_HPX_BACKEND=spack-hpx for MPI jobs.' >&2
        exit 2
      fi
      ;;
    spack-hpx)
      # Keep the MPI parcelport paired with the original cluster runtime.
      if ! type module >/dev/null 2>&1; then
        source /sw/riker/lmod/lmod/init/bash
      fi
      : "${MODULEPATH:=/etc/scl/modulefiles:/sw/riker/spack/modules/gcc/14.2.0/mpich-5.0.1/gcc-14.2.0:/sw/riker/spack/modules/gcc/14.2.0/gcc-14.2.0:/sw/riker/spack/modules/Core/26.01:/sw/riker/modulefiles}"
      export MODULEPATH
      module load gcc/14.2.0
      if ! command -v spack >/dev/null 2>&1; then
        export PATH="$HOME/spack/bin:$PATH"
      fi
      SPACK_HPX_PREFIX=${SPACK_HPX_PREFIX:-$(spack location -i 'hpx@1.11.0 networking=mpi %gcc@14.2.0')}
      SPACK_MPI_PREFIX=${SPACK_MPI_PREFIX:-$(spack location -i 'openmpi%gcc@14.2.0')}
      export SPACK_HPX_PREFIX SPACK_MPI_PREFIX
      ;;
    *)
      echo "Unknown KANGAROO_HPX_BACKEND='$KANGAROO_HPX_BACKEND' (use pixi-hpx or spack-hpx)" >&2
      exit 2
      ;;
  esac

  KANGAROO_PYTHON="$REPO_ROOT/.pixi/envs/$KANGAROO_HPX_BACKEND/bin/python"
  if [[ ! -x "$KANGAROO_PYTHON" ]]; then
    echo "Missing $KANGAROO_PYTHON; run pixi install -e $KANGAROO_HPX_BACKEND and install Kangaroo into that environment." >&2
    exit 2
  fi
  # Do not inherit the interpreter choice from an activated default environment.
  export CONDA_PREFIX="$REPO_ROOT/.pixi/envs/$KANGAROO_HPX_BACKEND"
  export PATH="$CONDA_PREFIX/bin:$PATH"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=1
  export HPXRUN_ARGS="${HPXRUN_ARGS:+$HPXRUN_ARGS }--hpx:ini=hpx.stacks.use_guard_pages=0"
}

kangaroo_launch() {
  if [[ "$KANGAROO_HPX_BACKEND" == pixi-hpx ]]; then
    "$REPO_ROOT/scripts/utils/hpxrun_pixi.sh" \
      -r srun -p none -l 1 -t "$CPUS_PER_TASK" "$KANGAROO_PYTHON" "$@"
  else
    "$REPO_ROOT/scripts/utils/hpxrun_spack_hpx.sh" \
      -r srun -p mpi -l "$NNODES" -t "$CPUS_PER_TASK" "$KANGAROO_PYTHON" "$@"
  fi
}
