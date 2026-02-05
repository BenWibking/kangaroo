#!/usr/bin/env bash
set -euo pipefail

# Run a command inside the pixi environment with tcmalloc preloaded.
exec bash -lc 'LD_PRELOAD=$CONDA_PREFIX/lib/libtcmalloc_minimal.so.4 "$@"' -- "$@"
