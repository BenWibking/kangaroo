#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 BATCH_SCRIPT PLOTFILE" >&2
  exit 2
fi

BATCH_SCRIPT=$1
PLOTFILE=$2

if [[ ! -f "$BATCH_SCRIPT" ]]; then
  echo "Batch script does not exist: $BATCH_SCRIPT" >&2
  exit 2
fi

if [[ ! -d "$PLOTFILE" ]]; then
  echo "Plotfile does not exist or is not a directory: $PLOTFILE" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BATCH_SCRIPT="$(cd "$(dirname "$BATCH_SCRIPT")" && pwd)/$(basename "$BATCH_SCRIPT")"
PLOTFILE="$(cd "$PLOTFILE" && pwd -P)"
export KANGAROO_REPO_ROOT="$REPO_ROOT"
export KANGAROO_HPX_BACKEND="${KANGAROO_HPX_BACKEND:-pixi-hpx}"
exec sbatch --export=ALL --chdir="$REPO_ROOT" "$BATCH_SCRIPT" "$PLOTFILE"
