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

exec sbatch "$BATCH_SCRIPT" "$PLOTFILE"
