#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIXI_PY="$REPO_ROOT/.pixi/envs/pixi-hpx/bin/python"
HPXRUN="$REPO_ROOT/.pixi/envs/pixi-hpx/bin/hpxrun.py"

if [[ ! -x "$PIXI_PY" ]]; then
  echo "pixi python not found at $PIXI_PY" >&2
  exit 1
fi
if [[ ! -f "$HPXRUN" ]]; then
  echo "hpxrun.py not found at $HPXRUN" >&2
  exit 1
fi

# Use the vendored tools even when called from a different Pixi environment.
export PATH="$REPO_ROOT/.pixi/envs/pixi-hpx/bin:$PATH"
export LD_LIBRARY_PATH="$REPO_ROOT/.pixi/envs/pixi-hpx/lib:${LD_LIBRARY_PATH:-}"

# With launcher options, supply the executable explicitly (as hpxrun.py expects).
if [[ $# -eq 0 || "$1" == -* || -x "$1" ]]; then
  exec "$PIXI_PY" "$HPXRUN" -p none "$@"
fi

exec "$PIXI_PY" "$HPXRUN" -p none "$PIXI_PY" "$@"
