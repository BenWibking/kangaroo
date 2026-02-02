#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXI_PY="$REPO_ROOT/.pixi/envs/default/bin/python"
HPXRUN="$REPO_ROOT/.pixi/envs/default/bin/hpxrun.py"

if [[ ! -x "$PIXI_PY" ]]; then
  echo "pixi python not found at $PIXI_PY" >&2
  exit 1
fi
if [[ ! -f "$HPXRUN" ]]; then
  echo "hpxrun.py not found at $HPXRUN" >&2
  exit 1
fi

if [[ $# -gt 0 && -x "$1" ]]; then
  exec "$PIXI_PY" "$HPXRUN" "$@"
fi

exec "$PIXI_PY" "$HPXRUN" "$PIXI_PY" "$@"
