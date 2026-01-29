#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${HPX_DIR:-}" ]]; then
  echo "$HPX_DIR"
  exit 0
fi

find_from_prefix() {
  local prefix="$1"
  local cand
  for cand in "$prefix/lib/cmake/HPX" "$prefix/lib64/cmake/HPX" "$prefix/share/HPX/cmake" "$prefix/share/HPX"; do
    if [[ -d "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

if command -v hpx-config >/dev/null 2>&1; then
  prefix="$(dirname "$(dirname "$(command -v hpx-config)")")"
  if find_from_prefix "$prefix"; then
    exit 0
  fi
fi

if command -v hpxcxx >/dev/null 2>&1; then
  prefix="$(dirname "$(dirname "$(command -v hpxcxx)")")"
  if find_from_prefix "$prefix"; then
    exit 0
  fi
fi

for prefix in /usr /usr/local /opt/homebrew /opt/local; do
  if find_from_prefix "$prefix"; then
    exit 0
  fi
done

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  if find_from_prefix "$CONDA_PREFIX"; then
    exit 0
  fi
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  if find_from_prefix "$VIRTUAL_ENV"; then
    exit 0
  fi
fi

echo "HPX_DIR not found. Set HPX_DIR or install HPX and ensure hpx-config/hpxcxx is on PATH." >&2
exit 1
