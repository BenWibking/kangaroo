from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _prioritize_repo_root() -> None:
    repo_real = os.path.realpath(REPO_ROOT)
    kept: list[str] = []
    for entry in sys.path:
        entry_abs = os.path.abspath(entry or os.getcwd())
        entry_real = os.path.realpath(entry_abs)
        if entry_real == repo_real:
            continue
        kept.append(entry)
    sys.path[:] = [REPO_ROOT, *kept]


_prioritize_repo_root()
