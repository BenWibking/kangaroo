#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from typing import List

from analysis.dashboard import main as dashboard_main


def _threads_config_present(argv: List[str]) -> bool:
    for idx, arg in enumerate(argv):
        if "hpx.os_threads" in arg:
            return True
        if arg == "--hpx-arg" and idx + 1 < len(argv):
            if "--hpx:threads" in argv[idx + 1]:
                return True
    return False


def _inject_threads(argv: List[str], threads: int) -> List[str]:
    if "--run" not in argv:
        return argv
    if _threads_config_present(argv):
        return argv
    insert_at = len(argv)
    if "--" in argv:
        insert_at = argv.index("--") + 1
    return [*argv[:insert_at], "--hpx-config", f"hpx.os_threads={threads}", *argv[insert_at:]]


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--threads-per-locality", type=int)
    args, remainder = parser.parse_known_args(argv)
    if args.threads_per_locality is not None:
        if args.threads_per_locality < 1:
            raise SystemExit("--threads-per-locality must be >= 1")
        remainder = _inject_threads(remainder, args.threads_per_locality)
    dashboard_main(remainder)


if __name__ == "__main__":
    main(sys.argv[1:])
