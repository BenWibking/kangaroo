#!/usr/bin/env python3
"""Print HPX build/configuration information from the bound runtime."""

from __future__ import annotations

from analysis import Runtime, hpx_configuration_string, run_console_main


def main() -> int:
    rt = Runtime()

    def _run() -> int:
        print(hpx_configuration_string())
        return 0

    return int(run_console_main(rt, _run))


if __name__ == "__main__":
    raise SystemExit(main())
