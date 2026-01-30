#!/usr/bin/env python3
"""Print HPX build/configuration information from the bound runtime."""

from __future__ import annotations

from analysis import hpx_configuration_string


def main() -> int:
    print(hpx_configuration_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
