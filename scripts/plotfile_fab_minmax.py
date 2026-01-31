#!/usr/bin/env python3
"""Print per-component min/max for each FAB in a plotfile."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis import PlotfileReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Print per-component min/max for each FAB.")
    parser.add_argument("plotfile", help="Path to the plotfile directory.")
    parser.add_argument("--level", type=int, default=None, help="Level to inspect (default: all).")
    parser.add_argument("--fab", type=int, default=None, help="FAB index to inspect (default: all).")
    args = parser.parse_args()

    if not os.path.isdir(args.plotfile):
        print(f"Plotfile path does not exist: {args.plotfile}")
        return 1

    reader = PlotfileReader(args.plotfile)
    header = reader.header()
    ncomp = int(header["ncomp"])
    var_names = [str(v) for v in header.get("var_names", [])]

    levels = [args.level] if args.level is not None else list(range(reader.num_levels()))
    for level in levels:
        nfabs = reader.num_fabs(level)
        fab_indices = [args.fab] if args.fab is not None else list(range(nfabs))
        print(f"Level {level}: {nfabs} FABs")
        for fab in fab_indices:
            payload = reader.read_fab(level, fab, 0, ncomp, return_ndarray=True)
            data = payload["data"]
            if data.ndim != 4:
                raise RuntimeError(f"unexpected FAB array shape: {data.shape}")
            # data shape: (ncomp, nz, ny, nx)
            mins = np.nanmin(data, axis=(1, 2, 3))
            maxs = np.nanmax(data, axis=(1, 2, 3))
            print(f"  FAB {fab}:")
            for c in range(ncomp):
                name = var_names[c] if c < len(var_names) else f"comp{c}"
                print(f"    {c:02d} {name}: min={mins[c]:.6e} max={maxs[c]:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
