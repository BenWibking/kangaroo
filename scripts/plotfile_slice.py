#!/usr/bin/env python3
"""Slice a plotfile FAB with Kangaroo UniformSlice and plot the result."""

from __future__ import annotations

import argparse
import numpy as np

import kangaroo as kr  # noqa: E402
from kangaroo.runtime import run_console_main  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Run Kangaroo UniformSlice on a plotfile FAB.")
    p.add_argument("plotfile_path")
    p.add_argument("--var")
    p.add_argument("--axis", choices=("x", "y", "z"), default="z")
    p.add_argument("--coord", type=float)
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--output")
    p.add_argument("--resolution")
    p.add_argument("--no-plot", action="store_true")
    a, u = p.parse_known_args()
    client = kr.Client.from_parsed_args(a, unknown_args=u)

    def _run() -> int:
        ds = client.open_dataset(a.plotfile_path)
        comp = a.var or next(iter(ds.fields))
        view = ds.geometry.plane(
            axis=a.axis,
            coord=a.coord,
            zoom=a.zoom,
            resolution=a.resolution,
        )
        rect = view.rect
        res = view.resolution
        labels = view.labels
        plane = view.plane
        arr = ds[comp].slice(
            axis=a.axis,
            coord=view.coord,
            rect=rect,
            resolution=res,
        )
        arr = arr.compute()

        if not a.no_plot:
            import matplotlib.pyplot as plt

            fig, axp = plt.subplots(figsize=(6, 5))
            kpc = 3.0856775814913673e21
            im = axp.imshow(
                np.ma.masked_invalid(np.log10(arr)),
                origin="lower",
                cmap="viridis",
                extent=(rect[0] / kpc, rect[2] / kpc, rect[1] / kpc, rect[3] / kpc),
            )
            axp.set_title(f"AMR cell-average of {comp} ({plane} plane)")
            axp.set_xlabel(f"{labels[0]} [kpc]")
            axp.set_ylabel(f"{labels[1]} [kpc]")
            fig.colorbar(im, ax=axp, label=f"log10({comp})")
            fig.tight_layout()
            fig.savefig(a.output, dpi=150) if a.output else plt.show()
        return 0

    return int(run_console_main(client.runtime, _run))


if __name__ == "__main__":
    raise SystemExit(main())
