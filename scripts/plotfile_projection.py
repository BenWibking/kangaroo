#!/usr/bin/env python3
"""Project a plotfile FAB along an axis with AMR-aware column integration."""

from __future__ import annotations

import argparse
import numpy as np

import kangaroo as kr  # noqa: E402
from kangaroo.runtime import run_console_main  # noqa: E402



def _parse_bounds(bounds: str) -> tuple[float, float]:
    try:
        b0, b1 = (float(v.strip()) for v in bounds.split(","))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("--axis-bounds must be in min,max format") from exc
    if b0 == b1:
        raise ValueError("--axis-bounds must specify two distinct values")
    return b0, b1



def main() -> int:
    p = argparse.ArgumentParser(description="Run Kangaroo UniformProjection on a plotfile FAB.")
    p.add_argument("plotfile")
    p.add_argument("--var")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--axis", choices=("x", "y", "z"), default="z")
    p.add_argument("--axis-bounds")
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--output")
    p.add_argument("--resolution")
    a, u = p.parse_known_args()

    client = kr.Client.from_parsed_args(a, unknown_args=u)

    def _run() -> int:
        base_level = 0
        ds = client.open_dataset(a.plotfile, level=base_level)
        comp = a.var or next(iter(ds.fields))

        view = ds.geometry.plane(
            axis=a.axis,
            zoom=a.zoom,
            resolution=a.resolution,
        )
        rect = view.rect
        res = view.resolution
        labels = view.labels
        plane = view.plane
        axis_bounds = _parse_bounds(a.axis_bounds) if a.axis_bounds else view.axis_bounds

        arr = ds[comp].project(
            axis=a.axis,
            bounds=axis_bounds,
            rect=rect,
            resolution=res,
            amr_cell_average=True,
        ).compute()

        import matplotlib.pyplot as plt

        nx, ny = res
        pixel_area = abs((rect[2] - rect[0]) / nx) * abs((rect[3] - rect[1]) / ny)
        arr = arr / pixel_area

        msun_g = 1.98847e33
        pc_cm = 3.0856775814913673e18
        cgs_surface_density_to_msun_pc2 = (pc_cm**2) / msun_g
        arr = arr * cgs_surface_density_to_msun_pc2

        fig, axp = plt.subplots(figsize=(6, 5))
        kpc = 1.0e3 * pc_cm
        extent = (rect[0] / kpc, rect[2] / kpc, rect[1] / kpc, rect[3] / kpc)
        im = axp.imshow(
            np.ma.masked_invalid(np.log10(arr)), origin="lower", cmap="viridis", extent=extent
        )
        axp.set_title(f"AMR projection of {comp} ({plane} plane)")
        axp.set_xlabel(f"{labels[0]} [kpc]")
        axp.set_ylabel(f"{labels[1]} [kpc]")
        fig.colorbar(im, ax=axp, label=f"log10({comp} [Msun pc^-2])")
        fig.tight_layout()
        fig.savefig(a.output, dpi=150) if a.output else plt.show()

        return 0

    return int(run_console_main(client.runtime, _run))


if __name__ == "__main__":
    raise SystemExit(main())
