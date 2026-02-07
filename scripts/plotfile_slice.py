#!/usr/bin/env python3
"""Slice a plotfile FAB with Kangaroo UniformSlice and plot the result."""

from __future__ import annotations

import argparse
import numpy as np

from analysis import Runtime  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402
from analysis.pipeline import pipeline  # noqa: E402


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
    rt = Runtime.from_parsed_args(a, unknown_args=u)

    ds = open_dataset(a.plotfile_path, runtime=rt)
    metadata = ds.metadata_bundle()
    runmeta = metadata.runmeta
    comp, field, _ = ds.resolve_field(a.var)
    view = ds.plane_geometry(
        axis=a.axis,
        level=0,
        coord=a.coord,
        zoom=a.zoom,
        resolution=a.resolution,
    )
    rect = view["rect"]
    res = view["resolution"]
    labels = view["labels"]
    plane = view["plane"]
    coord = view["coord"]

    pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    out = pipe.uniform_slice(field=pipe.field(field), axis=a.axis, coord=coord, rect=rect, resolution=res, out="slice")
    rt.run(pipe.plan(), runmeta=runmeta, dataset=ds)

    arr = rt.get_task_chunk_array(
        step=0,
        level=0,
        field=out.field,
        version=0,
        block=0,
        shape=res,
        dataset=ds,
    )

    if not a.no_plot:
        import matplotlib.pyplot as plt
        fig, axp = plt.subplots(figsize=(6, 5))
        kpc = 3.0856775814913673e21
        im = axp.imshow(np.ma.masked_invalid(np.log10(arr)), origin="lower", cmap="viridis", extent=(rect[0] / kpc, rect[2] / kpc, rect[1] / kpc, rect[3] / kpc))
        axp.set_title(f"AMR cell-average of {comp} ({plane} plane)")
        axp.set_xlabel(f"{labels[0]} [kpc]")
        axp.set_ylabel(f"{labels[1]} [kpc]")
        fig.colorbar(im, ax=axp, label=f"log10({comp})")
        fig.tight_layout()
        fig.savefig(a.output, dpi=150) if a.output else plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
