#!/usr/bin/env python3
"""Project stellar particle density with native-grid CIC deposition."""

from __future__ import annotations

import argparse
import numpy as np

from analysis import Runtime  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402
from analysis.pipeline import pipeline  # noqa: E402


def _parse_bounds(bounds: str) -> tuple[float, float]:
    try:
        b0, b1 = (float(v.strip()) for v in bounds.split(","))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("--axis-bounds must be in min,max format") from exc
    if b0 == b1:
        raise ValueError("--axis-bounds must specify two distinct values")
    return b0, b1


def main() -> int:
    p = argparse.ArgumentParser(
        description="Deposit stellar particles with CIC on native 2D AMR grids, then project."
    )
    p.add_argument("plotfile")
    p.add_argument("--axis", choices=("x", "y", "z"), default="z")
    p.add_argument("--axis-bounds")
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--output")
    p.add_argument("--resolution")
    p.add_argument(
        "--particle-type",
        "--particles",
        dest="particle_type",
        choices=("CIC_particles", "StochasticStellarPop_particles"),
        default="StochasticStellarPop_particles",
        help="Particle species to deposit.",
    )
    a = p.parse_args()

    rt = Runtime.from_parsed_args(a, unknown_args=[])

    base_level = 0
    ds = open_dataset(a.plotfile, level=base_level, runtime=rt)
    runmeta = ds.metadata_bundle().runmeta

    view = ds.plane_geometry(
        axis=a.axis,
        level=base_level,
        zoom=a.zoom,
        resolution=a.resolution,
    )
    rect = view["rect"]
    res = view["resolution"]
    labels = view["labels"]
    plane = view["plane"]
    axis_bounds = _parse_bounds(a.axis_bounds) if a.axis_bounds else view["axis_bounds"]

    particle_types = ds.list_particle_types()
    if a.particle_type not in particle_types:
        available = ", ".join(particle_types) if particle_types else "<none>"
        raise RuntimeError(
            f"particle type '{a.particle_type}' not found in plotfile; available: {available}"
        )

    pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    out = pipe.particle_cic_projection(
        particle_type=a.particle_type,
        axis=a.axis,
        axis_bounds=axis_bounds,
        rect=rect,
        resolution=res,
        out="stellar_projection",
    )
    rt.run(pipe.plan(), runmeta=runmeta, dataset=ds)

    arr = rt.get_task_chunk_array(
        step=0,
        level=base_level,
        field=out.field,
        version=0,
        block=0,
        shape=res,
        bytes_per_value=8,
        dataset=ds,
    )

    nx, ny = res
    pixel_area = abs((rect[2] - rect[0]) / nx) * abs((rect[3] - rect[1]) / ny)
    arr = arr / pixel_area

    msun_g = 1.98847e33
    pc_cm = 3.0856775814913673e18
    cgs_surface_density_to_msun_pc2 = (pc_cm**2) / msun_g
    arr = arr * cgs_surface_density_to_msun_pc2

    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    fig, axp = plt.subplots(figsize=(6, 5))
    kpc = 1.0e3 * pc_cm
    extent = (rect[0] / kpc, rect[2] / kpc, rect[1] / kpc, rect[3] / kpc)
    masked = np.ma.masked_invalid(arr)
    vmax = np.nanmax(arr)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    norm = SymLogNorm(linthresh=1.0e-2, vmin=0.0, vmax=vmax)
    im = axp.imshow(masked, origin="lower", cmap="viridis", extent=extent, norm=norm)
    axp.set_title(
        f"CIC stellar projection [{a.particle_type}] ({plane} plane)"
    )
    axp.set_xlabel(f"{labels[0]} [kpc]")
    axp.set_ylabel(f"{labels[1]} [kpc]")
    fig.colorbar(im, ax=axp, label=f"{a.particle_type} [Msun pc^-2] (symlog)")
    fig.tight_layout()
    fig.savefig(a.output, dpi=150) if a.output else plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
