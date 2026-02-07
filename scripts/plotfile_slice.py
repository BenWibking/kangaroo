#!/usr/bin/env python3
"""Slice a plotfile FAB with Kangaroo UniformSlice and plot the result."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis import Runtime  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402
from analysis.pipeline import pipeline  # noqa: E402


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _resolve_dataset_uri(path: str) -> tuple[str, str]:
    if path.startswith(("amrex://", "openpmd://", "file://")):
        return path, ("openpmd" if path.startswith("openpmd://") else "amrex")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "Header")):
        return f"amrex://{path}", "amrex"
    return f"openpmd://{path}", "openpmd"


def main() -> int:
    p = argparse.ArgumentParser(description="Run Kangaroo UniformSlice on a plotfile FAB.")
    p.add_argument("plotfile")
    p.add_argument("--var")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--amr-cell-average", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--axis", choices=("x", "y", "z"), default="z")
    p.add_argument("--coord", type=float)
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--output")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--resolution")
    p.add_argument("--hpx-config", action="append", default=None)
    p.add_argument("--hpx-arg", action="append", default=None)
    a, u = p.parse_known_args()
    hpx = (a.hpx_arg or []) + ([sys.argv[0], *u] if u else [])
    
    uri, kind = _resolve_dataset_uri(a.plotfile)
    rt = Runtime(hpx_config=a.hpx_config, hpx_args=hpx) if (a.hpx_config or hpx) else Runtime()

    base = 0 if a.amr_cell_average else a.level
    ds = open_dataset(uri, level=base, runtime=rt)
    meta = ds.metadata
    runmeta = ds.get_runmeta()

    from analysis import _core  # noqa: E402
    _core.set_global_dataset(ds._h)
    comp = a.var or meta["var_names"][0]
    if kind == "openpmd" and comp not in meta["var_names"]:
        raise RuntimeError(f"openPMD field '{comp}' not found")
    field = ds.field_id(comp)

    b0_lo, b0_hi = meta["level_boxes"][a.level][0]
    b0_elems = (b0_hi[0] - b0_lo[0] + 1) * (b0_hi[1] - b0_lo[1] + 1) * (b0_hi[2] - b0_lo[2] + 1)
    b0_chunk = rt.get_task_chunk(step=0, level=a.level, field=field, version=0, block=0)
    bpv = len(b0_chunk) // b0_elems

    prob_lo, prob_hi = meta["prob_lo"], meta["prob_hi"]
    domain_lo, domain_hi = meta["prob_domain"][base]
    nx, ny, nz = (int(domain_hi[i]) - int(domain_lo[i]) + 1 for i in range(3))
    cell_size = meta["cell_size"][base]
    ax = _axis_index(a.axis)
    coord = a.coord if a.coord is not None else prob_lo[ax] + (((int(domain_lo[ax]) + int(domain_hi[ax])) // 2) + 0.5) * cell_size[ax]

    def bounds(i: int, n: int, dlo: int) -> tuple[float, float]:
        dx = float(cell_size[i]) if np.isfinite(float(cell_size[i])) and float(cell_size[i]) != 0.0 else (float(prob_hi[i]) - float(prob_lo[i])) / max(1, n)
        lo = float(prob_lo[i]) + float(dlo) * dx
        hi = lo + dx * float(n)
        return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi != lo else (lo, lo + (float(n) if n else 1.0))

    x_lo, x_hi = bounds(0, nx, int(domain_lo[0]))
    y_lo, y_hi = bounds(1, ny, int(domain_lo[1]))
    z_lo, z_hi = bounds(2, nz, int(domain_lo[2]))
    rect, res, labels, plane = ((x_lo, y_lo, x_hi, y_hi), (nx, ny), ("x", "y"), "xy") if a.axis == "z" else (((x_lo, z_lo, x_hi, z_hi), (nx, nz), ("x", "z"), "xz") if a.axis == "y" else ((y_lo, z_lo, y_hi, z_hi), (ny, nz), ("y", "z"), "yz"))
    if a.resolution: res = tuple(int(v.strip()) for v in a.resolution.split(","))
    if a.zoom <= 0: raise ValueError("--zoom must be a positive number")
    if a.zoom != 1.0:
        xm, ym = 0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3])
        hw, hh = 0.5 * (rect[2] - rect[0]) / a.zoom, 0.5 * (rect[3] - rect[1]) / a.zoom
        rect = (xm - hw, ym - hh, xm + hw, ym + hh)

    pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    out = pipe.uniform_slice(field=pipe.field(field), axis=a.axis, coord=coord, rect=rect, resolution=res, out="slice", bytes_per_value=bpv, amr_cell_average=a.amr_cell_average)
    rt.run(pipe.plan(), runmeta=runmeta, dataset=ds)
    raw = rt.get_task_chunk(step=0, level=base, field=out.field, version=0, block=0)
    arr = np.frombuffer(raw, dtype=(np.float64 if bpv == 8 else np.float32), count=res[0] * res[1]).reshape(res)

    if not a.no_plot:
        import matplotlib.pyplot as plt
        fig, axp = plt.subplots(figsize=(6, 5))
        kpc = 3.0856775814913673e21
        im = axp.imshow(np.ma.masked_invalid(np.log10(arr)), origin="lower", cmap="viridis", extent=(rect[0] / kpc, rect[2] / kpc, rect[1] / kpc, rect[3] / kpc))
        axp.set_title(f"{'AMR cell-average' if a.amr_cell_average else 'UniformSlice'} of {comp} ({plane} plane)")
        axp.set_xlabel(f"{labels[0]} [kpc]")
        axp.set_ylabel(f"{labels[1]} [kpc]")
        fig.colorbar(im, ax=axp, label=f"log10({comp})")
        fig.tight_layout()
        fig.savefig(a.output, dpi=150) if a.output else plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
