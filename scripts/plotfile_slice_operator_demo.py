#!/usr/bin/env python3
"""Slice a single plotfile FAB with Kangaroo's UniformSlice and plot the result."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis import Plan, PlotfileReader, Runtime  # noqa: E402
from analysis.ctx import LoweringContext  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402
from analysis.ops import UniformSlice  # noqa: E402
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta  # noqa: E402
from analysis.runtime import plan_to_dict  # noqa: E402


def _resolve_component(hdr: dict[str, object], var: str | None) -> tuple[int, str]:
    names = [str(v) for v in hdr.get("var_names", [])]
    if not names:
        raise ValueError("plotfile header has no variable names")

    if var is None:
        return 0, names[0]

    if var.isdigit():
        idx = int(var)
        if idx < 0 or idx >= len(names):
            raise ValueError(f"component index {idx} out of range (0..{len(names)-1})")
        return idx, names[idx]

    if var in names:
        return names.index(var), var

    raise ValueError(f"variable '{var}' not found; available: {', '.join(names)}")


def _axis_index(axis: str) -> int:
    axes = {"x": 0, "y": 1, "z": 2}
    try:
        return axes[axis]
    except KeyError as exc:
        raise ValueError("axis must be one of x, y, z") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kangaroo UniformSlice on a plotfile FAB.")
    parser.add_argument("plotfile", help="Path to the plotfile directory.")
    parser.add_argument("--var", help="Variable name or component index to slice.")
    parser.add_argument("--level", type=int, default=0, help="AMR level index.")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z", help="Slice axis.")
    parser.add_argument("--coord", type=float, help="Slice coordinate in physical units.")
    parser.add_argument("--output", help="Optional path to save the plot as an image.")
    parser.add_argument(
        "--hpx-config",
        action="append",
        default=None,
        help="HPX config entry (repeatable, e.g. hpx.os_threads=2).",
    )
    parser.add_argument(
        "--hpx-arg",
        action="append",
        default=None,
        help="HPX command-line argument (repeatable, e.g. --hpx:localities=4).",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        unknown = [sys.argv[0], *unknown]

    if not os.path.isdir(args.plotfile):
        print(f"Plotfile path does not exist: {args.plotfile}")
        return 1

    try:
        if args.hpx_config or args.hpx_arg or unknown:
            hpx_args = []
            if args.hpx_arg:
                hpx_args.extend(args.hpx_arg)
            if unknown:
                hpx_args.extend(unknown)
            rt = Runtime(hpx_config=args.hpx_config, hpx_args=hpx_args)
        else:
            rt = Runtime()
    except Exception as exc:
        print("Runtime init failed (is the C++ module built?):", exc)
        return 1

    reader = PlotfileReader(args.plotfile)
    header = reader.header()
    meta = reader.metadata()
    comp_idx, comp_name = _resolve_component(header, args.var)

    prob_lo = tuple(meta.get("prob_lo", (0.0, 0.0, 0.0)))
    if len(prob_lo) != 3:
        raise RuntimeError(f"unexpected prob_lo metadata: {prob_lo}")

    prob_domain = meta.get("prob_domain", [])
    if not prob_domain or args.level >= len(prob_domain):
        raise RuntimeError("plotfile metadata missing prob_domain for requested level")
    domain_lo, domain_hi = prob_domain[args.level]
    nx = int(domain_hi[0]) - int(domain_lo[0]) + 1
    ny = int(domain_hi[1]) - int(domain_lo[1]) + 1
    nz = int(domain_hi[2]) - int(domain_lo[2]) + 1

    cell_size = meta.get("cell_size", [])
    if not cell_size:
        raise RuntimeError("plotfile metadata missing cell_size")
    dx = float(cell_size[args.level][0])

    # AMReX prob_lo corresponds to the physical location of prob_domain.lo.
    # The runtime expects coordinates as x0 + i*dx, so shift by domain_lo.
    x0 = (
        prob_lo[0] - int(domain_lo[0]) * dx,
        prob_lo[1] - int(domain_lo[1]) * dx,
        prob_lo[2] - int(domain_lo[2]) * dx,
    )

    level_boxes = meta.get("level_boxes", [])
    if not level_boxes or args.level >= len(level_boxes):
        raise RuntimeError(
            "plotfile metadata missing level_boxes for requested level. "
            "Rebuild the _plotfile binding to pick up metadata() updates."
        )
    boxes = level_boxes[args.level]
    if len(boxes) != reader.num_fabs(args.level):
        raise RuntimeError("level_boxes count does not match num_fabs for requested level")

    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(dx, dx, dx), x0=x0, ref_ratio=1),
                        boxes=[BlockBox(tuple(lo), tuple(hi)) for lo, hi in boxes],
                    )
                ],
            )
        ]
    )

    ds = open_dataset("memory://plotfile-slice", runmeta=runmeta, step=0, level=0, runtime=rt)
    field = ds.field_id(comp_name)
    dtype = reader.read_fab(args.level, 0, comp_idx, 1).get("dtype", "float32")
    bytes_per_value = 8 if dtype == "float64" else 4
    for block_id, (lo, hi) in enumerate(boxes):
        payload = reader.read_fab(args.level, block_id, comp_idx, 1, return_ndarray=True)
        arr = payload["data"]
        if arr.ndim != 4:
            raise RuntimeError(f"unexpected FAB array shape: {arr.shape}")
        data = arr[0].transpose(2, 1, 0)
        bx = int(hi[0]) - int(lo[0]) + 1
        by = int(hi[1]) - int(lo[1]) + 1
        bz = int(hi[2]) - int(lo[2]) + 1
        if data.shape != (bx, by, bz):
            raise RuntimeError(
                f"FAB {block_id} shape {data.shape} does not match box {lo}..{hi}"
            )
        if dtype == "float64":
            chunk = data.astype(np.float64).tobytes(order="C")
        else:
            chunk = data.astype(np.float32).tobytes(order="C")
        ds.set_chunk(field=field, block=block_id, data=chunk)

    axis_idx = _axis_index(args.axis)
    n_axis = (nx, ny, nz)[axis_idx]
    if args.coord is None:
        mid_idx = (int(domain_lo[axis_idx]) + int(domain_hi[axis_idx])) // 2
        coord = x0[axis_idx] + (mid_idx + 0.5) * dx
    else:
        coord = args.coord

    if args.axis == "z":
        prob_hi = tuple(meta.get("prob_hi", (0.0, 0.0, 0.0)))
        rect = (prob_lo[0], prob_lo[1], prob_hi[0], prob_hi[1])
        resolution = (nx, ny)
        plane_label = "xy"
    elif args.axis == "y":
        prob_hi = tuple(meta.get("prob_hi", (0.0, 0.0, 0.0)))
        rect = (prob_lo[0], prob_lo[2], prob_hi[0], prob_hi[2])
        resolution = (nx, nz)
        plane_label = "xz"
    else:
        prob_hi = tuple(meta.get("prob_hi", (0.0, 0.0, 0.0)))
        rect = (prob_lo[1], prob_lo[2], prob_hi[1], prob_hi[2])
        resolution = (ny, nz)
        plane_label = "yz"

    op = UniformSlice(
        field=field,
        axis=args.axis,
        coord=coord,
        rect=rect,
        resolution=resolution,
        bytes_per_value=bytes_per_value,
    )
    ctx = LoweringContext(runtime=rt._rt, runmeta=runmeta, dataset=ds)
    plan = Plan(stages=op.lower(ctx))

    print("PlanIR:")
    print(plan_to_dict(plan))

    try:
        rt.run(plan, runmeta=runmeta, dataset=ds)
    except Exception as exc:
        print("Runtime executed but raised (kernels may be missing):", exc)

    slice_field = plan.stages[-1].templates[0].outputs[0].field
    raw = rt.get_task_chunk(step=0, level=0, field=slice_field, version=0, block=0)
    np_dtype = np.float64 if bytes_per_value == 8 else np.float32
    slice_2d = np.frombuffer(raw, dtype=np_dtype, count=resolution[0] * resolution[1]).reshape(
        resolution
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(slice_2d.T, origin="lower", cmap="viridis")
    ax.set_title(f"UniformSlice of {comp_name} ({plane_label} plane)")
    ax.set_xlabel("index 0")
    ax.set_ylabel("index 1")
    fig.colorbar(im, ax=ax, label=comp_name)
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=150)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
