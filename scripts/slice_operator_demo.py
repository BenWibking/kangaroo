#!/usr/bin/env python3
"""Demonstrate the Kangaroo uniform slice operator on a synthetic 3D dataset."""

from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from analysis import Plan, Runtime
from analysis.runtime import plan_to_dict
from analysis.ctx import LoweringContext
from analysis.ops import UniformSlice
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta
from analysis.dataset import open_dataset


def make_synthetic_field(nx: int, ny: int, nz: int, dx: float, x0: tuple[float, float, float]) -> np.ndarray:
    """Create a simple synthetic scalar field on cell centers."""
    x = x0[0] + (np.arange(nx) + 0.5) * dx
    y = x0[1] + (np.arange(ny) + 0.5) * dx
    z = x0[2] + (np.arange(nz) + 0.5) * dx
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.sin(xx) + np.cos(yy) + 0.25 * zz


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kangaroo UniformSlice demo.")
    parser.add_argument("--output", help="Optional path to save the plot as an image.")
    args = parser.parse_args()

    try:
        rt = Runtime()
    except Exception as exc:
        print("Runtime init failed (is the C++ module built?):", exc)
        return 1

    # Single-block synthetic AMR level.
    nx = ny = nz = 64
    dx = 0.25
    x0 = (0.0, 0.0, 0.0)

    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(dx, dx, dx), x0=x0, ref_ratio=1),
                        boxes=[BlockBox((0, 0, 0), (nx - 1, ny - 1, nz - 1))],
                    )
                ],
            )
        ]
    )

    ds = open_dataset("memory://slice-demo", runmeta=runmeta, step=0, level=0, runtime=rt)
    field = ds.field_id("scalar")

    data = make_synthetic_field(nx, ny, nz, dx, x0)
    ds.set_chunk(field=field, block=0, data=data.astype(np.float32).tobytes(order="C"))

    # Define a z-slice through the middle of the domain.
    axis = "z"
    k = nz // 2
    coord = x0[2] + (k + 0.5) * dx
    rect = (x0[0], x0[1], x0[0] + nx * dx, x0[1] + ny * dx)
    resolution = (nx, ny)

    op = UniformSlice(field=field, axis=axis, coord=coord, rect=rect, resolution=resolution)
    ctx = LoweringContext(runtime=rt._rt, runmeta=runmeta, dataset=ds)
    plan = Plan(stages=op.lower(ctx))

    plan_ir = plan_to_dict(plan)
    print("PlanIR:")
    print(json.dumps(plan_ir, indent=2))

    try:
        rt.run(plan, runmeta=runmeta, dataset=ds)
    except Exception as exc:
        print("Runtime executed but raised (kernels may be missing):", exc)

    # Fetch the task-graph output bytes and decode to a 2D float32 array.
    slice_field = plan.stages[0].templates[0].outputs[0].field
    raw = rt.get_task_chunk(step=0, level=0, field=slice_field, version=0, block=0)
    slice_2d = np.frombuffer(raw, dtype=np.float32, count=nx * ny).reshape((nx, ny))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(slice_2d.T, origin="lower", cmap="viridis")
    ax.set_title("Kangaroo UniformSlice example (z mid-plane)")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, label="value")
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=150)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
