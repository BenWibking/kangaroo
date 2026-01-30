#!/usr/bin/env python3
"""Demonstrate the Kangaroo uniform slice operator on a synthetic 3D dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis import Plan, Runtime  # noqa: E402
from analysis.runtime import plan_to_dict  # noqa: E402
from analysis.ctx import LoweringContext  # noqa: E402
from analysis.ops import UniformSlice  # noqa: E402
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402


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

    # Two-block synthetic AMR level along x.
    nx = ny = nz = 64
    dx = 0.25
    x0 = (0.0, 0.0, 0.0)
    nx_half = nx // 2

    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(dx, dx, dx), x0=x0, ref_ratio=1),
                        boxes=[
                            BlockBox((0, 0, 0), (nx_half - 1, ny - 1, nz - 1)),
                            BlockBox((nx_half, 0, 0), (nx - 1, ny - 1, nz - 1)),
                        ],
                    )
                ],
            )
        ]
    )

    ds = open_dataset("memory://slice-demo", runmeta=runmeta, step=0, level=0, runtime=rt)
    field = ds.field_id("scalar")

    block_data: list[np.ndarray] = []
    for block_id, box in enumerate(runmeta.steps[0].levels[0].boxes):
        bx = box.hi[0] - box.lo[0] + 1
        by = box.hi[1] - box.lo[1] + 1
        bz = box.hi[2] - box.lo[2] + 1
        x0_block = (
            x0[0] + box.lo[0] * dx,
            x0[1] + box.lo[1] * dx,
            x0[2] + box.lo[2] * dx,
        )
        data = make_synthetic_field(bx, by, bz, dx, x0_block)
        block_data.append(data)
        ds.set_chunk(field=field, block=block_id, data=data.astype(np.float32).tobytes(order="C"))

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
    # Note: Kangaroo's UniformSlice output is laid out with in-plane axes
    # swapped relative to NumPy's (x, y) indexing, so the raw array is (y, x).
    slice_field = plan.stages[-1].templates[0].outputs[0].field
    raw = rt.get_task_chunk(step=0, level=0, field=slice_field, version=0, block=0)
    slice_2d = np.frombuffer(raw, dtype=np.float32, count=nx * ny).reshape((nx, ny))
    full_field = np.concatenate(block_data, axis=0)
    expected_slice = full_field[:, :, k]
    kangaroo_slice = slice_2d.T
    max_abs_diff = np.max(np.abs(kangaroo_slice - expected_slice))
    print(f"Slice comparison: max abs diff = {max_abs_diff:.6e}")
    print(
        "Slice comparison: allclose = "
        f"{np.allclose(kangaroo_slice, expected_slice, rtol=1e-6, atol=1e-6)}"
    )

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
