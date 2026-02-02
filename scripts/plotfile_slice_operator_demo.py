#!/usr/bin/env python3
"""Slice a single plotfile FAB with Kangaroo's UniformSlice and plot the result."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis import Plan, Runtime  # noqa: E402
from analysis.ctx import LoweringContext  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402
from analysis.ops import UniformSlice  # noqa: E402
from analysis.runtime import log_task_event, plan_to_dict, set_event_log_path  # noqa: E402


def _axis_index(axis: str) -> int:
    axes = {"x": 0, "y": 1, "z": 2}
    try:
        return axes[axis]
    except KeyError as exc:
        raise ValueError("axis must be one of x, y, z") from exc


class EventLogger:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._counter = 0

    @contextmanager
    def span(self, name: str):
        if not self._enabled:
            yield
            return
        self._counter += 1
        span_id = f"py:{self._counter}:{name}"
        start = time.time()
        log_task_event(name, "start", start=start, end=start, event_id=span_id, worker_label="python")
        try:
            yield
        finally:
            end = time.time()
            log_task_event(name, "end", start=start, end=end, event_id=span_id, worker_label="python")


def main() -> int:
    event_log_path = os.environ.get("KANGAROO_EVENT_LOG")
    if event_log_path:
        set_event_log_path(event_log_path)
    logger = EventLogger(event_log_path is not None)
    with logger.span("setup/entrypoint"):
        pass
    with logger.span("setup/imports"):
        global np, plt
        import numpy as np
        import matplotlib.pyplot as plt
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
        with logger.span("setup/runtime_init"):
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

    with logger.span("setup/runmeta_dataset"):
        ds = open_dataset(f"amrex://{args.plotfile}", level=args.level, runtime=rt)
        
        # Set global dataset to enable early chunk fetching for type detection
        from analysis import _core
        _core.set_global_dataset(ds._h)
        
        runmeta = ds.get_runmeta()
        meta = ds.metadata

        if args.var:
            comp_name = args.var
        else:
            comp_name = meta["var_names"][0]
        field = ds.field_id(comp_name)

        # Detect bytes_per_value by reading the first block
        level_boxes = meta["level_boxes"][args.level]
        if not level_boxes:
            raise RuntimeError(f"No boxes found for level {args.level}")
        
        b0_lo, b0_hi = level_boxes[0]
        # box dimensions are inclusive
        b0_nx = b0_hi[0] - b0_lo[0] + 1
        b0_ny = b0_hi[1] - b0_lo[1] + 1
        b0_nz = b0_hi[2] - b0_lo[2] + 1
        b0_elems = b0_nx * b0_ny * b0_nz
        
        b0_chunk = rt.get_task_chunk(step=0, level=args.level, field=field, version=0, block=0)
        bytes_per_value = len(b0_chunk) // b0_elems
        if bytes_per_value not in (4, 8):
             print(f"Warning: detected unusual bytes_per_value: {bytes_per_value}")

    prob_lo = meta["prob_lo"]
    prob_hi = meta["prob_hi"]
    domain_lo, domain_hi = meta["prob_domain"][args.level]
    nx = int(domain_hi[0]) - int(domain_lo[0]) + 1
    ny = int(domain_hi[1]) - int(domain_lo[1]) + 1
    nz = int(domain_hi[2]) - int(domain_lo[2]) + 1
    dx = meta["cell_size"][args.level][0]

    axis_idx = _axis_index(args.axis)
    if args.coord is None:
        mid_idx = (int(domain_lo[axis_idx]) + int(domain_hi[axis_idx])) // 2
        coord = prob_lo[axis_idx] + (mid_idx + 0.5) * dx
    else:
        coord = args.coord

    if args.axis == "z":
        rect = (prob_lo[0], prob_lo[1], prob_hi[0], prob_hi[1])
        resolution = (nx, ny)
        plane_label = "xy"
    elif args.axis == "y":
        rect = (prob_lo[0], prob_lo[2], prob_hi[0], prob_hi[2])
        resolution = (nx, nz)
        plane_label = "xz"
    else:
        rect = (prob_lo[1], prob_lo[2], prob_hi[1], prob_hi[2])
        resolution = (ny, nz)
        plane_label = "yz"

    with logger.span("setup/plan_build"):
        ctx = LoweringContext(runtime=rt._rt, runmeta=runmeta, dataset=ds)
        op = UniformSlice(
            field=field,
            axis=args.axis,
            coord=coord,
            rect=rect,
            resolution=resolution,
            bytes_per_value=bytes_per_value,
        )
        stages = op.lower(ctx)
        plan = Plan(stages=stages)

        plan_payload = plan_to_dict(plan)
        print("PlanIR:")
        print(json.dumps(plan_payload, indent=2, sort_keys=True))
        plan_out = os.environ.get("KANGAROO_DASHBOARD_PLAN")
        if plan_out:
            try:
                with open(plan_out, "w", encoding="utf-8") as handle:
                    json.dump(plan_payload, handle, indent=2, sort_keys=True)
            except OSError as exc:
                print(f"Failed to write plan JSON to {plan_out}: {exc}")

    try:
        with logger.span("runtime/execute_plan"):
            rt.run(plan, runmeta=runmeta, dataset=ds)
    except Exception as exc:
        print("Runtime executed but raised (kernels may be missing):", exc)

    with logger.span("postprocess/fetch_output"):
        slice_field = plan.stages[-1].templates[0].outputs[0].field
        raw = rt.get_task_chunk(step=0, level=args.level, field=slice_field, version=0, block=0)
        np_dtype = np.float64 if bytes_per_value == 8 else np.float32
        slice_2d = np.frombuffer(raw, dtype=np_dtype, count=resolution[0] * resolution[1]).reshape(
            resolution
        )

    with logger.span("postprocess/plot"):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.log10(slice_2d.T), origin="lower", cmap="viridis")
        ax.set_title(f"UniformSlice of {comp_name} ({plane_label} plane)")
        ax.set_xlabel("index 0")
        ax.set_ylabel("index 1")
        fig.colorbar(im, ax=ax, label=f"log10({comp_name})")
        fig.tight_layout()
        if args.output:
            fig.savefig(args.output, dpi=150)
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())