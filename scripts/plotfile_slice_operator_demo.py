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
    parser.add_argument(
        "--amr-cell-average",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AMR-aware cell-average slicing (default: enabled).",
    )
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z", help="Slice axis.")
    parser.add_argument("--coord", type=float, help="Slice coordinate in physical units.")
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom factor (yt-style): >1 zooms in, <1 zooms out (default: 1).",
    )
    parser.add_argument("--output", help="Optional path to save the plot as an image.")
    parser.add_argument(
        "--resolution",
        help="Override output resolution as Nx,Ny (e.g. 512,512).",
    )
    parser.add_argument(
        "--dump-area",
        help="Optional path to save the per-pixel covered area array as a .npy file.",
    )
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
        base_level = 0 if args.amr_cell_average else args.level
        ds = open_dataset(f"amrex://{args.plotfile}", level=base_level, runtime=rt)
        
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
    domain_lo, domain_hi = meta["prob_domain"][base_level]
    nx = int(domain_hi[0]) - int(domain_lo[0]) + 1
    ny = int(domain_hi[1]) - int(domain_lo[1]) + 1
    nz = int(domain_hi[2]) - int(domain_lo[2]) + 1
    cell_size = meta["cell_size"][base_level]
    dx = cell_size[0]

    axis_idx = _axis_index(args.axis)
    if args.coord is None:
        mid_idx = (int(domain_lo[axis_idx]) + int(domain_hi[axis_idx])) // 2
        coord = prob_lo[axis_idx] + (mid_idx + 0.5) * cell_size[axis_idx]
    else:
        coord = args.coord

    def _axis_bounds(axis: int, n_cells: int, domain_axis_lo: int) -> tuple[float, float]:
        dx_axis = float(cell_size[axis])
        if not np.isfinite(dx_axis) or dx_axis == 0.0:
            span = float(prob_hi[axis]) - float(prob_lo[axis])
            if n_cells > 0 and np.isfinite(span) and span != 0.0:
                dx_axis = span / float(n_cells)
            else:
                dx_axis = 1.0
        lo = float(prob_lo[axis]) + float(domain_axis_lo) * dx_axis
        hi = lo + dx_axis * float(n_cells)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            hi = lo + float(n_cells) if n_cells else lo + 1.0
        return lo, hi

    x_lo, x_hi = _axis_bounds(0, nx, int(domain_lo[0]))
    y_lo, y_hi = _axis_bounds(1, ny, int(domain_lo[1]))
    z_lo, z_hi = _axis_bounds(2, nz, int(domain_lo[2]))

    if args.axis == "z":
        rect = (x_lo, y_lo, x_hi, y_hi)
        resolution = (nx, ny)
        plane_label = "xy"
        axis_labels = ("x", "y")
    elif args.axis == "y":
        rect = (x_lo, z_lo, x_hi, z_hi)
        resolution = (nx, nz)
        plane_label = "xz"
        axis_labels = ("x", "z")
    else:
        rect = (y_lo, z_lo, y_hi, z_hi)
        resolution = (ny, nz)
        plane_label = "yz"
        axis_labels = ("y", "z")
    if args.resolution:
        try:
            nx_in, ny_in = (int(v.strip()) for v in args.resolution.split(","))
            if nx_in <= 0 or ny_in <= 0:
                raise ValueError
            resolution = (nx_in, ny_in)
        except ValueError as exc:
            raise ValueError("--resolution must be in Nx,Ny format with positive integers") from exc
    if args.zoom <= 0:
        raise ValueError("--zoom must be a positive number")
    if args.zoom != 1.0:
        x_mid = 0.5 * (rect[0] + rect[2])
        y_mid = 0.5 * (rect[1] + rect[3])
        half_width = 0.5 * (rect[2] - rect[0]) / args.zoom
        half_height = 0.5 * (rect[3] - rect[1]) / args.zoom
        rect = (x_mid - half_width, y_mid - half_height, x_mid + half_width, y_mid + half_height)

    with logger.span("setup/plan_build"):
        ctx = LoweringContext(runtime=rt._rt, runmeta=runmeta, dataset=ds)
        op = UniformSlice(
            field=field,
            axis=args.axis,
            coord=coord,
            rect=rect,
            resolution=resolution,
            bytes_per_value=bytes_per_value,
            amr_cell_average=args.amr_cell_average,
        )
        stages = op.lower(ctx)
        plan = Plan(stages=stages)

        plan_payload = plan_to_dict(plan)
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
        raw = rt.get_task_chunk(step=0, level=base_level, field=slice_field, version=0, block=0)
        np_dtype = np.float64 if bytes_per_value == 8 else np.float32
        slice_2d = np.frombuffer(raw, dtype=np_dtype, count=resolution[0] * resolution[1]).reshape(
            resolution
        )
        if args.dump_area:
            area_field = plan.stages[-1].templates[0].inputs[1].field
            raw_area = rt.get_task_chunk(step=0, level=base_level, field=area_field, version=0, block=0)
            area_2d = np.frombuffer(raw_area, dtype=np.float64, count=resolution[0] * resolution[1]).reshape(
                resolution
            )
            np.save(args.dump_area, area_2d)
            area_by_level = {}
            for stage in plan.stages:
                for tmpl in stage.templates:
                    if tmpl.kernel == "uniform_slice_cellavg_accumulate":
                        lvl = tmpl.domain.level
                        if len(tmpl.outputs) >= 2:
                            area_by_level[lvl] = tmpl.outputs[1].field
                    elif tmpl.kernel == "uniform_slice_reduce" and "area_reduce" in tmpl.name:
                        lvl = tmpl.domain.level
                        if tmpl.outputs:
                            area_by_level[lvl] = tmpl.outputs[0].field
            if area_by_level:
                root, ext = os.path.splitext(args.dump_area)
                for lvl, fid in sorted(area_by_level.items()):
                    raw_lvl = rt.get_task_chunk(step=0, level=lvl, field=fid, version=0, block=0)
                    area_lvl = np.frombuffer(raw_lvl, dtype=np.float64, count=resolution[0] * resolution[1]).reshape(
                        resolution
                    )
                    np.save(f"{root}_l{lvl}{ext}", area_lvl)

    with logger.span("postprocess/plot"):
        fig, ax = plt.subplots(figsize=(6, 5))
        log_slice = np.ma.masked_invalid(np.log10(slice_2d))
        kpc_in_cm = 3.0856775814913673e21
        rect_kpc = tuple(coord / kpc_in_cm for coord in rect)
        extent_kpc = (rect_kpc[0], rect_kpc[2], rect_kpc[1], rect_kpc[3])
        im = ax.imshow(log_slice, origin="lower", cmap="viridis", extent=extent_kpc)
        avg_label = "AMR cell-average" if args.amr_cell_average else "UniformSlice"
        ax.set_title(f"{avg_label} of {comp_name} ({plane_label} plane)")
        ax.set_xlabel(f"{axis_labels[0]} [kpc]")
        ax.set_ylabel(f"{axis_labels[1]} [kpc]")
        fig.colorbar(im, ax=ax, label=f"log10({comp_name})")
        fig.tight_layout()
        if args.output:
            fig.savefig(args.output, dpi=150)
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
