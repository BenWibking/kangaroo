#!/usr/bin/env python3
"""Project a plotfile FAB along an axis with AMR-aware column integration."""

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

from analysis import Runtime  # noqa: E402
from analysis.pipeline import pipeline  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402
from analysis.runtime import log_task_event, plan_to_dict, set_event_log_path  # noqa: E402


def _parse_bounds(bounds: str) -> tuple[float, float]:
    try:
        b0, b1 = (float(v.strip()) for v in bounds.split(","))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("--axis-bounds must be in min,max format") from exc
    if not (b0 < b1 or b1 < b0):
        raise ValueError("--axis-bounds must specify two distinct values")
    return b0, b1


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
    parser = argparse.ArgumentParser(description="Run Kangaroo UniformProjection on a plotfile FAB.")
    parser.add_argument("plotfile", help="Path to a plotfile directory or openPMD dataset.")
    parser.add_argument(
        "--var",
        help="Variable name or component index to project (openPMD: mesh or mesh/component).",
    )
    parser.add_argument("--level", type=int, default=0, help="AMR level index (for input type detection).")
    parser.add_argument(
        "--amr-cell-average",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AMR-aware cell-average projection (default: enabled).",
    )
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z", help="Projection axis.")
    parser.add_argument(
        "--axis-bounds",
        help="Projection bounds along the axis as min,max in physical units.",
    )
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

    if not args.amr_cell_average:
        raise ValueError("UniformProjection requires AMR-aware cell-average semantics")

    try:
        with logger.span("setup/runtime_init"):
            rt = Runtime.from_parsed_args(args, unknown_args=unknown)
    except Exception as exc:
        print("Runtime init failed (is the C++ module built?):", exc)
        return 1

    with logger.span("setup/runmeta_dataset"):
        base_level = 0
        ds = open_dataset(args.plotfile, level=base_level, runtime=rt)
        meta_bundle = ds.metadata_bundle()
        runmeta = meta_bundle.runmeta
        comp_name, field, _ = ds.resolve_field(args.var)
        bytes_per_value = ds.infer_bytes_per_value(rt, field=field, level=args.level)
        if bytes_per_value not in (4, 8):
            print(f"Warning: detected unusual bytes_per_value: {bytes_per_value}")

    view = ds.plane_geometry(
        axis=args.axis,
        level=base_level,
        zoom=args.zoom,
        resolution=args.resolution,
    )
    rect = view["rect"]
    resolution = view["resolution"]
    plane_label = view["plane"]
    axis_labels = view["labels"]
    axis_bounds = _parse_bounds(args.axis_bounds) if args.axis_bounds else view["axis_bounds"]

    with logger.span("setup/plan_build"):
        pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        proj_out = pipe.uniform_projection(
            field=pipe.field(field),
            axis=args.axis,
            axis_bounds=axis_bounds,
            rect=rect,
            resolution=resolution,
            out="projection",
            bytes_per_value=bytes_per_value,
            amr_cell_average=args.amr_cell_average,
        )
        plan = pipe.plan()

        plan_payload = plan_to_dict(plan)
        plan_out = os.environ.get("KANGAROO_DASHBOARD_PLAN")
        if plan_out:
            try:
                with open(plan_out, "w", encoding="utf-8") as handle:
                    json.dump(plan_payload, handle, indent=2, sort_keys=True)
            except OSError as exc:
                print(f"Failed to write plan JSON to {plan_out}: {exc}")

    io_mode = os.environ.get("KANGAROO_IO_MODE", "normal").strip().lower()
    if io_mode == "preload_inputs":
        with logger.span("setup/preload_inputs"):
            rt.preload(runmeta=runmeta, dataset=ds, fields=[field])
    elif io_mode not in {"", "normal"}:
        raise ValueError("KANGAROO_IO_MODE must be one of: normal, preload_inputs")

    try:
        with logger.span("runtime/execute_plan"):
            rt.run(plan, runmeta=runmeta, dataset=ds)
    except Exception as exc:
        print("Runtime executed but raised (kernels may be missing):", exc)

    with logger.span("postprocess/fetch_output"):
        proj_field = proj_out.field
        proj_2d = rt.get_task_chunk_array(
            step=0,
            level=base_level,
            field=proj_field,
            version=0,
            block=0,
            shape=resolution,
            bytes_per_value=bytes_per_value,
            dataset=ds,
        )

    with logger.span("postprocess/plot"):
        fig, ax = plt.subplots(figsize=(6, 5))
        log_proj = np.ma.masked_invalid(np.log10(proj_2d))
        kpc_in_cm = 3.0856775814913673e21
        rect_kpc = tuple(coord / kpc_in_cm for coord in rect)
        extent_kpc = (rect_kpc[0], rect_kpc[2], rect_kpc[1], rect_kpc[3])
        im = ax.imshow(log_proj, origin="lower", cmap="viridis", extent=extent_kpc)
        ax.set_title(f"AMR projection of {comp_name} ({plane_label} plane)")
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
