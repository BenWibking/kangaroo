#!/usr/bin/env python3
"""Slice a plotfile with yt and plot the result for benchmarking comparisons."""

from __future__ import annotations

import argparse
import os
import time


def _axis_index(axis: str) -> int:
    axes = {"x": 0, "y": 1, "z": 2}
    try:
        return axes[axis]
    except KeyError as exc:
        raise ValueError("axis must be one of x, y, z") from exc


def _resolve_amrex_path(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def _find_yt_field(ds, var: str | None):
    if var is None:
        return ds.field_list[0]

    # Accept either fully-qualified yt fields like "boxlib,density" or plain names.
    if "," in var:
        left, right = (v.strip() for v in var.split(",", 1))
        candidate = (left, right)
        if candidate in ds.field_list or candidate in ds.derived_field_list:
            return candidate

    for f in ds.field_list:
        if f[1] == var:
            return f
    for f in ds.derived_field_list:
        if f[1] == var:
            return f

    raise RuntimeError(f"yt field '{var}' not found")


class Timer:
    def __init__(self) -> None:
        self._t0: dict[str, float] = {}
        self.dt: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._t0[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        t0 = self._t0.pop(name, None)
        if t0 is None:
            return
        self.dt[name] = time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run yt slice on a plotfile.")
    parser.add_argument("plotfile", help="Path to a plotfile directory.")
    parser.add_argument("--var", help="Field name to slice (e.g. density or boxlib,density).")
    parser.add_argument("--level", type=int, default=0, help="AMR level index (used when --no-amr-cell-average).")
    parser.add_argument(
        "--amr-cell-average",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep CLI parity with plotfile_slice.py (default: enabled).",
    )
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z", help="Slice axis.")
    parser.add_argument("--coord", type=float, help="Slice coordinate in physical (code_length) units.")
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom factor: >1 zooms in, <1 zooms out (default: 1).",
    )
    parser.add_argument("--output", help="Optional path to save the plot as an image.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting and image output (useful for compute-only benchmarks).",
    )
    parser.add_argument("--resolution", help="Override output resolution as Nx,Ny (e.g. 512,512).")
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MPI parallel yt mode via mpi4py (default: enabled).",
    )
    args = parser.parse_args()

    if args.zoom <= 0:
        raise ValueError("--zoom must be a positive number")

    timer = Timer()

    timer.start("imports")
    import numpy as np
    import yt
    import matplotlib.pyplot as plt

    mpi_mod = None
    if args.parallel:
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None
            comm = None
            rank = 0
            size = 1
            print("Warning: mpi4py not found; running in serial mode.")
        else:
            mpi_mod = MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            yt.enable_parallelism()
    else:
        comm = None
        rank = 0
        size = 1
    timer.stop("imports")

    if comm is not None:
        comm.Barrier()

    try:
        plotfile_path = _resolve_amrex_path(args.plotfile)
    except FileNotFoundError:
        if rank == 0:
            print(f"Plotfile path does not exist: {args.plotfile}")
        return 1

    timer.start("yt/load")
    ds = yt.load(plotfile_path)
    timer.stop("yt/load")

    if comm is not None:
        comm.Barrier()

    timer.start("yt/metadata")
    field = _find_yt_field(ds, args.var)

    domain_left = ds.domain_left_edge.to_value("code_length")
    domain_right = ds.domain_right_edge.to_value("code_length")
    domain_dims0 = np.array(ds.domain_dimensions, dtype=int)
    refine_by = int(getattr(ds, "refine_by", 2))

    if args.amr_cell_average:
        dims = domain_dims0
    else:
        if args.level < 0:
            raise ValueError("--level must be non-negative")
        dims = domain_dims0 * (refine_by ** args.level)

    x_lo, y_lo, z_lo = domain_left
    x_hi, y_hi, z_hi = domain_right

    if args.axis == "z":
        rect = (x_lo, y_lo, x_hi, y_hi)
        resolution = (int(dims[0]), int(dims[1]))
        plane_label = "xy"
        axis_labels = ("x", "y")
    elif args.axis == "y":
        rect = (x_lo, z_lo, x_hi, z_hi)
        resolution = (int(dims[0]), int(dims[2]))
        plane_label = "xz"
        axis_labels = ("x", "z")
    else:
        rect = (y_lo, z_lo, y_hi, z_hi)
        resolution = (int(dims[1]), int(dims[2]))
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

    if args.coord is None:
        axis_idx = _axis_index(args.axis)
        coord = 0.5 * (float(domain_left[axis_idx]) + float(domain_right[axis_idx]))
    else:
        coord = args.coord

    if args.zoom != 1.0:
        x_mid = 0.5 * (rect[0] + rect[2])
        y_mid = 0.5 * (rect[1] + rect[3])
        half_width = 0.5 * (rect[2] - rect[0]) / args.zoom
        half_height = 0.5 * (rect[3] - rect[1]) / args.zoom
        rect = (x_mid - half_width, y_mid - half_height, x_mid + half_width, y_mid + half_height)

    if args.axis == "x":
        center = (coord, 0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3]))
    elif args.axis == "y":
        center = (0.5 * (rect[0] + rect[2]), coord, 0.5 * (rect[1] + rect[3]))
    else:
        center = (0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3]), coord)

    timer.stop("yt/metadata")

    if comm is not None:
        comm.Barrier()

    timer.start("yt/slice")
    slc = ds.slice(_axis_index(args.axis), coord)
    timer.stop("yt/slice")

    if comm is not None:
        comm.Barrier()

    timer.start("yt/frb")
    width = ((rect[2] - rect[0], "code_length"), (rect[3] - rect[1], "code_length"))
    frb = slc.to_frb(width=width, resolution=resolution, center=center)
    timer.stop("yt/frb")

    if comm is not None:
        comm.Barrier()

    timer.start("yt/extract")
    slice_2d = np.array(frb[field], dtype=np.float64)
    timer.stop("yt/extract")

    if comm is not None:
        comm.Barrier()

    if rank == 0 and not args.no_plot:
        timer.start("postprocess/plot")
        fig, ax = plt.subplots(figsize=(6, 5))
        log_slice = np.ma.masked_invalid(np.log10(slice_2d))
        kpc_in_cm = 3.0856775814913673e21
        rect_kpc = tuple(c / kpc_in_cm for c in rect)
        extent_kpc = (rect_kpc[0], rect_kpc[2], rect_kpc[1], rect_kpc[3])
        im = ax.imshow(log_slice, origin="lower", cmap="viridis", extent=extent_kpc)
        avg_label = "AMR cell-average" if args.amr_cell_average else f"Level {args.level}"
        field_label = field[1] if isinstance(field, tuple) else str(field)
        ax.set_title(f"yt {avg_label} slice of {field_label} ({plane_label} plane)")
        ax.set_xlabel(f"{axis_labels[0]} [kpc]")
        ax.set_ylabel(f"{axis_labels[1]} [kpc]")
        fig.colorbar(im, ax=ax, label=f"log10({field_label})")
        fig.tight_layout()
        if args.output:
            fig.savefig(args.output, dpi=150)
        else:
            plt.show()
        timer.stop("postprocess/plot")

    if comm is not None:
        comm.Barrier()

    if comm is not None:
        stage_keys = (
            "imports",
            "yt/load",
            "yt/metadata",
            "yt/slice",
            "yt/frb",
            "yt/extract",
            "postprocess/plot",
        )
        reduced: dict[str, float] = {}
        for key in stage_keys:
            local = float(timer.dt.get(key, 0.0))
            reduced[key] = comm.allreduce(local, op=mpi_mod.MAX)
        total = sum(reduced.values())
    else:
        reduced = dict(timer.dt)
        total = sum(reduced.values())

    if rank == 0:
        print(f"MPI ranks: {size}")
        print("Timing summary (seconds, rank-max):")
        for key in (
            "imports",
            "yt/load",
            "yt/metadata",
            "yt/slice",
            "yt/frb",
            "yt/extract",
            "postprocess/plot",
        ):
            if key in reduced and reduced[key] > 0.0:
                print(f"  {key:16s} {reduced[key]:.6f}")
        print(f"  {'total':16s} {total:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
