#!/usr/bin/env python3
"""Compute a particle-mass histogram using the Kangaroo C++ runtime."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from analysis import Runtime  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402
from analysis.pipeline import pipeline  # noqa: E402


def _parse_range(range_arg: str) -> tuple[float, float]:
    try:
        r0, r1 = (float(v.strip()) for v in range_arg.split(","))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("--range must be provided as min,max") from exc
    if r0 == r1:
        raise ValueError("--range must include two distinct values")
    if r0 > r1:
        raise ValueError("--range must be increasing (min,max)")
    if not np.isfinite(r0) or not np.isfinite(r1):
        raise ValueError("--range must be finite")
    return r0, r1


def _write_csv(path: Path, edges: np.ndarray, counts: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_lo", "bin_hi", "count"])
        for lo, hi, count in zip(edges[:-1], edges[1:], counts):
            writer.writerow([f"{lo:.8e}", f"{hi:.8e}", f"{count:.8e}"])


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute a histogram of particle masses using the Kangaroo C++ runtime."
    )
    p.add_argument("plotfile")
    p.add_argument(
        "--particle-type",
        "--particles",
        dest="particle_type",
        default=None,
        help="Particle species name (defaults to the first available).",
    )
    p.add_argument("--bins", type=int, default=64, help="Number of histogram bins.")
    p.add_argument("--range", dest="hist_range", help="Histogram range as min,max.")
    p.add_argument(
        "--log-range",
        action="store_true",
        help="Interpret --range values as log10(mass).",
    )
    p.add_argument(
        "--density",
        action="store_true",
        help="Normalize counts by total and bin width.",
    )
    p.add_argument(
        "--output",
        help="Optional output path (.npz for NumPy, otherwise CSV).",
    )
    p.add_argument(
        "--plot-output",
        default="particle_mass_histogram.png",
        help="PNG path for the histogram plot (default: particle_mass_histogram.png).",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top modes (exact mass values) to report.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting the histogram.",
    )
    a, u = p.parse_known_args()

    rt = Runtime.from_parsed_args(a, unknown_args=u)
    ds = open_dataset(a.plotfile, runtime=rt, step=0, level=0)
    runmeta = ds.get_runmeta()

    particle_types = ds.list_particle_types()
    if not particle_types:
        raise RuntimeError("No particle species found in plotfile.")
    particle_type = a.particle_type or particle_types[0]
    if particle_type not in particle_types:
        available = ", ".join(particle_types)
        raise RuntimeError(
            f"particle type '{particle_type}' not found in plotfile; available: {available}"
        )
    particle_fields = ds.list_particle_fields(particle_type)
    if "mass" not in particle_fields:
        available = ", ".join(particle_fields) if particle_fields else "<none>"
        raise RuntimeError(
            f"particle field 'mass' not found for '{particle_type}'; available: {available}"
        )

    topk_specified = any(arg == "--topk" or arg.startswith("--topk=") for arg in sys.argv[1:])
    if topk_specified:
        if a.topk > 0:
            pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
            mode_vals, mode_counts = pipe.particle_topk_modes(
                particle_type,
                "mass",
                k=a.topk,
            )
            msun_g = 1.98847e33
            print("topk_mass_modes (value_msun,count):")
            for v, c in zip(mode_vals, mode_counts):
                if not np.isfinite(v) or c <= 0:
                    continue
                print(f"{v / msun_g:.8e},{int(c)}")
        return 0

    hist_range: tuple[float, float] | None = None
    if a.hist_range:
        r0, r1 = _parse_range(a.hist_range)
        if a.log_range:
            hist_range = (10.0**r0, 10.0**r1)
        else:
            hist_range = (r0, r1)
    else:
        range_pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        mass_h = range_pipe.particle_field(particle_type, "mass")
        rmin = range_pipe.particle_min(mass_h)
        rmax = range_pipe.particle_max(mass_h)
        if not np.isfinite(rmin) or not np.isfinite(rmax):
            raise RuntimeError("Particle mass range contains non-finite values.")
        if rmin == rmax:
            raise RuntimeError("Particle mass range is degenerate; provide --range explicitly.")
        hist_range = (min(rmin, rmax), max(rmin, rmax))
    if hist_range[0] <= 0.0:
        raise RuntimeError("Log-mass binning requires a strictly positive minimum mass.")

    pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    mass_h = pipe.particle_field(particle_type, "mass")
    log_edges = np.linspace(
        np.log10(hist_range[0]),
        np.log10(hist_range[1]),
        int(a.bins) + 1,
    )
    edges = np.power(10.0, log_edges)
    # Guard against floating-point rounding excluding min/max from the edge range.
    edges[0] = min(edges[0], hist_range[0])
    edges[-1] = max(edges[-1], hist_range[1])
    edges[0] = np.nextafter(edges[0], 0.0)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    counts, edges = pipe.particle_histogram1d(
        mass_h,
        bins=edges,
        density=a.density,
    )
    if np.sum(counts) == 0.0:
        verify_pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        v_mass = verify_pipe.particle_field(particle_type, "mass")
        in_range = verify_pipe.particle_count(
            verify_pipe.particle_and(
                verify_pipe.particle_gt(v_mass, hist_range[0]),
                verify_pipe.particle_le(v_mass, hist_range[1]),
            )
        )
        if in_range > 0:
            print(
                "Warning: runtime histogram returned all zeros; "
                "falling back to a NumPy histogram for this run."
            )
            masses = v_mass.values
            mask = np.isfinite(masses) & (masses >= hist_range[0]) & (masses <= hist_range[1])
            masses = masses[mask]
            counts, edges = np.histogram(masses, bins=edges, density=a.density)

    if a.topk > 0:
        mode_vals, mode_counts = pipe.particle_topk_modes(
            particle_type,
            "mass",
            k=a.topk,
        )
        msun_g = 1.98847e33
        print("topk_mass_modes (value_msun,count):")
        for v, c in zip(mode_vals, mode_counts):
            if not np.isfinite(v) or c <= 0:
                continue
            print(f"{v / msun_g:.8e},{int(c)}")

    log_range = (np.log10(hist_range[0]), np.log10(hist_range[1]))
    print(
        "particle_type="
        f"{particle_type} bins={int(a.bins)} "
        f"log_range=({log_range[0]:.6e}, {log_range[1]:.6e}) "
        f"mass_range=({hist_range[0]:.6e}, {hist_range[1]:.6e})"
    )
    print("bin_lo,bin_hi,count")
    for lo, hi, count in zip(edges[:-1], edges[1:], counts):
        print(f"{lo:.8e},{hi:.8e},{count:.8e}")

    if a.output:
        out_path = Path(a.output)
        if out_path.suffix == ".npz":
            np.savez(out_path, edges=edges, counts=counts)
        else:
            _write_csv(out_path, edges, counts)

    if np.sum(counts) == 0.0:
        print(
            "Warning: histogram counts are all zero; check --range/--log-range "
            "and that particle masses fall within the selected range."
        )

    if not a.no_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        widths = np.diff(edges)
        ax.bar(edges[:-1], counts, width=widths, align="edge", edgecolor="black")
        msun_g = 1.98847e33
        ax.set_xlabel("particle mass [Msun]")
        ax.set_ylabel("count" if not a.density else "density")
        ax.set_title(f"Particle mass histogram ({particle_type})")
        if np.any(edges <= 0.0):
            raise RuntimeError("Log-scaled histogram requires positive mass bin edges.")
        ax.set_xscale("log")
        if np.all(counts > 0.0):
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1.0)
        ticks = ax.get_xticks()
        if ticks.size:
            ax.set_xticklabels([f"{t / msun_g:g}" for t in ticks])
        fig.tight_layout()
        fig.savefig(a.plot_output, dpi=150)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
