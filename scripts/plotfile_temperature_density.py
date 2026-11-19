#!/usr/bin/env python3
"""Compute a mass-weighted density–temperature phase diagram from an AMReX plotfile."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("plotfile")
    parser.add_argument("--density-range", nargs=2, type=float, required=True, metavar=("MIN", "MAX"))
    parser.add_argument("--temperature-range", nargs=2, type=float, required=True, metavar=("MIN", "MAX"))
    parser.add_argument("--bins", nargs=2, type=int, default=(128, 128), metavar=("DENSITY", "TEMPERATURE"))
    parser.add_argument("--density")
    parser.add_argument("--temperature")
    parser.add_argument("--output-json", type=Path, default=Path("temperature_density_histogram.json"))
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    args, unknown = parser.parse_known_args()
    for name, (lo, hi) in (("density", args.density_range), ("temperature", args.temperature_range)):
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi
                and math.log10(lo) < math.log10(hi)):
            parser.error(f"--{name}-range requires two finite, positive, increasing bounds")
    if not all(0 < n <= 2**31 - 1 for n in args.bins):
        parser.error("--bins requires two positive 32-bit integers")
    if unknown and args.validate_only:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args, unknown


def main() -> int:
    args, unknown = parse_args()
    if args.validate_only:
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np

    from analysis import Runtime, run_console_main
    from analysis.dataset import open_dataset
    from analysis.pipeline import pipeline
    from analysis.runtime import trace_phase
    from scripts.plotfile_flux_surface import MSUN_G, TEMPERATURE_CANDIDATES, _pick_field

    rt = Runtime.from_parsed_args(args, unknown_args=unknown)

    def compute() -> int:
        with trace_phase("python_open_plotfile"):
            ds = open_dataset(args.plotfile, runtime=rt)
        with trace_phase("python_load_plotfile_metadata"):
            bundle = ds.metadata_bundle()
            available = bundle.dataset["var_names"]
            density = _pick_field("density", args.density, available)
            temperature = _pick_field(
                "temperature", args.temperature, available, candidates=TEMPERATURE_CANDIDATES
            )
        with trace_phase("python_build_temperature_density_plan"):
            pipe = pipeline(runtime=rt, runmeta=bundle.runmeta, dataset=ds)
            hist = pipe.histogram2d(
                pipe.field(density), pipe.field(temperature),
                x_range=tuple(args.density_range), y_range=tuple(args.temperature_range),
                bins=tuple(args.bins), weight_mode="cell_mass", log_bins=True,
            )
        pipe.run(progress_bar=args.progress)
        counts = rt.get_task_chunk_array(
            step=ds.step, level=ds.level, field=hist.counts.field, block=0, dataset=ds
        )
        mass = np.asarray(counts, dtype=np.float64) / MSUN_G
        density_edges, temperature_edges = hist.edges
        payload = {
            "plotfile": args.plotfile,
            "time": bundle.dataset.get("time"),
            "density_field": density,
            "temperature_field": temperature,
            "density_units": "g/cm^3",
            "temperature_units": "K",
            "mass_units": "Msun",
            "axis_order": ["density", "temperature"],
            "density_edges": density_edges,
            "temperature_edges": temperature_edges,
            "mass_msun": mass.tolist(),
            "total_mass_msun": float(mass.sum(dtype=np.float64)),
        }
        args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {args.output_json} (in-range mass {payload['total_mass_msun']:.6g} Msun)")
        return 0

    return run_console_main(rt, compute)


if __name__ == "__main__":
    raise SystemExit(main())
