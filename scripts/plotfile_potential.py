#!/usr/bin/env python3
"""Reconstruct portable total-potential models from an explicit snapshot sequence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from potential_replay.fit import fit_samples, quadrature_points


def read_inputs(paths: list[str], manifest: Path | None) -> list[tuple[Path, dict]]:
    """CSV accepts the campaign's plotfile/phase columns; time columns are provenance only."""
    if bool(paths) == bool(manifest):
        raise ValueError("provide plotfile paths OR --manifest")
    base = Path.cwd()
    if manifest:
        source = Path(manifest).expanduser().resolve()
        base = source.parent
        with source.open() as stream:
            if source.suffix.lower() == ".csv":
                entries = list(csv.DictReader(stream))
            else:
                entries = json.load(stream)
                if isinstance(entries, dict):
                    entries = entries["snapshots"]
    else:
        entries = [{"plotfile": p} for p in paths]
    if not entries:
        raise ValueError("empty snapshot list")
    result = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"plotfile": entry}
        path = Path(entry["plotfile"]).expanduser()
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        if not (path / "Header").is_file():
            raise FileNotFoundError(f"full plotfile Header unavailable: {path}")
        result.append((path, dict(entry)))
    if len({p for p, _ in result}) != len(result):
        raise ValueError("duplicate plotfile in input list")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plotfiles", nargs="*")
    p.add_argument(
        "--manifest",
        type=Path,
        help="JSON list or CSV with plotfile and optional phase",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--field", default="gpot")
    p.add_argument(
        "--center",
        nargs=3,
        type=float,
        required=True,
        help="fixed center in native length units",
    )
    p.add_argument(
        "--orientation",
        nargs=9,
        type=float,
        default=np.eye(3).ravel().tolist(),
        help="row-major world_from_model orthonormal matrix",
    )
    p.add_argument(
        "--rmax", type=float, help="native length units; required unless --knots"
    )
    p.add_argument("--zmin", type=float)
    p.add_argument("--zmax", type=float)
    p.add_argument("--nr", type=int, default=33)
    p.add_argument("--nz", type=int, default=33)
    p.add_argument(
        "--knots",
        type=Path,
        help='JSON {"radii": [...], "heights": [...]} in native units',
    )
    p.add_argument("--nphi", type=int, default=32)
    p.add_argument("--mmax", type=int, default=4)
    p.add_argument(
        "--support",
        type=float,
        required=True,
        help="MLS cube half-width in native length units",
    )
    p.add_argument("--condition-limit", type=float, default=1e10)
    p.add_argument("--sample-batch-size", type=int, default=128)
    p.add_argument("--length-to-cm", type=float, required=True)
    p.add_argument("--time-to-s", type=float, required=True)
    p.add_argument("--potential-to-cm2-s2", type=float, required=True)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--hpx-config", action="append", default=[])
    p.add_argument("--hpx-arg", action="append", default=[])
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    inputs = read_inputs(args.plotfiles, args.manifest)
    if args.knots:
        if any(v is not None for v in (args.rmax, args.zmin, args.zmax)):
            raise ValueError("--knots cannot be combined with domain bounds")
        knots = json.loads(args.knots.read_text())
        radii, heights = np.asarray(knots["radii"]), np.asarray(knots["heights"])
    else:
        if any(v is None for v in (args.rmax, args.zmin, args.zmax)):
            raise ValueError("provide --rmax, --zmin and --zmax or --knots")
        radii = np.linspace(0, args.rmax, args.nr)
        heights = np.linspace(args.zmin, args.zmax, args.nz)
    rotation = np.asarray(args.orientation).reshape(3, 3)
    points = quadrature_points(
        radii, heights, args.nphi, center=args.center, rotation=rotation
    )
    units = {
        "length_to_cm": args.length_to_cm,
        "time_to_s": args.time_to_s,
        "potential_to_cm2_s2": args.potential_to_cm2_s2,
    }
    # Validate numerical choices before starting HPX or writing anything.
    fit_samples(
        radii,
        heights,
        np.zeros(points.shape[:-1]),
        mmax=args.mmax,
        center=args.center,
        rotation=rotation,
        time=0,
        units=units,
    )
    if (
        not np.isfinite(args.support)
        or args.support <= 0
        or args.sample_batch_size < 1
        or not np.isfinite(args.condition_limit)
        or args.condition_limit <= 1
    ):
        raise ValueError("invalid support, batch size or conditioning limit")
    output = args.output_dir.expanduser().resolve()
    names = [f"snapshot_{i:04d}_{path.name}.npz" for i, (path, _) in enumerate(inputs)]
    for name in [*names, "timeseries.json"]:
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite {output / name}")

    from analysis import PlotfileReader, Runtime, run_console_main
    from analysis.dataset import open_dataset
    from analysis.potential_extract import extract_samples

    runtime = Runtime.from_parsed_args(args)

    def run():
        records = []
        # Read all metadata first: reject missing fields, non-3D data, duplicate times.
        opened = []
        for path, source in inputs:
            ds = open_dataset(str(path), runtime=runtime, step=0, level=0)
            bundle = ds.metadata_bundle()
            meta = bundle.dataset
            header_meta = PlotfileReader(str(path)).metadata()
            if header_meta["spacedim"] != 3 or header_meta["coord_sys"] != 0:
                raise ValueError(f"expected full 3D plotfile: {path}")
            # Current dataset metadata adapter assumes isotropic cells. Do not silently
            # use its geometry for anisotropic files until that separate API supports it.
            if any(
                not np.allclose(dx, [dx[0]] * 3, rtol=1e-12, atol=0)
                for dx in meta["cell_size"]
            ):
                raise ValueError(
                    "plotfile adapter currently requires isotropic AMR cells"
                )
            time = float(meta["time"])
            if not np.isfinite(time):
                raise ValueError(f"nonfinite Header time: {path}")
            if args.field not in meta["var_names"]:
                raise ValueError(f"field {args.field!r} not found in {path}")
            _, field, _ = ds.resolve_field(args.field)
            opened.append((ds, bundle, int(field), time))
        if len({item[3] for item in opened}) != len(opened):
            raise ValueError("snapshot Header times must be distinct")
        output.mkdir(parents=True, exist_ok=True)
        for (path, source), name, (ds, bundle, field, time) in zip(
            inputs, names, opened
        ):
            print(f"extracting {path} at Header time {time:.17g}", flush=True)
            samples, diagnostics = extract_samples(
                runtime,
                ds,
                bundle.runmeta,
                field,
                points.reshape(-1, 3),
                half_width=args.support,
                batch_size=args.sample_batch_size,
                condition_limit=args.condition_limit,
                progress=args.progress,
            )
            meta = bundle.dataset
            geometry = {
                key: meta[key]
                for key in (
                    "cell_size",
                    "ref_ratio",
                    "prob_lo",
                    "prob_hi",
                    "prob_domain",
                )
            }
            geometry["box_counts"] = [len(b) for b in meta["level_boxes"]]
            geometry["boxes_sha256"] = hashlib.sha256(
                json.dumps(meta["level_boxes"], separators=(",", ":")).encode()
            ).hexdigest()
            provenance = {
                "plotfile": str(path),
                "field": args.field,
                "input_record": source,
                "header_sha256": hashlib.sha256(
                    (path / "Header").read_bytes()
                ).hexdigest(),
                "geometry": geometry,
                "estimator": "leaf-volume quadratic MLS",
                "support_half_width": args.support,
                "condition_limit": args.condition_limit,
                "coverage_tolerance": 1e-8,
                "sample_batch_size": args.sample_batch_size,
                "diagnostics": {
                    key: {"min": float(value.min()), "max": float(value.max())}
                    for key, value in diagnostics.items()
                },
            }
            model = fit_samples(
                radii,
                heights,
                samples.reshape(points.shape[:-1]),
                mmax=args.mmax,
                center=args.center,
                rotation=rotation,
                time=time,
                units=units,
                provenance=provenance,
            )
            model.save(output / name)
            records.append(
                {
                    "model": name,
                    "time": time,
                    "time_seconds": time * args.time_to_s,
                    "phase": source.get("phase"),
                    "plotfile": str(path),
                    "cell_size": meta["cell_size"],
                    "header_sha256": provenance["header_sha256"],
                }
            )
            print(f"wrote {output / name}", flush=True)
        records.sort(key=lambda entry: entry["time"])
        series = {
            "schema": "kangaroo.potential-timeseries",
            "version": 1,
            "snapshots": records,
            "units": units,
            "center": args.center,
            "world_from_model": rotation.tolist(),
            "frame": "fixed",
            "temporal_interpolation": "none",
            "input_manifest": str(args.manifest.resolve()) if args.manifest else None,
        }
        with (output / "timeseries.json").open("x") as stream:
            json.dump(series, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return 0

    return int(run_console_main(runtime, run))


if __name__ == "__main__":
    raise SystemExit(main())
