#!/usr/bin/env python3
"""Measure enclosed-SFR profiles within origin-centered cylinders in one pass.

Quokka deposits ``particle.StochasticStellarPop.birth_mass_density`` via
``DerivedParticleDeposition`` as the CIC birth-mass density of stochastic
stellar populations younger than ``t_age`` (5 Myr in the Phase1_EMEF deck),
normalized by ``normalization_expr = 1/(5 Myr)``.  The deposited field is
therefore an SFR rate density in g cm^-3 s^-1 (the default
``--field-kind rate_density``): the enclosed SFR is the plain volume integral
of the field over the cylinder, with no extra window division.  For
plotfiles whose deposited field is a plain birth-mass density in g cm^-3,
pass ``--field-kind mass_density`` and the averaging window
``--sfr-window-myr``; the enclosed SFR is then the integral divided by that
window.

Both profiles are derived from a single AMR-aware 2-D cylindrical-moments
pass (kernel ``cylindrical_moments_accumulate``): uncovered cells are binned
by cylindrical radius and |z|, accumulating the field integral and sampled
volume per (r, z) bin.  Enclosed quantities follow by cumulative summation,
so the disk profile SFR(R) for |z| <= --disk-height-kpc and the wind profile
SFR(h) for r <= --wind-radius-kpc share one field read and one accumulate
pass.  Coarse cells covered by finer levels are excluded, and cells are
binned by cell-center coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis import Runtime, run_console_main  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402
from analysis.pipeline import pipeline  # noqa: E402

PC_CM = 3.0856775814913673e18
KPC_CM = 1.0e3 * PC_CM
MSUN_G = 1.98847e33
YR_S = 365.25 * 24.0 * 3600.0
MYR_S = 1.0e6 * YR_S
DEFAULT_FIELD = "particle.StochasticStellarPop.birth_mass_density"
FIELD_KINDS = ("rate_density", "mass_density")


def _finite_positive(value: float, description: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{description} must be finite and positive")
    return result


def _finite_positive_log_range(
    low: float, high: float, nbins: int, description: str
) -> np.ndarray:
    low = _finite_positive(low, f"{description} minimum")
    high = _finite_positive(high, f"{description} maximum")
    if high <= low:
        raise ValueError(f"{description} maximum must exceed its minimum")
    if int(nbins) <= 0:
        raise ValueError("nbins must be positive")
    values = np.logspace(np.log10(low), np.log10(high), int(nbins))
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{description} values must be finite and positive")
    return values


def _metadata_var_names(meta: dict) -> list[str]:
    return [str(name) for name in meta.get("var_names", [])]


def _merged_edges(scan_cm: np.ndarray, anchor_cm: float) -> np.ndarray:
    """Edges [0] ∪ scan ∪ {anchor}, sorted and deduplicated."""

    values = [0.0]
    values.extend(float(v) for v in np.asarray(scan_cm, dtype=np.float64))
    values.append(float(anchor_cm))
    values = sorted(values)
    deduplicated = [values[0]]
    for value in values[1:]:
        if value > deduplicated[-1]:
            deduplicated.append(value)
    return np.asarray(deduplicated, dtype=np.float64)


def _upper_edge_indices(edges: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Bin index whose upper edge equals each sample (samples must be edges)."""

    lookup = {float(edge): idx for idx, edge in enumerate(edges[1:])}
    indices = []
    for sample in np.asarray(samples, dtype=np.float64):
        idx = lookup.get(float(sample))
        if idx is None:
            raise RuntimeError(
                f"Sample edge {sample!r} is missing from the merged edge grid."
            )
        indices.append(int(idx))
    return np.asarray(indices, dtype=np.int64)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Measure enclosed-SFR profiles within origin-centered cylinders "
            "from the deposited StochasticStellarPop birth-mass density in a "
            "single AMR-aware cylindrical-moments pass."
        )
    )
    p.add_argument("plotfile")
    p.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        help=(
            "Deposited birth-mass density field "
            f"(default: {DEFAULT_FIELD})."
        ),
    )
    p.add_argument(
        "--field-kind",
        choices=FIELD_KINDS,
        default="rate_density",
        help=(
            "Units of the deposited field: 'rate_density' is an SFR rate "
            "density in g cm^-3 s^-1 (Quokka's normalized birth-mass "
            "deposition), so the enclosed SFR is the plain volume integral; "
            "'mass_density' is a plain birth-mass density in g cm^-3, so the "
            "enclosed SFR is the integral divided by --sfr-window-myr "
            "(default: rate_density)."
        ),
    )
    p.add_argument(
        "--sfr-window-myr",
        type=float,
        default=10.0,
        help=(
            "Formation window of the deposited SSP tracer in Myr; used only "
            "with --field-kind mass_density (default: 10)."
        ),
    )
    p.add_argument(
        "--rmin-kpc", "--rmin_kpc", dest="rmin_kpc", type=float, default=0.1,
        help="Innermost radius of the disk scan in kpc (default: 0.1).",
    )
    p.add_argument(
        "--rmax-kpc", "--rmax_kpc", dest="rmax_kpc", type=float, default=30.0,
        help="Outermost radius of the disk scan in kpc (default: 30).",
    )
    p.add_argument(
        "--zmin-kpc", "--zmin_kpc", dest="zmin_kpc", type=float, default=0.1,
        help="Innermost half-height of the wind scan in kpc (default: 0.1).",
    )
    p.add_argument(
        "--zmax-kpc", "--zmax_kpc", dest="zmax_kpc", type=float, default=30.0,
        help="Outermost half-height of the wind scan in kpc (default: 30).",
    )
    p.add_argument(
        "--nbins",
        type=int,
        default=50,
        help=(
            "Number of log-spaced sample radii and half-heights "
            "(default: 50)."
        ),
    )
    p.add_argument(
        "--disk-height-kpc",
        "--disk_height_kpc",
        dest="disk_height_kpc",
        type=float,
        default=1.0,
        help="Disk half-height |z| for the SFR(R) profile (default: 1).",
    )
    p.add_argument(
        "--wind-radius-kpc",
        "--wind_radius_kpc",
        dest="wind_radius_kpc",
        type=float,
        default=5.0,
        help="Cylinder radius for the SFR(h) profile (default: 5).",
    )
    p.add_argument(
        "--output-json",
        help=(
            "Output JSON for the disk profile SFR(R) "
            "(default: enclosed_sfr_disk.json)."
        ),
    )
    p.add_argument(
        "--output-json-wind",
        "--output_json_wind",
        dest="output_json_wind",
        help=(
            "Output JSON for the wind profile SFR(h) "
            "(default: enclosed_sfr_wind.json)."
        ),
    )
    p.add_argument(
        "--output-json-joint",
        "--output_json_joint",
        dest="output_json_joint",
        help=(
            "Output JSON for the joint 2-D bin integrals "
            "(default: enclosed_sfr_joint.json)."
        ),
    )
    p.add_argument("--list-fields", action="store_true")
    p.add_argument("--progress", action="store_true")
    a, u = p.parse_known_args()

    rt = Runtime.from_parsed_args(a, unknown_args=u)

    def _run() -> int:
        ds = open_dataset(a.plotfile, runtime=rt, step=0, level=0)
        bundle = ds.metadata_bundle()
        available = _metadata_var_names(bundle.dataset)

        if a.list_fields:
            for idx, name in enumerate(available):
                print(f"{idx:03d} {name}")
            return 0

        field_kind = str(a.field_kind)
        if field_kind not in FIELD_KINDS:
            raise RuntimeError(f"Unsupported field kind: {field_kind!r}")
        window_myr = _finite_positive(a.sfr_window_myr, "sfr-window-myr")
        window_s = window_myr * MYR_S

        radii_kpc = _finite_positive_log_range(
            a.rmin_kpc, a.rmax_kpc, a.nbins, "radius"
        )
        heights_kpc = _finite_positive_log_range(
            a.zmin_kpc, a.zmax_kpc, a.nbins, "half-height"
        )
        height_kpc = _finite_positive(a.disk_height_kpc, "disk-height-kpc")
        radius_kpc = _finite_positive(a.wind_radius_kpc, "wind-radius-kpc")
        if radius_kpc <= float(radii_kpc[0]):
            raise RuntimeError(
                "wind-radius-kpc must exceed the smallest scan radius"
            )
        if height_kpc <= float(heights_kpc[0]):
            raise RuntimeError(
                "disk-height-kpc must exceed the smallest scan half-height"
            )

        field_name = str(a.field)
        if field_name not in available:
            matches = [name for name in available if "birth_mass" in name]
            hint = f" Similar fields: {', '.join(matches)}" if matches else ""
            raise RuntimeError(
                f"Field {field_name!r} not found in plotfile.{hint}"
            )
        resolved_name, field_id, _ = ds.resolve_field(field_name)
        print(
            f"enclosed SFR field: {resolved_name}, kind = {field_kind}"
            + (
                f", window = {window_myr:g} Myr"
                if field_kind == "mass_density"
                else ""
            ),
            file=sys.stderr,
            flush=True,
        )

        radial_edges_cm = _merged_edges(radii_kpc * KPC_CM, radius_kpc * KPC_CM)
        z_edges_cm = _merged_edges(heights_kpc * KPC_CM, height_kpc * KPC_CM)
        print(
            f"cylindrical moments: {len(radial_edges_cm) - 1} radial bins x "
            f"{len(z_edges_cm) - 1} height bins, one pass",
            file=sys.stderr,
            flush=True,
        )

        pipe = pipeline(runtime=rt, runmeta=bundle.runmeta, dataset=ds)
        birth = pipe.field(int(field_id))
        handle = pipe.cylindrical_moments(
            birth,
            radial_edges=tuple(float(v) for v in radial_edges_cm),
            z_edges=tuple(float(v) for v in z_edges_cm),
            center=(0.0, 0.0, 0.0),
            out="enclosed_sfr_moments",
        )
        pipe.run(progress_bar=bool(a.progress))

        moments = rt.get_task_chunk_array(
            step=0,
            level=0,
            field=handle.field,
            version=0,
            block=0,
            dtype=np.float64,
            dataset=ds,
        )
        bins_r, bins_z = handle.bins
        moments = np.asarray(moments, dtype=np.float64).reshape(
            bins_r, bins_z, 2
        )
        bin_integral = moments[:, :, 0]
        bin_volume = moments[:, :, 1]
        if not np.all(np.isfinite(bin_integral)) or not np.all(
            np.isfinite(bin_volume)
        ):
            raise RuntimeError("Cylindrical moments must be finite.")

        radius_of_bin = radial_edges_cm[1:]
        disk_stop_z = _upper_edge_indices(z_edges_cm, [height_kpc * KPC_CM])[0]
        wind_stop_r = _upper_edge_indices(
            radial_edges_cm, [radius_kpc * KPC_CM]
        )[0]

        # Disk profile: SFR(R) for |z| <= disk height.  Bin (i_r, i_z) spans
        # [radial_edges[i_r], radial_edges[i_r + 1]) x [z_edges[i_z],
        # z_edges[i_z + 1]); bins are upper-edge inclusive at the grid ends.
        disk_annulus = bin_integral[:, : disk_stop_z + 1].sum(axis=1)
        disk_annulus_volume = bin_volume[:, : disk_stop_z + 1].sum(axis=1)
        disk_enclosed = np.cumsum(disk_annulus)
        disk_volume = np.cumsum(disk_annulus_volume)

        # Wind profile: SFR(h) within the wind cylinder radius.
        wind_slab = bin_integral[: wind_stop_r + 1, :].sum(axis=0)
        wind_slab_volume = bin_volume[: wind_stop_r + 1, :].sum(axis=0)
        wind_enclosed = np.cumsum(wind_slab)
        wind_volume = np.cumsum(wind_slab_volume)

        if field_kind == "rate_density":
            # The field is an SFR rate density in g cm^-3 s^-1: the volume
            # integral is the SFR in g/s, converted to M_sun/yr.
            sfr_factor = YR_S / MSUN_G
        else:
            # The field is a plain birth-mass density in g cm^-3: divide the
            # volume integral (a mass) by the averaging window.
            sfr_factor = YR_S / (window_s * MSUN_G)
        disk_sfr = disk_enclosed * sfr_factor
        wind_sfr = wind_enclosed * sfr_factor

        time_seconds = (
            float(bundle.dataset["time"]) if "time" in bundle.dataset else None
        )
        field_units = (
            "g cm^-3 s^-1" if field_kind == "rate_density" else "g cm^-3"
        )
        integral_units = "g s^-1" if field_kind == "rate_density" else "g"
        definition = (
            "enclosed SFR = volume integral of the deposited SFR rate density "
            "over the cylinder"
            if field_kind == "rate_density"
            else (
                "enclosed SFR = (volume integral of the deposited birth-mass "
                "density over the cylinder) / sfr_window"
            )
        )
        common: dict = {
            "plotfile": str(a.plotfile),
            "field": resolved_name,
            "time": time_seconds,
            "field_kind": field_kind,
            "field_units": field_units,
            "integral_units": integral_units,
            "sfr_window_myr": (
                float(window_myr) if field_kind == "mass_density" else None
            ),
            "sfr_window_s": (
                float(window_s) if field_kind == "mass_density" else None
            ),
            "center": [0.0, 0.0, 0.0],
            "radius_binning": "cell-center",
            "amr_coverage": "coarse cells covered by finer levels excluded",
            "units": "M_sun / yr",
            "definition": definition,
        }

        disk_scan_r = _upper_edge_indices(radial_edges_cm, radii_kpc * KPC_CM)
        wind_scan_z = _upper_edge_indices(z_edges_cm, heights_kpc * KPC_CM)

        disk_result = dict(common)
        disk_result.update(
            {
                "mode": "radius",
                "height_kpc": float(height_kpc),
                "z_bounds_kpc": [-float(height_kpc), float(height_kpc)],
                "radii_kpc": [float(v) for v in radii_kpc],
                "annulus_field_integral": [
                    float(v) for v in disk_annulus[disk_scan_r]
                ],
                "enclosed_field_integral": [
                    float(v) for v in disk_enclosed[disk_scan_r]
                ],
                "enclosed_sfr_msun_per_yr": [
                    float(v) for v in disk_sfr[disk_scan_r]
                ],
                "sampled_volume_kpc3": [
                    float(v / KPC_CM**3) for v in disk_volume[disk_scan_r]
                ],
            }
        )

        wind_result = dict(common)
        wind_result.update(
            {
                "mode": "height",
                "radius_kpc": float(radius_kpc),
                "heights_kpc": [float(v) for v in heights_kpc],
                "enclosed_field_integral": [
                    float(v) for v in wind_enclosed[wind_scan_z]
                ],
                "enclosed_sfr_msun_per_yr": [
                    float(v) for v in wind_sfr[wind_scan_z]
                ],
                "sampled_volume_kpc3": [
                    float(v / KPC_CM**3) for v in wind_volume[wind_scan_z]
                ],
            }
        )

        joint_result = dict(common)
        joint_result.update(
            {
                "mode": "joint",
                "radial_edges_kpc": [
                    float(v / KPC_CM) for v in radial_edges_cm
                ],
                "z_edges_kpc": [float(v / KPC_CM) for v in z_edges_cm],
                "bin_field_integral": [
                    [float(v) for v in row] for row in bin_integral
                ],
                "bin_sampled_volume_kpc3": [
                    [float(v / KPC_CM**3) for v in row] for row in bin_volume
                ],
            }
        )

        print(json.dumps(joint_result, indent=2, sort_keys=True))
        base = os.path.dirname(os.path.abspath(a.plotfile))
        default_dir = (
            base if os.access(base, os.W_OK) else os.path.abspath(os.curdir)
        )
        outputs = (
            (
                a.output_json
                or os.path.join(default_dir, "enclosed_sfr_disk.json"),
                disk_result,
            ),
            (
                a.output_json_wind
                or os.path.join(default_dir, "enclosed_sfr_wind.json"),
                wind_result,
            ),
            (
                a.output_json_joint
                or os.path.join(default_dir, "enclosed_sfr_joint.json"),
                joint_result,
            ),
        )
        for path, payload in outputs:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"wrote {path}", file=sys.stderr, flush=True)
        return 0

    return run_console_main(rt, _run)


if __name__ == "__main__":
    raise SystemExit(main())
