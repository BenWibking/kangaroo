#!/usr/bin/env python3
"""Integrate fluxes through a spherical surface in an AMReX plotfile."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterable

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis import Runtime, run_console_main  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402
from analysis.pipeline import pipeline  # noqa: E402


COMPONENTS = (
    "mass_flux_sphere",
    "hydro_energy_flux_sphere",
    "mhd_energy_flux_sphere",
    "passive_scalar_flux_sphere",
)
SIGN_BINS = ("negative", "positive")
MSUN_G = 1.98847e33
YR_S = 365.25 * 24.0 * 3600.0

FIELD_CANDIDATES = {
    "density": ("density", "rho", "gasDensity", "Density"),
    "momx": ("xmom", "momx", "x1Momentum", "x-GasMomentum", "momentum_x", "gasMomentum_x"),
    "momy": ("ymom", "momy", "x2Momentum", "y-GasMomentum", "momentum_y", "gasMomentum_y"),
    "momz": ("zmom", "momz", "x3Momentum", "z-GasMomentum", "momentum_z", "gasMomentum_z"),
    "energy": ("energy", "Egas", "gasEnergy", "GasEnergy", "totalEnergy", "Energy"),
    "scalar": ("scalar_0", "scalar0", "Scalar0", "passive_scalar", "passiveScalar", "scalar"),
    "bx": ("x-BField", "Bx", "bx", "magx", "magnetic_x", "B_x", "cell_centered_Bx"),
    "by": ("y-BField", "By", "by", "magy", "magnetic_y", "B_y", "cell_centered_By"),
    "bz": ("z-BField", "Bz", "bz", "magz", "magnetic_z", "B_z", "cell_centered_Bz"),
}


def _parse_field_arg(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _metadata_var_names(meta: dict) -> list[str]:
    return [str(v) for v in meta.get("var_names", [])]


def _pick_field(role: str, explicit: str | None, available: Iterable[str]) -> str:
    names = list(available)
    name_set = set(names)
    if explicit:
        if explicit not in name_set:
            raise RuntimeError(
                f"Field {explicit!r} for {role!r} is not listed in plotfile metadata. "
                f"Use --list-fields; available fields include: {', '.join(names[:40])}"
            )
        return explicit
    for candidate in FIELD_CANDIDATES[role]:
        if candidate in name_set:
            return candidate
    raise RuntimeError(
        f"Could not infer field for {role!r}. "
        f"Pass --{role}; available fields include: {', '.join(names[:40])}"
    )


def _resolve_required_field(ds, *, role: str, explicit: str | None, available: Iterable[str]) -> tuple[str, int]:
    name = _pick_field(role, explicit, available)
    resolved, field_id, _ = ds.resolve_field(name)
    return resolved, int(field_id)


def _finite_radius(value: float) -> float:
    radius = float(value)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    return radius


def _finite_radii(values: Iterable[float]) -> np.ndarray:
    radii = np.asarray([float(value) for value in values], dtype=np.float64)
    if radii.size == 0 or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii must be finite and positive")
    return radii


def _flux_rows_and_derived(
    radii: np.ndarray,
    values: np.ndarray,
    *,
    pc_cm: float,
) -> tuple[list[dict], dict]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(radii), len(SIGN_BINS), len(COMPONENTS)):
        raise ValueError("flux values must have shape (num_radii, 2, 4)")
    summed_values = values.sum(axis=1)
    flux_rows = [
        {
            "radius": float(radii[i]),
            "radius_kpc": float(radii[i] / (1.0e3 * pc_cm)),
            "fluxes": {name: float(summed_values[i, j]) for j, name in enumerate(COMPONENTS)},
            "flux_bins": {
                sign: {
                    name: float(values[i, sign_idx, j])
                    for j, name in enumerate(COMPONENTS)
                }
                for sign_idx, sign in enumerate(SIGN_BINS)
            },
        }
        for i in range(len(radii))
    ]
    derived = {
        "mass_flux_msun_per_yr": (
            float(summed_values[0, 0] * YR_S / MSUN_G) if len(radii) == 1 else None
        ),
        "mass_flux_msun_per_yr_bins": (
            {
                sign: float(values[0, sign_idx, 0] * YR_S / MSUN_G)
                for sign_idx, sign in enumerate(SIGN_BINS)
            }
            if len(radii) == 1
            else None
        ),
        "mass_flux_msun_per_yr_by_radius": [
            {
                "radius": float(radii[i]),
                "radius_kpc": float(radii[i] / (1.0e3 * pc_cm)),
                "mass_flux_msun_per_yr": float(summed_values[i, 0] * YR_S / MSUN_G),
                "mass_flux_msun_per_yr_bins": {
                    sign: float(values[i, sign_idx, 0] * YR_S / MSUN_G)
                    for sign_idx, sign in enumerate(SIGN_BINS)
                },
            }
            for i in range(len(radii))
        ],
    }
    return flux_rows, derived


def _bounds_1d(lo: int, hi: int, x0: float, dx: float, origin: int) -> tuple[float, float]:
    return x0 + (lo - origin) * dx, x0 + (hi + 1 - origin) * dx


def _block_intersects_sphere(level_meta, block, radius: float) -> bool:
    radius2 = radius * radius
    geom = level_meta.geom
    lo2 = 0.0
    hi2 = 0.0
    for axis in range(3):
        x0, x1 = _bounds_1d(
            block.lo[axis],
            block.hi[axis],
            geom.x0[axis],
            geom.dx[axis],
            geom.index_origin[axis],
        )
        if x1 < 0.0:
            lo2 += x1 * x1
        elif x0 > 0.0:
            lo2 += x0 * x0
        hi2 += max(abs(x0), abs(x1)) ** 2
    return lo2 <= radius2 <= hi2


def _intersecting_validation_blocks(ds, runmeta, *, radius: float) -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    for level_idx, level in enumerate(runmeta.steps[ds.step].levels):
        for block_idx, block in enumerate(level.boxes):
            if _block_intersects_sphere(level, block, radius):
                samples.append((level_idx, block_idx))
                break
    if not samples:
        raise ValueError("radius does not intersect any mesh block")
    return samples


def _intersecting_validation_blocks_for_radii(ds, runmeta, *, radii: Iterable[float]) -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for radius in radii:
        for sample in _intersecting_validation_blocks(ds, runmeta, radius=float(radius)):
            if sample not in seen:
                seen.add(sample)
                samples.append(sample)
    return samples


def _validate_selected_fields(
    ds,
    runmeta,
    fields: dict[str, tuple[str, int]],
    *,
    radii: Iterable[float],
) -> None:
    _intersecting_validation_blocks_for_radii(ds, runmeta, radii=radii)
    available = set(_metadata_var_names(ds.metadata))
    for role, (name, field_id) in fields.items():
        if name not in available:
            raise RuntimeError(
                f"Selected field {role!r} -> {name!r} is not listed in plotfile metadata. "
                f"field_id={field_id}"
            )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run Kangaroo flux_surface_integral on a real AMReX plotfile."
    )
    p.add_argument("plotfile")
    p.add_argument("--radius", type=float, help="Sphere radius in plotfile coordinate units.")
    p.add_argument("--radius-kpc", type=float, help="Sphere radius in kpc; converted to cm.")
    p.add_argument(
        "--rmin-kpc",
        "--rmin_kpc",
        dest="rmin_kpc",
        type=float,
        help="Minimum sphere radius in kpc.",
    )
    p.add_argument(
        "--rmax-kpc",
        "--rmax_kpc",
        dest="rmax_kpc",
        type=float,
        help="Maximum sphere radius in kpc.",
    )
    p.add_argument(
        "--nbins",
        dest="nbins",
        type=int,
        default=64,
        help="Number of log-spaced radii for --rmin-kpc/--rmax-kpc.",
    )
    p.add_argument("--density")
    p.add_argument("--momx")
    p.add_argument("--momy")
    p.add_argument("--momz")
    p.add_argument("--energy")
    p.add_argument("--scalar")
    p.add_argument("--bx")
    p.add_argument("--by")
    p.add_argument("--bz")
    p.add_argument("--gamma", type=float, default=5.0 / 3.0)
    p.add_argument(
        "--bytes-per-value",
        type=int,
        choices=(4, 8),
        default=8,
        help="Input field precision in bytes; defaults to 8 for double-precision plotfiles.",
    )
    p.add_argument("--output-json", help="Write flux values and field bindings to this JSON file.")
    p.add_argument("--list-fields", action="store_true", help="Print plotfile field names and exit.")
    p.add_argument("--progress", action="store_true", help="Show Kangaroo task progress.")
    a, u = p.parse_known_args()

    rt = Runtime.from_parsed_args(a, unknown_args=u)

    def _run() -> int:
        ds = open_dataset(a.plotfile, runtime=rt, step=0, level=0)
        bundle = ds.metadata_bundle()
        runmeta = bundle.runmeta
        available = _metadata_var_names(bundle.dataset)

        if a.list_fields:
            for idx, name in enumerate(available):
                print(f"{idx:03d} {name}")
            return 0

        pc_cm = 3.0856775814913673e18
        single_radius_count = sum(value is not None for value in (a.radius, a.radius_kpc))
        range_radius_count = sum(value is not None for value in (a.rmin_kpc, a.rmax_kpc))
        if single_radius_count == 0 and range_radius_count == 0:
            raise RuntimeError("Pass --radius, --radius-kpc, or both --rmin-kpc and --rmax-kpc")
        if single_radius_count > 0 and range_radius_count > 0:
            raise RuntimeError("Pass either a single radius or --rmin-kpc/--rmax-kpc, not both")
        if single_radius_count > 1:
            raise RuntimeError("Pass only one of --radius or --radius-kpc")
        if range_radius_count not in (0, 2):
            raise RuntimeError("Pass both --rmin-kpc and --rmax-kpc")

        if range_radius_count == 2:
            if int(a.nbins) <= 0:
                raise ValueError("nbins must be positive")
            rmin_kpc = _finite_radius(a.rmin_kpc)
            rmax_kpc = _finite_radius(a.rmax_kpc)
            if rmin_kpc > rmax_kpc:
                raise ValueError("rmin_kpc must be less than or equal to rmax_kpc")
            radii_kpc = _finite_radii(
                np.logspace(np.log10(rmin_kpc), np.log10(rmax_kpc), int(a.nbins))
            )
            radii = _finite_radii(radii_kpc * 1.0e3 * pc_cm)
        else:
            radius = _finite_radius(
                a.radius if a.radius is not None else float(a.radius_kpc) * 1.0e3 * pc_cm
            )
            radii = _finite_radii([radius])

        fields: dict[str, tuple[str, int]] = {}
        for role in FIELD_CANDIDATES:
            fields[role] = _resolve_required_field(
                ds,
                role=role,
                explicit=_parse_field_arg(getattr(a, role)),
                available=available,
            )

        print(
            "flux surface fields: "
            + ", ".join(f"{role}={name}" for role, (name, _) in fields.items()),
            file=sys.stderr,
            flush=True,
        )
        _validate_selected_fields(ds, runmeta, fields, radii=radii)
        bytes_per_value = int(a.bytes_per_value)

        pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        flux = pipe.flux_surface_integral(
            pipe.field(fields["density"][1]),
            momentum=(
                pipe.field(fields["momx"][1]),
                pipe.field(fields["momy"][1]),
                pipe.field(fields["momz"][1]),
            ),
            energy=pipe.field(fields["energy"][1]),
            passive_scalar=pipe.field(fields["scalar"][1]),
            magnetic_field=(
                pipe.field(fields["bx"][1]),
                pipe.field(fields["by"][1]),
                pipe.field(fields["bz"][1]),
            ),
            radius=radii,
            gamma=float(a.gamma),
            bytes_per_value=bytes_per_value,
            out="flux_surface_integral",
        )
        pipe.run(progress_bar=bool(a.progress))

        values = rt.get_task_chunk_array(
            step=0,
            level=0,
            field=flux.field,
            version=0,
            block=0,
            shape=(len(radii), 2, 4),
            dtype=np.float64,
            dataset=ds,
        )
        flux_rows, derived = _flux_rows_and_derived(radii, values, pc_cm=pc_cm)
        result = {
            "plotfile": a.plotfile,
            "time": float(bundle.dataset["time"]) if "time" in bundle.dataset else None,
            "radius": float(radii[0]) if len(radii) == 1 else None,
            "radius_kpc": float(radii[0] / (1.0e3 * pc_cm)) if len(radii) == 1 else None,
            "radii": [float(radius) for radius in radii],
            "radii_kpc": [float(radius / (1.0e3 * pc_cm)) for radius in radii],
            "nbins": int(len(radii)),
            "gamma": float(a.gamma),
            "bytes_per_value": int(bytes_per_value),
            "fields": {role: name for role, (name, _) in fields.items()},
            "fluxes": flux_rows[0]["fluxes"] if len(radii) == 1 else None,
            "flux_bins": flux_rows[0]["flux_bins"] if len(radii) == 1 else None,
            "fluxes_by_radius": flux_rows,
        }
        result["derived"] = derived

        print(json.dumps(result, indent=2, sort_keys=True))
        if a.output_json:
            with open(a.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
                f.write("\n")
        return 0

    return int(run_console_main(rt, _run))


if __name__ == "__main__":
    raise SystemExit(main())
