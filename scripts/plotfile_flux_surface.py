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
    if explicit:
        return explicit
    names = list(available)
    name_set = set(names)
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


def _validate_selected_fields(
    ds,
    rt,
    runmeta,
    fields: dict[str, tuple[str, int]],
    *,
    radius: float,
) -> None:
    validation_blocks = _intersecting_validation_blocks(ds, runmeta, radius=radius)
    for role, (name, field_id) in fields.items():
        for level_idx, block_idx in validation_blocks:
            try:
                raw = rt.get_task_chunk(
                    step=ds.step,
                    level=level_idx,
                    field=field_id,
                    version=0,
                    block=block_idx,
                    dataset=ds,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Selected field {role!r} -> {name!r} is not readable at level {level_idx}, "
                    f"block {block_idx}. Use --list-fields and pass explicit field names if "
                    "inference picked the wrong component."
                ) from exc
            if not raw:
                raise RuntimeError(
                    f"Selected field {role!r} -> {name!r} returned an empty chunk at "
                    f"level {level_idx}, block {block_idx}."
                )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run Kangaroo flux_surface_integral on a real AMReX plotfile."
    )
    p.add_argument("plotfile")
    p.add_argument("--radius", type=float, help="Sphere radius in plotfile coordinate units.")
    p.add_argument("--radius-kpc", type=float, help="Sphere radius in kpc; converted to cm.")
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

        if a.radius is None and a.radius_kpc is None:
            raise RuntimeError("Pass either --radius or --radius-kpc")
        if a.radius is not None and a.radius_kpc is not None:
            raise RuntimeError("Pass only one of --radius or --radius-kpc")

        pc_cm = 3.0856775814913673e18
        radius = _finite_radius(a.radius if a.radius is not None else float(a.radius_kpc) * 1.0e3 * pc_cm)

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
        _validate_selected_fields(ds, rt, runmeta, fields, radius=radius)

        bytes_per_value = ds.infer_bytes_per_value(rt, field=fields["density"][1], level=0)

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
            radius=radius,
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
            shape=(4,),
            dtype=np.float64,
            dataset=ds,
        )

        result = {
            "plotfile": a.plotfile,
            "radius": radius,
            "radius_kpc": radius / (1.0e3 * pc_cm),
            "gamma": float(a.gamma),
            "bytes_per_value": int(bytes_per_value),
            "fields": {role: name for role, (name, _) in fields.items()},
            "fluxes": {name: float(values[i]) for i, name in enumerate(COMPONENTS)},
        }

        msun_g = 1.98847e33
        yr_s = 365.25 * 24.0 * 3600.0
        result["derived"] = {
            "mass_flux_msun_per_yr": float(values[0] * yr_s / msun_g),
        }

        print(json.dumps(result, indent=2, sort_keys=True))
        if a.output_json:
            with open(a.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
                f.write("\n")
        return 0

    return int(run_console_main(rt, _run))


if __name__ == "__main__":
    raise SystemExit(main())
