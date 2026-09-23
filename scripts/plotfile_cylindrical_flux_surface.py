#!/usr/bin/env python3
"""Integrate fluxes through origin-centered, z-aligned cylinders in a plotfile.

Use --radius-kpc and --height-kpc for one cylinder. For a one-dimensional
profile, either vary half-height with --zmin-kpc/--zmax-kpc at fixed radius or
vary radius with --rmin-kpc/--rmax-kpc at fixed half-height. The wall section
contains radial angular-momentum flux; endcaps contain the vertical
angular-momentum flux. Magnetic fields must use B**2/2 energy normalization.
Angular-momentum flux is in g cm**2 s**-2 for cgs input. Sign bins classify
each flux's sign, not the gas velocity.
"""

from __future__ import annotations

import argparse
import json
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
from scripts.plotfile_flux_surface import (  # noqa: E402
    COMPONENTS as SPHERE_COMPONENTS,
    FIELD_CANDIDATES,
    MSUN_G,
    SIGN_BINS,
    TEMPERATURE_CANDIDATES,
    YR_S,
    _finite_radius,
    _metadata_var_names,
    _parse_field_arg,
    _parse_temperature_bins,
    _pick_field,
    _resolve_required_field,
)


COMPONENTS = tuple(name.replace("_sphere", "_cylinder") for name in SPHERE_COMPONENTS) + (
    "advective_radial_angular_momentum_flux_cylinder",
    "maxwell_radial_angular_momentum_flux_cylinder",
    "advective_vertical_angular_momentum_flux_cylinder",
    "maxwell_vertical_angular_momentum_flux_cylinder",
)
GEOMETRIC_SECTIONS = ("endcaps", "walls")


def _finite_heights(values: Iterable[float]) -> np.ndarray:
    heights = np.asarray([float(value) for value in values], dtype=np.float64)
    if heights.size == 0 or not np.all(np.isfinite(heights)) or np.any(heights <= 0.0):
        raise ValueError("heights must be finite and positive")
    return heights


def _finite_radii(values: Iterable[float]) -> np.ndarray:
    radii = np.asarray([float(value) for value in values], dtype=np.float64)
    if radii.size == 0 or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii must be finite and positive")
    return radii


def _flux_rows_and_derived(
    heights: np.ndarray,
    values: np.ndarray,
    *,
    pc_cm: float,
    temperature_bins: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    values = np.asarray(values, dtype=np.float64)
    if temperature_bins is None:
        if values.shape == (len(heights), len(SIGN_BINS), len(COMPONENTS)):
            values_by_temperature_section = np.zeros(
                (
                    len(heights),
                    len(SIGN_BINS),
                    1,
                    len(GEOMETRIC_SECTIONS),
                    len(COMPONENTS),
                ),
                dtype=np.float64,
            )
            values_by_temperature_section[:, :, 0, 1, :] = values
        elif values.shape == (
            len(heights),
            len(SIGN_BINS),
            len(GEOMETRIC_SECTIONS),
            len(COMPONENTS),
        ):
            values_by_temperature_section = values[:, :, np.newaxis, :, :]
        else:
            raise ValueError("flux values must have shape (num_heights, 2, 2, 8)")
        temperature_edges = None
    else:
        expected_shape = (
            len(heights),
            len(SIGN_BINS),
            len(temperature_bins) - 1,
            len(GEOMETRIC_SECTIONS),
            len(COMPONENTS),
        )
        legacy_shape = (
            len(heights),
            len(SIGN_BINS),
            len(temperature_bins) - 1,
            len(COMPONENTS),
        )
        if values.shape == legacy_shape:
            values_by_temperature_section = np.zeros(expected_shape, dtype=np.float64)
            values_by_temperature_section[:, :, :, 1, :] = values
        elif values.shape != expected_shape:
            raise ValueError(
                "flux values must have shape (num_heights, 2, num_temperature_bins, 2, 8)"
            )
        else:
            values_by_temperature_section = values
        temperature_edges = temperature_bins
    values_by_temperature = values_by_temperature_section.sum(axis=3)
    sign_summed_values = values_by_temperature.sum(axis=2)
    summed_values = sign_summed_values.sum(axis=1)
    section_summed_values = values_by_temperature_section.sum(axis=2)

    def temperature_bin_rows(height_idx: int, sign_idx: int) -> list[dict]:
        if temperature_edges is None:
            return []
        return [
            {
                "temperature_min": float(temperature_edges[temp_idx]),
                "temperature_max": float(temperature_edges[temp_idx + 1]),
                "fluxes": {
                    name: float(
                        values_by_temperature[height_idx, sign_idx, temp_idx, j]
                    )
                    for j, name in enumerate(COMPONENTS)
                },
                "fluxes_by_geometric_section": {
                    section: {
                        name: float(
                            values_by_temperature_section[
                                height_idx, sign_idx, temp_idx, section_idx, j
                            ]
                        )
                        for j, name in enumerate(COMPONENTS)
                    }
                    for section_idx, section in enumerate(GEOMETRIC_SECTIONS)
                },
            }
            for temp_idx in range(len(temperature_edges) - 1)
        ]

    flux_rows = [
        {
            "height": float(heights[i]),
            "height_kpc": float(heights[i] / (1.0e3 * pc_cm)),
            "z_min": float(-heights[i]),
            "z_max": float(heights[i]),
            "fluxes": {name: float(summed_values[i, j]) for j, name in enumerate(COMPONENTS)},
            "flux_bins": {
                sign: {
                    name: float(sign_summed_values[i, sign_idx, j])
                    for j, name in enumerate(COMPONENTS)
                }
                for sign_idx, sign in enumerate(SIGN_BINS)
            },
            "flux_bins_by_geometric_section": {
                sign: {
                    section: {
                        name: float(section_summed_values[i, sign_idx, section_idx, j])
                        for j, name in enumerate(COMPONENTS)
                    }
                    for section_idx, section in enumerate(GEOMETRIC_SECTIONS)
                }
                for sign_idx, sign in enumerate(SIGN_BINS)
            },
            "flux_bins_by_temperature": (
                {
                    sign: temperature_bin_rows(i, sign_idx)
                    for sign_idx, sign in enumerate(SIGN_BINS)
                }
                if temperature_edges is not None
                else None
            ),
        }
        for i in range(len(heights))
    ]
    # Sum both component sign bins before forming the net total. Opposite
    # angular-momentum signs need not correspond to inward/outward gas.
    net_by_section = section_summed_values.sum(axis=1)
    derived = {
        "radial_angular_momentum_flux_by_height": [
            {
                "height": float(heights[i]),
                "height_kpc": float(heights[i] / (1.0e3 * pc_cm)),
                "by_geometric_section": {
                    section: {
                        "advective": float(net_by_section[i, section_idx, 4]),
                        "maxwell": float(net_by_section[i, section_idx, 5]),
                        "total": float(net_by_section[i, section_idx, 4:6].sum()),
                    }
                    for section_idx, section in enumerate(GEOMETRIC_SECTIONS)
                },
            }
            for i in range(len(heights))
        ],
        "vertical_angular_momentum_flux_by_height": [
            {
                "height": float(heights[i]),
                "height_kpc": float(heights[i] / (1.0e3 * pc_cm)),
                "by_geometric_section": {
                    section: {
                        "advective": float(net_by_section[i, section_idx, 6]),
                        "maxwell": float(net_by_section[i, section_idx, 7]),
                        "total": float(net_by_section[i, section_idx, 6:8].sum()),
                    }
                    for section_idx, section in enumerate(GEOMETRIC_SECTIONS)
                },
            }
            for i in range(len(heights))
        ],
        "mass_flux_msun_per_yr": (
            float(summed_values[0, 0] * YR_S / MSUN_G) if len(heights) == 1 else None
        ),
        "mass_flux_msun_per_yr_by_height": [
            {
                "height": float(heights[i]),
                "height_kpc": float(heights[i] / (1.0e3 * pc_cm)),
                "mass_flux_msun_per_yr": float(summed_values[i, 0] * YR_S / MSUN_G),
                "mass_flux_msun_per_yr_bins": {
                    sign: float(sign_summed_values[i, sign_idx, 0] * YR_S / MSUN_G)
                    for sign_idx, sign in enumerate(SIGN_BINS)
                },
                "mass_flux_msun_per_yr_bins_by_geometric_section": {
                    sign: {
                        section: float(
                            section_summed_values[i, sign_idx, section_idx, 0]
                            * YR_S
                            / MSUN_G
                        )
                        for section_idx, section in enumerate(GEOMETRIC_SECTIONS)
                    }
                    for sign_idx, sign in enumerate(SIGN_BINS)
                },
            }
            for i in range(len(heights))
        ],
    }
    return flux_rows, derived


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run Kangaroo cylindrical_flux_surface_integral on a real AMReX plotfile."
    )
    p.add_argument("plotfile")
    p.add_argument("--radius", type=float, help="Cylinder radius in plotfile coordinate units.")
    p.add_argument("--radius-kpc", type=float, help="Cylinder radius in kpc; converted to cm.")
    p.add_argument("--rmin-kpc", "--rmin_kpc", dest="rmin_kpc", type=float)
    p.add_argument("--rmax-kpc", "--rmax_kpc", dest="rmax_kpc", type=float)
    p.add_argument("--height", type=float, help="Half-height in plotfile coordinate units.")
    p.add_argument("--height-kpc", type=float, help="Half-height in kpc; converted to cm.")
    p.add_argument("--zmin-kpc", "--zmin_kpc", dest="zmin_kpc", type=float)
    p.add_argument("--zmax-kpc", "--zmax_kpc", dest="zmax_kpc", type=float)
    p.add_argument(
        "--nbins",
        type=int,
        default=64,
        help="Number of log-spaced values in the requested radius or half-height range.",
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
    p.add_argument("--temperature")
    p.add_argument("--temperature-bins", nargs="+")
    p.add_argument("--gamma", type=float, default=5.0 / 3.0)
    p.add_argument("--output-json")
    p.add_argument("--list-fields", action="store_true")
    p.add_argument("--progress", action="store_true")
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
            raise RuntimeError(
                "Pass --radius, --radius-kpc, or both --rmin-kpc and --rmax-kpc"
            )
        if single_radius_count > 0 and range_radius_count > 0:
            raise RuntimeError(
                "Pass either a single radius or --rmin-kpc/--rmax-kpc, not both"
            )
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
                a.radius
                if a.radius is not None
                else float(a.radius_kpc) * 1.0e3 * pc_cm
            )
            radii = _finite_radii([radius])

        single_height_count = sum(value is not None for value in (a.height, a.height_kpc))
        range_height_count = sum(value is not None for value in (a.zmin_kpc, a.zmax_kpc))
        if single_height_count == 0 and range_height_count == 0:
            raise RuntimeError("Pass --height, --height-kpc, or both --zmin-kpc and --zmax-kpc")
        if single_height_count > 0 and range_height_count > 0:
            raise RuntimeError("Pass either a single height or --zmin-kpc/--zmax-kpc, not both")
        if single_height_count > 1:
            raise RuntimeError("Pass only one of --height or --height-kpc")
        if range_height_count not in (0, 2):
            raise RuntimeError("Pass both --zmin-kpc and --zmax-kpc")
        if range_height_count == 2:
            if int(a.nbins) <= 0:
                raise ValueError("nbins must be positive")
            zmin_kpc = _finite_radius(a.zmin_kpc)
            zmax_kpc = _finite_radius(a.zmax_kpc)
            if zmin_kpc > zmax_kpc:
                raise ValueError("zmin_kpc must be less than or equal to zmax_kpc")
            heights_kpc = _finite_heights(
                np.logspace(np.log10(zmin_kpc), np.log10(zmax_kpc), int(a.nbins))
            )
            heights = _finite_heights(heights_kpc * 1.0e3 * pc_cm)
        else:
            height = _finite_radius(
                a.height if a.height is not None else float(a.height_kpc) * 1.0e3 * pc_cm
            )
            heights = _finite_heights([height])

        if len(radii) > 1 and len(heights) > 1:
            raise RuntimeError(
                "Radius and half-height cannot both vary in one run; use a radius "
                "range with --height/--height-kpc, or a single radius with a height range"
            )

        temperature_bins = (
            _parse_temperature_bins(a.temperature_bins)
            if a.temperature_bins is not None
            else None
        )

        fields: dict[str, tuple[str, int]] = {}
        for role in FIELD_CANDIDATES:
            fields[role] = _resolve_required_field(
                ds,
                role=role,
                explicit=_parse_field_arg(getattr(a, role)),
                available=available,
            )
        if temperature_bins is not None:
            temperature_name = _pick_field(
                "temperature",
                _parse_field_arg(a.temperature),
                available,
                candidates=TEMPERATURE_CANDIDATES,
            )
            resolved, field_id, _ = ds.resolve_field(temperature_name)
            fields["temperature"] = (resolved, int(field_id))

        print(
            "cylindrical flux surface fields: "
            + ", ".join(f"{role}={name}" for role, (name, _) in fields.items()),
            file=sys.stderr,
            flush=True,
        )
        pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        density = pipe.field(fields["density"][1])
        momentum = tuple(
            pipe.field(fields[role][1]) for role in ("momx", "momy", "momz")
        )
        energy = pipe.field(fields["energy"][1])
        passive_scalar = pipe.field(fields["scalar"][1])
        magnetic_field = tuple(
            pipe.field(fields[role][1]) for role in ("bx", "by", "bz")
        )
        temperature = (
            pipe.field(fields["temperature"][1])
            if temperature_bins is not None
            else None
        )
        fluxes = [
            pipe.cylindrical_flux_surface_integral(
                density,
                momentum=momentum,
                energy=energy,
                passive_scalar=passive_scalar,
                magnetic_field=magnetic_field,
                radius=float(radius),
                height=heights,
                temperature=temperature,
                temperature_bins=temperature_bins,
                gamma=float(a.gamma),
                out=f"cylindrical_flux_surface_integral_r{radius_idx}",
            )
            for radius_idx, radius in enumerate(radii)
        ]
        pipe.run(progress_bar=bool(a.progress))

        value_shape = (
            (
                len(heights),
                2,
                len(temperature_bins) - 1,
                len(GEOMETRIC_SECTIONS),
                len(COMPONENTS),
            )
            if temperature_bins is not None
            else (len(heights), 2, len(GEOMETRIC_SECTIONS), len(COMPONENTS))
        )
        rows_and_derived = []
        for flux in fluxes:
            values = rt.get_task_chunk_array(
                step=0,
                level=0,
                field=flux.field,
                version=0,
                block=0,
                dtype=np.float64,
                dataset=ds,
            )
            values = values.reshape(value_shape)
            rows_and_derived.append(
                _flux_rows_and_derived(
                    heights,
                    values,
                    pc_cm=pc_cm,
                    temperature_bins=temperature_bins,
                )
            )

        flux_rows, first_derived = rows_and_derived[0]
        flux_rows_by_radius = []
        derived_by_radius = {
            "mass_flux_msun_per_yr_by_radius": [],
            "radial_angular_momentum_flux_by_radius": [],
            "vertical_angular_momentum_flux_by_radius": [],
        }
        if len(heights) == 1:
            for radius, (radius_flux_rows, radius_derived) in zip(
                radii, rows_and_derived
            ):
                row = dict(radius_flux_rows[0])
                row.update(
                    radius=float(radius),
                    radius_kpc=float(radius / (1.0e3 * pc_cm)),
                )
                flux_rows_by_radius.append(row)
                for height_key, radius_key in (
                    ("mass_flux_msun_per_yr_by_height", "mass_flux_msun_per_yr_by_radius"),
                    (
                        "radial_angular_momentum_flux_by_height",
                        "radial_angular_momentum_flux_by_radius",
                    ),
                    (
                        "vertical_angular_momentum_flux_by_height",
                        "vertical_angular_momentum_flux_by_radius",
                    ),
                ):
                    derived_row = dict(radius_derived[height_key][0])
                    derived_row.update(
                        radius=float(radius),
                        radius_kpc=float(radius / (1.0e3 * pc_cm)),
                    )
                    derived_by_radius[radius_key].append(derived_row)
            if len(radii) > 1:
                derived = derived_by_radius
            else:
                derived = dict(first_derived)
                derived.update(derived_by_radius)
        else:
            derived = first_derived
        result = {
            "plotfile": a.plotfile,
            "time": float(bundle.dataset["time"]) if "time" in bundle.dataset else None,
            "radius": float(radii[0]) if len(radii) == 1 else None,
            "radius_kpc": (
                float(radii[0] / (1.0e3 * pc_cm)) if len(radii) == 1 else None
            ),
            "radii": [float(radius) for radius in radii],
            "radii_kpc": [float(radius / (1.0e3 * pc_cm)) for radius in radii],
            "height": float(heights[0]) if len(heights) == 1 else None,
            "height_kpc": float(heights[0] / (1.0e3 * pc_cm)) if len(heights) == 1 else None,
            "heights": [float(height) for height in heights],
            "heights_kpc": [float(height / (1.0e3 * pc_cm)) for height in heights],
            "nbins": int(max(len(radii), len(heights))),
            "num_radii": int(len(radii)),
            "num_heights": int(len(heights)),
            "temperature_bins": (
                [float(edge) for edge in temperature_bins]
                if temperature_bins is not None
                else None
            ),
            "gamma": float(a.gamma),
            "radial_angular_momentum_axis": "z",
            "vertical_angular_momentum_axis": "z",
            "endcap_normal_convention": "upper +z, lower -z; both summed",
            "center": [0.0, 0.0, 0.0],
            "magnetic_normalization": "magnetic_energy_density = B^2/2",
            "radial_angular_momentum_flux_units": "g cm^2 s^-2 (for cgs input)",
            "vertical_angular_momentum_flux_units": "g cm^2 s^-2 (for cgs input)",
            "sign_bin_convention": "sign of each flux component, not velocity",
            "components": list(COMPONENTS),
            "fields": {role: name for role, (name, _) in fields.items()},
            "fluxes": (
                flux_rows[0]["fluxes"]
                if len(radii) == 1 and len(heights) == 1
                else None
            ),
            "flux_bins": (
                flux_rows[0]["flux_bins"]
                if len(radii) == 1 and len(heights) == 1
                else None
            ),
            "flux_bins_by_geometric_section": (
                flux_rows[0]["flux_bins_by_geometric_section"]
                if len(radii) == 1 and len(heights) == 1
                else None
            ),
            "flux_bins_by_temperature": (
                flux_rows[0]["flux_bins_by_temperature"]
                if len(radii) == 1
                and len(heights) == 1
                and temperature_bins is not None
                else None
            ),
            "fluxes_by_height": flux_rows if len(radii) == 1 else None,
            "fluxes_by_radius": flux_rows_by_radius if len(heights) == 1 else None,
            "derived": derived,
        }

        print(json.dumps(result, indent=2, sort_keys=True))
        if a.output_json:
            with open(a.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
                f.write("\n")
        return 0

    return run_console_main(rt, _run)


if __name__ == "__main__":
    raise SystemExit(main())
