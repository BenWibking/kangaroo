#!/usr/bin/env python3
"""Integrate fluxes through fixed-radius cylindrical surfaces in an AMReX plotfile."""

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


COMPONENTS = tuple(name.replace("_sphere", "_cylinder") for name in SPHERE_COMPONENTS)
GEOMETRIC_SECTIONS = ("endcaps", "walls")


def _finite_heights(values: Iterable[float]) -> np.ndarray:
    heights = np.asarray([float(value) for value in values], dtype=np.float64)
    if heights.size == 0 or not np.all(np.isfinite(heights)) or np.any(heights <= 0.0):
        raise ValueError("heights must be finite and positive")
    return heights


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
            raise ValueError("flux values must have shape (num_heights, 2, 2, 4)")
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
                "flux values must have shape (num_heights, 2, num_temperature_bins, 2, 4)"
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
    derived = {
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
    p.add_argument("--height", type=float, help="Half-height in plotfile coordinate units.")
    p.add_argument("--height-kpc", type=float, help="Half-height in kpc; converted to cm.")
    p.add_argument("--zmin-kpc", "--zmin_kpc", dest="zmin_kpc", type=float)
    p.add_argument("--zmax-kpc", "--zmax_kpc", dest="zmax_kpc", type=float)
    p.add_argument("--nbins", type=int, default=64, help="Number of log-spaced half-heights.")
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
    p.add_argument("--bytes-per-value", type=int, choices=(4, 8), default=8)
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
        radius_count = sum(value is not None for value in (a.radius, a.radius_kpc))
        if radius_count != 1:
            raise RuntimeError("Pass exactly one of --radius or --radius-kpc")
        radius = _finite_radius(
            a.radius if a.radius is not None else float(a.radius_kpc) * 1.0e3 * pc_cm
        )

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
        flux = pipe.cylindrical_flux_surface_integral(
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
            height=heights,
            temperature=(
                pipe.field(fields["temperature"][1])
                if temperature_bins is not None
                else None
            ),
            temperature_bins=temperature_bins,
            gamma=float(a.gamma),
            bytes_per_value=int(a.bytes_per_value),
            out="cylindrical_flux_surface_integral",
        )
        pipe.run(progress_bar=bool(a.progress))

        values = rt.get_task_chunk_array(
            step=0,
            level=0,
            field=flux.field,
            version=0,
            block=0,
            shape=(
                (len(heights), 2, len(temperature_bins) - 1, len(GEOMETRIC_SECTIONS), 4)
                if temperature_bins is not None
                else (len(heights), 2, len(GEOMETRIC_SECTIONS), 4)
            ),
            dtype=np.float64,
            dataset=ds,
        )
        flux_rows, derived = _flux_rows_and_derived(
            heights,
            values,
            pc_cm=pc_cm,
            temperature_bins=temperature_bins,
        )
        result = {
            "plotfile": a.plotfile,
            "time": float(bundle.dataset["time"]) if "time" in bundle.dataset else None,
            "radius": float(radius),
            "radius_kpc": float(radius / (1.0e3 * pc_cm)),
            "height": float(heights[0]) if len(heights) == 1 else None,
            "height_kpc": float(heights[0] / (1.0e3 * pc_cm)) if len(heights) == 1 else None,
            "heights": [float(height) for height in heights],
            "heights_kpc": [float(height / (1.0e3 * pc_cm)) for height in heights],
            "nbins": int(len(heights)),
            "temperature_bins": (
                [float(edge) for edge in temperature_bins]
                if temperature_bins is not None
                else None
            ),
            "gamma": float(a.gamma),
            "bytes_per_value": int(a.bytes_per_value),
            "fields": {role: name for role, (name, _) in fields.items()},
            "fluxes": flux_rows[0]["fluxes"] if len(heights) == 1 else None,
            "flux_bins": flux_rows[0]["flux_bins"] if len(heights) == 1 else None,
            "flux_bins_by_geometric_section": (
                flux_rows[0]["flux_bins_by_geometric_section"]
                if len(heights) == 1
                else None
            ),
            "flux_bins_by_temperature": (
                flux_rows[0]["flux_bins_by_temperature"]
                if len(heights) == 1 and temperature_bins is not None
                else None
            ),
            "fluxes_by_height": flux_rows,
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
