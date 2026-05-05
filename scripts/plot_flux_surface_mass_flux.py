#!/usr/bin/env python3
"""Plot mass flux as a function of radius from flux_surface.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MSUN_PER_YEAR_LABEL = r"$M_\odot\,yr^{-1}$"
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_MYR = 1.0e6 * SECONDS_PER_YEAR


def _load_mass_flux(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | None]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    time = data.get("time")
    time_myr = None if time is None else float(time) / SECONDS_PER_MYR

    rows = data.get("derived", {}).get("mass_flux_msun_per_yr_by_radius")
    if rows is not None:
        radius = np.asarray([row["radius_kpc"] for row in rows], dtype=np.float64)
        mass_flux = np.asarray(
            [row["mass_flux_msun_per_yr"] for row in rows],
            dtype=np.float64,
        )
        mass_flux_negative = np.asarray(
            [
                row.get("mass_flux_msun_per_yr_bins", {}).get(
                    "negative",
                    min(row["mass_flux_msun_per_yr"], 0.0),
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        mass_flux_positive = np.asarray(
            [
                row.get("mass_flux_msun_per_yr_bins", {}).get(
                    "positive",
                    max(row["mass_flux_msun_per_yr"], 0.0),
                )
                for row in rows
            ],
            dtype=np.float64,
        )
    else:
        radii_kpc = data.get("radii_kpc")
        flux_rows = data.get("fluxes_by_radius")
        if radii_kpc is None or flux_rows is None:
            raise ValueError(
                "JSON must contain derived.mass_flux_msun_per_yr_by_radius "
                "or both radii_kpc and fluxes_by_radius."
            )

        msun_g = 1.98847e33
        radius = np.asarray(radii_kpc, dtype=np.float64)
        mass_flux = np.asarray(
            [
                row["fluxes"]["mass_flux_sphere"] * SECONDS_PER_YEAR / msun_g
                for row in flux_rows
            ],
            dtype=np.float64,
        )
        mass_flux_negative = np.asarray(
            [
                row.get("flux_bins", {})
                .get("negative", {})
                .get(
                    "mass_flux_sphere",
                    min(row["fluxes"]["mass_flux_sphere"], 0.0),
                )
                * SECONDS_PER_YEAR
                / msun_g
                for row in flux_rows
            ],
            dtype=np.float64,
        )
        mass_flux_positive = np.asarray(
            [
                row.get("flux_bins", {})
                .get("positive", {})
                .get(
                    "mass_flux_sphere",
                    max(row["fluxes"]["mass_flux_sphere"], 0.0),
                )
                * SECONDS_PER_YEAR
                / msun_g
                for row in flux_rows
            ],
            dtype=np.float64,
        )

    if radius.size == 0:
        raise ValueError("No radius samples found.")
    if not (
        radius.shape
        == mass_flux.shape
        == mass_flux_negative.shape
        == mass_flux_positive.shape
    ):
        raise ValueError("Radius and mass-flux arrays have different lengths.")
    if not (
        np.all(np.isfinite(radius))
        and np.all(np.isfinite(mass_flux))
        and np.all(np.isfinite(mass_flux_negative))
        and np.all(np.isfinite(mass_flux_positive))
    ):
        raise ValueError("Radius and mass-flux values must be finite.")

    order = np.argsort(radius)
    return (
        radius[order],
        mass_flux[order],
        mass_flux_negative[order],
        mass_flux_positive[order],
        time_myr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot mass flux versus radius from a flux_surface.json file."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="flux_surface.json",
        help="Input flux-surface JSON file (default: flux_surface.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mass_flux_vs_radius.png",
        help="Output image path (default: mass_flux_vs_radius.png).",
    )
    parser.add_argument(
        "--title",
        default="Mass flux through spherical surfaces",
        help="Plot title.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of symlog.",
    )
    args = parser.parse_args()

    radius_kpc, mass_flux, mass_flux_negative, mass_flux_positive, time_myr = (
        _load_mass_flux(Path(args.input))
    )
    title = args.title
    if time_myr is not None:
        title = f"{title}, t = {time_myr:.3f} Myr"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        radius_kpc,
        mass_flux_positive,
        marker="o",
        linewidth=1.6,
        markersize=4,
        label="positive",
    )
    ax.plot(
        radius_kpc,
        mass_flux_negative,
        marker="o",
        linewidth=1.6,
        markersize=4,
        label="negative",
    )
    ax.plot(
        radius_kpc,
        mass_flux,
        color="0.25",
        linestyle="--",
        linewidth=1.2,
        label="net",
    )
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xscale("log")
    if not args.linear_y:
        ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("radius [kpc]")
    ax.set_ylabel(f"mass flux [{MSUN_PER_YEAR_LABEL}]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
