#!/usr/bin/env python3
"""Plot mass flux as a function of radius from flux_surface.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MSUN_PER_YEAR_LABEL = r"$M_\odot\,yr^{-1}$"


def _load_mass_flux(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    rows = data.get("derived", {}).get("mass_flux_msun_per_yr_by_radius")
    if rows is not None:
        radius = np.asarray([row["radius_kpc"] for row in rows], dtype=np.float64)
        mass_flux = np.asarray(
            [row["mass_flux_msun_per_yr"] for row in rows],
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
        yr_s = 365.25 * 24.0 * 3600.0
        radius = np.asarray(radii_kpc, dtype=np.float64)
        mass_flux = np.asarray(
            [row["fluxes"]["mass_flux_sphere"] * yr_s / msun_g for row in flux_rows],
            dtype=np.float64,
        )

    if radius.size == 0:
        raise ValueError("No radius samples found.")
    if radius.shape != mass_flux.shape:
        raise ValueError("Radius and mass-flux arrays have different lengths.")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(mass_flux)):
        raise ValueError("Radius and mass-flux values must be finite.")

    order = np.argsort(radius)
    return radius[order], mass_flux[order]


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

    radius_kpc, mass_flux = _load_mass_flux(Path(args.input))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(radius_kpc, mass_flux, marker="o", linewidth=1.6, markersize=4)
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xscale("log")
    if not args.linear_y:
        ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("radius [kpc]")
    ax.set_ylabel(f"mass flux [{MSUN_PER_YEAR_LABEL}]")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
