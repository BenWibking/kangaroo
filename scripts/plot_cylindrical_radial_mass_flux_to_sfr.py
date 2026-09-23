#!/usr/bin/env python3
"""Plot the ratio of radial wall mass flux to enclosed SFR versus radius.

The disk panel of ``angular_momentum_flux_wind_vs_disk.png`` overlays the net
mass flux through the cylinder walls, Phi_mass(R) (negative = inward, in
M_sun/yr), on the enclosed star formation rate SFR(<R) within the same slab
(|z| <= 1 kpc).  Their ratio Phi_mass / SFR(<R) is a dimensionless measure of
how much of the enclosed star formation is replenished by radial inflow
through the wall at radius R: -1 means inflow exactly balances the enclosed
SFR (steady gas mass inside R), values between -1 and 0 mean partial
replenishment drawing down the local reservoir, values below -1 mean the
interior is gaining gas, and values above 0 mean net outflow.

Radii and wall fluxes come from ``cylindrical_radial_flux.json``.  The
enclosed SFR is interpolated in log-radius onto those sample positions from
``enclosed_sfr_disk.json``, which must cover the same |z| <= 1 kpc slab.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import plot_cylindrical_wind_disk_angular_momentum_flux as wind_disk

FLUX_JSON = "cylindrical_radial_flux.json"
SFR_JSON = "enclosed_sfr_disk.json"
RATIO_COLOR = "crimson"
SLAB_TOLERANCE_KPC = 1.0e-6


def _sfr_at_radii(
    sfr_x: np.ndarray,
    sfr: np.ndarray,
    radii: np.ndarray,
    description: str,
) -> np.ndarray:
    """Interpolate the enclosed SFR in log-radius onto sample positions."""

    if radii.size == 0:
        raise ValueError(f"The {description} scan contains no radii.")
    if (
        radii.min() < sfr_x.min() * (1.0 - SLAB_TOLERANCE_KPC)
        or radii.max() > sfr_x.max() * (1.0 + SLAB_TOLERANCE_KPC)
    ):
        raise ValueError(
            f"The {description} scan spans radii "
            f"[{radii.min():.3g}, {radii.max():.3g}] kpc outside the "
            f"enclosed-SFR support [{sfr_x.min():.3g}, {sfr_x.max():.3g}] kpc."
        )
    return np.interp(np.log(radii), np.log(sfr_x), sfr)


def _check_slab(
    flux_data: dict[str, Any],
    sfr_data: dict[str, Any],
    description: str,
) -> None:
    """Require the flux and SFR measurements to cover the same z slab."""

    height_kpc = flux_data.get("height_kpc")
    z_bounds_kpc = sfr_data.get("z_bounds_kpc")
    if height_kpc is None or z_bounds_kpc is None:
        raise ValueError(
            f"The {description} flux JSON lacks height_kpc or the SFR JSON "
            "lacks z_bounds_kpc; cannot verify that both cover one slab."
        )
    if not np.isclose(
        float(height_kpc),
        abs(float(z_bounds_kpc[1])),
        rtol=0.0,
        atol=SLAB_TOLERANCE_KPC,
    ):
        raise ValueError(
            f"The {description} fluxes cover |z| <= {float(height_kpc):g} kpc "
            f"but the enclosed SFR covers |z| <= "
            f"{abs(float(z_bounds_kpc[1])):g} kpc; the ratio is undefined "
            "for mismatched slabs."
        )


def _check_times(
    flux_data: dict[str, Any], sfr_data: dict[str, Any]
) -> float | None:
    """Warn if the flux and SFR measurements come from different times."""

    if flux_data.get("time") is None or sfr_data.get("time") is None:
        return None
    time = float(flux_data["time"])
    if not np.isclose(time, float(sfr_data["time"]), rtol=0.0, atol=1.0):
        print(
            "warning: flux time "
            f"{time:.6g} s differs from SFR time {float(sfr_data['time']):.6g} s",
            file=wind_disk.sys.stderr,
        )
    return time


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the ratio of the net radial wall mass flux to the "
            "enclosed star formation rate versus radius at fixed |z| <= 1 kpc."
        )
    )
    parser.add_argument(
        "--flux-input",
        type=Path,
        default=Path(FLUX_JSON),
        help=(
            "Radius-scan JSON with radial wall fluxes at fixed half-height "
            f"(default: {FLUX_JSON})."
        ),
    )
    parser.add_argument(
        "--sfr-input",
        type=Path,
        default=Path(SFR_JSON),
        help=(
            "Radius-scan enclosed-SFR JSON for the same |z| slab "
            f"(default: {SFR_JSON})."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("radial_mass_flux_to_sfr.png"),
        help="Output image path (default: radial_mass_flux_to_sfr.png).",
    )
    args = parser.parse_args()

    flux_data = wind_disk._load_json(args.flux_input)
    sfr_data = wind_disk._load_json(args.sfr_input)

    _check_slab(flux_data, sfr_data, "disk")
    time = _check_times(flux_data, sfr_data)

    radii, mass_flux = wind_disk._load_mass_flux(
        flux_data, "walls", "radius", "disk"
    )
    sfr_x, sfr, _ = wind_disk._load_enclosed_sfr(sfr_data, "radii_kpc", "disk")
    sfr_at = _sfr_at_radii(sfr_x, sfr, radii, "disk")
    if np.any(sfr_at <= 0.0):
        raise ValueError("Enclosed SFR must be positive to form the ratio.")
    ratio = mass_flux / sfr_at

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.axhline(
        0.0,
        color="0.45",
        lw=1.0,
    )
    ax.axhline(
        -1.0,
        color="black",
        lw=1.0,
        ls="--",
        label=r"inflow exactly replenishes SFR ($\Phi/SFR=-1$)",
    )
    ax.plot(
        radii,
        ratio,
        color=RATIO_COLOR,
        lw=2.0,
        marker="o",
        ms=4.0,
        label=r"net wall mass flux / enclosed SFR",
    )

    # Region labels: y coordinates are in data units because the transform
    # only maps the horizontal position to axes fraction.
    ax.text(
        0.985,
        0.16,
        "net outflow",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize="small",
        color="0.35",
    )
    ax.text(
        0.985,
        -0.18,
        "net inflow",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize="small",
        color="0.35",
    )

    ax.set_xscale("log")
    ax.set_xlim(float(radii.min()) * 0.9, float(radii.max()) * 1.1)
    ax.set_ylim(float(ratio.min()) - 0.12, 0.30)
    ax.set_xlabel(wind_disk.angular_momentum.X_AXES["radius"]["xlabel"])
    ax.set_ylabel(r"$\Phi_{\rm mass}(R)\,/\,\mathrm{SFR}(<R)$")
    ax.set_title(
        wind_disk._append_time(
            "Radial gas supply vs. star formation ($|z| \\leq 1$ kpc slab)",
            wind_disk._time_myr(flux_data),
        )
    )
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize="small", loc="lower left")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
