#!/usr/bin/env python3
"""Plot the z-angular-momentum flux and mass flux, wind and disk, side by side.

The left panel shows wind transport: the vertical L_z flux through the cylinder
endcaps as a function of half-height at fixed cylinder radius, read from
``cylindrical_flux_surface.json``.  The right panel shows disk transport: the
radial L_z flux through the cylinder walls as a function of radius at fixed
half-height, read from ``cylindrical_radial_flux.json``.  Both panels share a
symlog y-scale for the L_z flux so the magnitudes are directly comparable, and
each panel overlays the measured net mass flux through the same geometric
section and the enclosed star formation rate (from deposited SSP birth-mass
density) on a secondary vertical axis in solar masses per year.  The L_z fluxes are
converted from their native cgs units (``g cm^2 s^-2``, equivalently torque
in erg) to ``M_sun kpc^2 yr^-2``; the mass flux and SFR use the derived
M_sun/yr columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import plot_cylindrical_flux_surface_angular_momentum_flux as angular_momentum


MASS_FLUX_LABEL = r"net mass flux [$M_\odot\,yr^{-1}$]"
MASS_FLUX_COLOR = "crimson"
MASS_FLUX_LINTHRESH = 0.1
SFR_COLOR = "seagreen"
WIND_SFR_JSON = "enclosed_sfr_wind.json"
DISK_SFR_JSON = "enclosed_sfr_disk.json"
# Toomre-Q profile CSV supplying the orbital frequency Omega(R) in the
# same |z| <= 1 kpc slab as the disk flux scan.
DISK_OMEGA_CSV = "toomre_q_noCGM_1uG_Phase1_EMEF/plt0230330_toomre_q.csv"
IMPLIED_COLOR = "darkorchid"
MYR_PER_YR = 1.0e-6

MSUN_G = 1.98847e33
KPC_CM = 3.0856775814913673e18 * 1.0e3
YR_S = 365.25 * 24.0 * 3600.0
# Native flux units are g cm^2 s^-2 (torque, erg); scale to M_sun kpc^2
# yr^-2.
ERG_TO_MSUN_KPC2_YR2 = YR_S**2 / (MSUN_G * KPC_CM**2)
LZ_FLUX_UNITS_LABEL = r"$M_\odot\,$kpc$^2\,$yr$^{-2}$"


def _convert_lz_flux_units(
    flux: angular_momentum.AngularMomentumFlux,
) -> angular_momentum.AngularMomentumFlux:
    """Convert L_z flux columns from native cgs to M_sun kpc^2 yr^-2."""

    return flux._replace(
        advective=flux.advective * ERG_TO_MSUN_KPC2_YR2,
        maxwell=flux.maxwell * ERG_TO_MSUN_KPC2_YR2,
        total=flux.total * ERG_TO_MSUN_KPC2_YR2,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _time_myr(data: dict[str, Any]) -> float | None:
    time = data.get("time")
    if time is None:
        return None
    return float(time) / angular_momentum.SECONDS_PER_MYR


def _append_time(title: str, time_myr: float | None) -> str:
    if time_myr is None:
        return title
    return f"{title}, t = {time_myr:.3f} Myr"


def _load_enclosed_sfr(
    data: dict[str, Any],
    x_key: str,
    description: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (x_kpc, enclosed SFR [M_sun/yr], averaging window [Myr])."""

    x = np.asarray(data.get(x_key, []), dtype=np.float64)
    sfr = np.asarray(data.get("enclosed_sfr_msun_per_yr", []), dtype=np.float64)
    if x.size == 0 or sfr.size != x.size:
        raise ValueError(
            f"The {description} SFR JSON lacks matching {x_key} and "
            "enclosed_sfr_msun_per_yr entries."
        )
    if not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError(
            f"{description.capitalize()} SFR sample positions must be finite "
            "and positive."
        )
    if not np.all(np.isfinite(sfr)) or np.any(sfr < 0.0):
        raise ValueError(f"{description.capitalize()} SFR values must be finite.")
    window_raw = data.get("sfr_window_myr")
    window = float(window_raw) if window_raw is not None else None
    order = np.argsort(x)
    return x[order], sfr[order], window


def _resolve_sfr_path(
    explicit: Path | None, default_name: str, description: str
) -> Path | None:
    path = explicit if explicit is not None else Path(default_name)
    if path.exists():
        return path
    if explicit is not None:
        raise FileNotFoundError(f"SFR input {path} not found.")
    print(
        f"note: {default_name} not found; skipping the enclosed SFR overlay "
        f"on the {description} panel",
        file=sys.stderr,
    )
    return None


def _load_orbital_frequency(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (radius_kpc, Omega [yr^-1]) from a Toomre-Q profile CSV.

    Uses only rows flagged valid with a finite, positive ``omega_myr_inv``;
    the column stores Omega in rad/Myr, converted here to rad/yr.
    """

    import csv

    radii: list[float] = []
    omega: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("valid", "")).strip().lower() not in ("true", "1"):
                continue
            radius = row.get("radius_kpc")
            omega_myr = row.get("omega_myr_inv")
            if radius is None or omega_myr is None:
                continue
            try:
                radius_value = float(radius)
                omega_value = float(omega_myr) * MYR_PER_YR
            except ValueError:
                continue
            if (
                not np.isfinite(radius_value)
                or radius_value <= 0.0
                or not np.isfinite(omega_value)
                or omega_value <= 0.0
            ):
                continue
            radii.append(radius_value)
            omega.append(omega_value)
    x = np.asarray(radii, dtype=np.float64)
    values = np.asarray(omega, dtype=np.float64)
    if x.size == 0:
        raise ValueError(f"No valid Omega(R) rows in {path}.")
    order = np.argsort(x)
    return x[order], values[order]


def _load_mass_flux(
    data: dict[str, Any],
    section: str,
    x_axis: str,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_kpc, net mass flux [M_sun/yr]) through one geometric section.

    The net flux sums the negative and positive sign bins of the requested
    section (endcaps or walls).  If per-section bins are unavailable, the
    section-summed net flux is used instead.
    """
    derived_key = (
        "mass_flux_msun_per_yr_by_height"
        if x_axis == "height"
        else "mass_flux_msun_per_yr_by_radius"
    )
    x_config = angular_momentum.X_AXES[x_axis]
    rows = data.get("derived", {}).get(derived_key)
    if rows is None:
        raise ValueError(
            f"The {description} JSON does not contain derived.{derived_key}."
        )

    x_values: list[float] = []
    net_values: list[float] = []
    for row in rows:
        bins = row.get("mass_flux_msun_per_yr_bins_by_geometric_section")
        if bins is not None:
            try:
                net = (
                    float(bins["negative"][section])
                    + float(bins["positive"][section])
                )
            except KeyError as error:
                raise ValueError(
                    f"The {description} JSON lacks per-section mass fluxes "
                    f"for the {section!r} section."
                ) from error
        else:
            net_total = row.get("mass_flux_msun_per_yr")
            if net_total is None:
                raise ValueError(
                    f"The {description} JSON lacks mass fluxes at "
                    f"{x_config['description']} "
                    f"{row.get(x_config['row_key'])!r}."
                )
            net = float(net_total)
        x_values.append(float(row[x_config["row_key"]]))
        net_values.append(net)

    x = np.asarray(x_values, dtype=np.float64)
    net = np.asarray(net_values, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError(
            f"{description.capitalize()} {x_config['description']} values must "
            "be finite and positive."
        )
    if not np.all(np.isfinite(net)):
        raise ValueError(f"{description.capitalize()} mass fluxes must be finite.")
    order = np.argsort(x)
    return x[order], net[order]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the L_z angular momentum flux and the net mass flux in the "
            "wind (vertical transport through endcaps versus half-height at "
            "fixed radius) beside the disk (radial transport through walls "
            "versus radius at fixed half-height)."
        )
    )
    parser.add_argument(
        "--wind-input",
        type=Path,
        default=Path("cylindrical_flux_surface.json"),
        help=(
            "Height-scan JSON with vertical endcap fluxes at fixed radius "
            "(default: cylindrical_flux_surface.json)."
        ),
    )
    parser.add_argument(
        "--disk-input",
        type=Path,
        default=Path("cylindrical_radial_flux.json"),
        help=(
            "Radius-scan JSON with radial wall fluxes at fixed half-height "
            "(default: cylindrical_radial_flux.json)."
        ),
    )
    parser.add_argument(
        "--wind-sfr-input",
        type=Path,
        default=None,
        help=(
            "Height-scan enclosed-SFR JSON for the wind panel "
            f"(default: {WIND_SFR_JSON} when present)."
        ),
    )
    parser.add_argument(
        "--disk-sfr-input",
        type=Path,
        default=None,
        help=(
            "Radius-scan enclosed-SFR JSON for the disk panel "
            f"(default: {DISK_SFR_JSON} when present)."
        ),
    )
    parser.add_argument(
        "--disk-omega-csv",
        type=Path,
        default=None,
        help=(
            "Toomre-Q profile CSV with the orbital frequency Omega(R) for "
            "the implied steady-state mass flux on the disk panel "
            f"(default: {DISK_OMEGA_CSV} when present)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("angular_momentum_flux_wind_vs_disk.png"),
        help="Output image path (default: angular_momentum_flux_wind_vs_disk.png).",
    )
    args = parser.parse_args()

    wind_data = _load_json(args.wind_input)
    disk_data = _load_json(args.disk_input)

    # Wind: vertical L_z transport and mass loading through the endcaps,
    # versus half-height, at fixed cylinder radius.
    wind = _convert_lz_flux_units(
        angular_momentum._load_flux(wind_data, "vertical", "height")
    )
    wind_mass_x, wind_mass = _load_mass_flux(
        wind_data, "endcaps", "height", "wind"
    )
    wind_sfr_x = wind_sfr = None
    wind_sfr_window = None
    wind_sfr_path = _resolve_sfr_path(args.wind_sfr_input, WIND_SFR_JSON, "wind")
    if wind_sfr_path is not None:
        wind_sfr_x, wind_sfr, wind_sfr_window = _load_enclosed_sfr(
            _load_json(wind_sfr_path), "heights_kpc", "wind"
        )
    # Disk: radial L_z transport and mass flux through the walls versus
    # radius, at fixed half-height.
    disk = _convert_lz_flux_units(
        angular_momentum._load_flux(disk_data, "radial", "radius")
    )
    disk_mass_x, disk_mass = _load_mass_flux(disk_data, "walls", "radius", "disk")
    disk_sfr_x = disk_sfr = None
    disk_sfr_window = None
    disk_sfr_path = _resolve_sfr_path(args.disk_sfr_input, DISK_SFR_JSON, "disk")
    if disk_sfr_path is not None:
        disk_sfr_x, disk_sfr, disk_sfr_window = _load_enclosed_sfr(
            _load_json(disk_sfr_path), "radii_kpc", "disk"
        )

    # Implied steady-state mass flux on the disk panel: the viscous
    # accretion relation Mdot = Phi_Lz * (2*pi / Omega(R)) / A(R) with
    # annular wall area A(R) = 4*pi*R*H, i.e. Mdot = Phi_Lz / (2*Omega*R*H).
    disk_implied_x = disk_implied = None
    omega_path = args.disk_omega_csv
    if omega_path is None:
        candidate = Path(DISK_OMEGA_CSV)
        if candidate.exists():
            omega_path = candidate
    elif not omega_path.exists():
        raise FileNotFoundError(f"Omega input {omega_path} not found.")
    if omega_path is not None:
        omega_x, omega_yr = _load_orbital_frequency(omega_path)
        disk_height_kpc = disk_data.get("height_kpc")
        if disk_height_kpc is None or float(disk_height_kpc) <= 0.0:
            raise ValueError(
                "The disk JSON lacks a positive height_kpc; the implied "
                "steady-state mass flux needs the slab half-height H."
            )
        inside = (disk.x_kpc >= omega_x[0]) & (disk.x_kpc <= omega_x[-1])
        radius_kpc = disk.x_kpc[inside]
        omega_at = np.interp(
            np.log(radius_kpc), np.log(omega_x), omega_yr
        )
        disk_implied_x = radius_kpc
        disk_implied = disk.total[inside] * (2.0 * np.pi / omega_at) / (
            4.0 * np.pi * radius_kpc * float(disk_height_kpc)
        )

    largest_lz = max(
        float(np.max(np.abs(flux.total))) for flux in (wind, disk)
    )
    lz_linthresh = largest_lz * 1.0e-4 if largest_lz > 0.0 else 1.0
    lz_ylim = (-largest_lz * 1.05, largest_lz * 1.05)

    largest_mass = max(
        float(np.max(np.abs(wind_mass))),
        float(np.max(np.abs(disk_mass))),
    )
    if wind_sfr is not None:
        largest_mass = max(largest_mass, float(np.max(wind_sfr)))
    if disk_sfr is not None:
        largest_mass = max(largest_mass, float(np.max(disk_sfr)))
    if disk_implied is not None:
        largest_mass = max(largest_mass, float(np.max(np.abs(disk_implied))))
    mass_ylim = (-largest_mass * 1.05, largest_mass * 1.05)

    wind_radius = wind_data.get("radius_kpc")
    wind_title = "Wind: endcap $L_z$ and mass flux"
    if wind_radius is not None:
        wind_title += f", R = {float(wind_radius):g} kpc"
    disk_height = disk_data.get("height_kpc")
    disk_title = "Disk: wall $L_z$ and mass flux"
    if disk_height is not None:
        disk_title += f", |z| <= {float(disk_height):g} kpc"

    wind_time = _time_myr(wind_data)
    disk_time = _time_myr(disk_data)
    suptitle = r"$L_z$ angular momentum flux: wind vs disk"
    if wind_time is not None and wind_time == disk_time:
        suptitle = _append_time(suptitle, wind_time)
    else:
        wind_title = _append_time(wind_title, wind_time)
        disk_title = _append_time(disk_title, disk_time)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.5), squeeze=False)
    panels = (
        (
            axes[0][0],
            wind,
            wind_mass_x,
            wind_mass,
            wind_sfr_x,
            wind_sfr,
            wind_sfr_window,
            None,
            None,
            wind_title,
            "half-height [kpc]",
        ),
        (
            axes[0][1],
            disk,
            disk_mass_x,
            disk_mass,
            disk_sfr_x,
            disk_sfr,
            disk_sfr_window,
            disk_implied_x,
            disk_implied,
            disk_title,
            "radius [kpc]",
        ),
    )
    for (
        ax,
        flux,
        mass_x,
        mass,
        sfr_x,
        sfr,
        sfr_window,
        implied_x,
        implied,
        title,
        xlabel,
    ) in panels:
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        ax.plot(flux.x_kpc, flux.total, "o--", label="total $L_z$")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=lz_linthresh)
        ax.set_ylim(lz_ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            r"$L_z$ angular momentum flux [" + LZ_FLUX_UNITS_LABEL + "]"
        )
        ax.set_title(title, fontsize="medium")
        ax.grid(True, which="both", alpha=0.25)

        mass_ax = ax.twinx()
        mass_ax.plot(
            mass_x,
            mass,
            "s-",
            color=MASS_FLUX_COLOR,
            markersize=3.5,
            linewidth=1.6,
            label="net mass flux",
        )
        if sfr is not None and sfr_x is not None:
            window_label = (
                f" ({sfr_window:g} Myr)" if sfr_window is not None else ""
            )
            mass_ax.plot(
                sfr_x,
                sfr,
                "d--",
                color=SFR_COLOR,
                markersize=3.5,
                linewidth=1.6,
                label=r"enclosed SFR [$M_\odot\,yr^{-1}$]"
                + window_label,
            )
        if implied is not None and implied_x is not None:
            mass_ax.plot(
                implied_x,
                implied,
                "v-.",
                color=IMPLIED_COLOR,
                markersize=3.5,
                linewidth=1.6,
                label=(
                    r"implied steady-state $\dot M$ "
                    r"[$\Phi_{L_z}\,(2\pi/\Omega)/(4\pi RH)$]"
                ),
            )
        mass_ax.set_yscale("symlog", linthresh=MASS_FLUX_LINTHRESH)
        mass_ax.set_ylim(mass_ylim)
        mass_ax.set_ylabel(MASS_FLUX_LABEL, color=MASS_FLUX_COLOR)
        mass_ax.tick_params(axis="y", labelcolor=MASS_FLUX_COLOR)

        handles, labels = ax.get_legend_handles_labels()
        mass_handles, mass_labels = mass_ax.get_legend_handles_labels()
        ax.legend(
            handles + mass_handles, labels + mass_labels, fontsize="small"
        )
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
