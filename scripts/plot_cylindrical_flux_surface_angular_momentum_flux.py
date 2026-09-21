#!/usr/bin/env python3
"""Plot the z-angular-momentum flux through cylindrical surfaces.

Radial transport is read from the cylinder walls and vertical transport from
the two endcaps.  The JSON values are plotted in their native cgs units,
``g cm^2 s^-2`` (equivalently, torque in erg).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np


SECONDS_PER_MYR = 1.0e6 * 365.25 * 24.0 * 3600.0
FLUX_UNITS_LABEL = r"g cm$^2$ s$^{-2}$"
DIRECTIONS = {
    "radial": {
        "section": "walls",
        "derived_key": "radial_angular_momentum_flux_by_height",
        "advective_key": "advective_radial_angular_momentum_flux_cylinder",
        "maxwell_key": "maxwell_radial_angular_momentum_flux_cylinder",
        "title": "Radial $L_z$ flux through walls",
    },
    "vertical": {
        "section": "endcaps",
        "derived_key": "vertical_angular_momentum_flux_by_height",
        "advective_key": "advective_vertical_angular_momentum_flux_cylinder",
        "maxwell_key": "maxwell_vertical_angular_momentum_flux_cylinder",
        "title": "Vertical $L_z$ flux through endcaps",
    },
}


class AngularMomentumFlux(NamedTuple):
    height_kpc: np.ndarray
    advective: np.ndarray
    maxwell: np.ndarray
    total: np.ndarray


def _as_finite_array(values: list[float], description: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0:
        raise ValueError(f"No {description} samples found.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{description.capitalize()} values must be finite.")
    return result


def _load_from_derived(
    data: dict[str, Any], direction: str
) -> AngularMomentumFlux | None:
    config = DIRECTIONS[direction]
    rows = data.get("derived", {}).get(config["derived_key"])
    if rows is None:
        return None
    if not rows:
        raise ValueError(f"No {direction} angular-momentum-flux samples found.")

    section = config["section"]
    height = _as_finite_array(
        [row["height_kpc"] for row in rows], "half-height"
    )
    section_rows = [row["by_geometric_section"][section] for row in rows]
    advective = _as_finite_array(
        [row["advective"] for row in section_rows], "advective flux"
    )
    maxwell = _as_finite_array(
        [row["maxwell"] for row in section_rows], "Maxwell flux"
    )
    total = _as_finite_array(
        [row.get("total", row["advective"] + row["maxwell"]) for row in section_rows],
        "total flux",
    )
    return _sort_and_validate(height, advective, maxwell, total)


def _load_from_flux_rows(
    data: dict[str, Any], direction: str
) -> AngularMomentumFlux:
    config = DIRECTIONS[direction]
    rows = data.get("fluxes_by_height")
    if rows is None:
        fluxes = data.get("fluxes")
        if fluxes is None or data.get("height_kpc") is None:
            raise ValueError(
                "JSON must contain angular-momentum fluxes in derived profiles "
                "or in fluxes_by_height."
            )
        rows = [
            {
                "height_kpc": data["height_kpc"],
                "fluxes": fluxes,
                "flux_bins_by_geometric_section": data.get(
                    "flux_bins_by_geometric_section"
                ),
            }
        ]
    if not rows:
        raise ValueError("No height samples found.")

    section = config["section"]

    def component(row: dict[str, Any], key: str) -> float:
        try:
            bins = row.get("flux_bins_by_geometric_section")
            if bins is not None:
                return sum(
                    float(bins[sign][section][key])
                    for sign in ("negative", "positive")
                )
            # Older wall-only JSON stored only section-summed fluxes.  It can
            # be used for radial transport, but cannot represent the endcaps.
            if section == "walls":
                return float(row["fluxes"][key])
        except KeyError as error:
            raise ValueError(
                f"JSON does not contain the required flux field {key!r}; "
                "regenerate it with the current cylindrical flux-surface script."
            ) from error
        raise ValueError(
            "JSON does not contain geometric-section fluxes needed for "
            "vertical angular-momentum transport."
        )

    height = _as_finite_array(
        [row["height_kpc"] for row in rows], "half-height"
    )
    advective = _as_finite_array(
        [component(row, config["advective_key"]) for row in rows],
        "advective flux",
    )
    maxwell = _as_finite_array(
        [component(row, config["maxwell_key"]) for row in rows], "Maxwell flux"
    )
    return _sort_and_validate(height, advective, maxwell, advective + maxwell)


def _sort_and_validate(
    height: np.ndarray,
    advective: np.ndarray,
    maxwell: np.ndarray,
    total: np.ndarray,
) -> AngularMomentumFlux:
    if not (height.shape == advective.shape == maxwell.shape == total.shape):
        raise ValueError(
            "Height and angular-momentum-flux arrays have different lengths."
        )
    if np.any(height <= 0.0):
        raise ValueError("Half-height values must be positive for a log-scaled plot.")
    order = np.argsort(height)
    return AngularMomentumFlux(
        height[order], advective[order], maxwell[order], total[order]
    )


def _load_flux(data: dict[str, Any], direction: str) -> AngularMomentumFlux:
    derived = _load_from_derived(data, direction)
    if derived is not None:
        return derived
    return _load_from_flux_rows(data, direction)


def _plot_direction(
    ax: plt.Axes,
    flux: AngularMomentumFlux,
    direction: str,
    linear_y: bool,
) -> None:
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.plot(flux.height_kpc, flux.advective, "o-", label="advective")
    ax.plot(flux.height_kpc, flux.maxwell, "o-", label="Maxwell")
    ax.plot(flux.height_kpc, flux.total, "o--", label="total")
    ax.set_xscale("log")
    if not linear_y:
        largest = max(
            float(np.max(np.abs(flux.advective))),
            float(np.max(np.abs(flux.maxwell))),
            float(np.max(np.abs(flux.total))),
        )
        ax.set_yscale("symlog", linthresh=max(largest * 1.0e-4, 1.0))
    ax.set_xlabel("half-height [kpc]")
    ax.set_ylabel(r"$L_z$ angular momentum flux [" + FLUX_UNITS_LABEL + "]")
    ax.set_title(DIRECTIONS[direction]["title"])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize="small")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the L_z angular momentum flux versus half-height from "
            "cylindrical_flux_surface.json."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="cylindrical_flux_surface.json",
        help="Input JSON file (default: cylindrical_flux_surface.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="angular_momentum_flux_vs_height.png",
        help="Output image path (default: angular_momentum_flux_vs_height.png).",
    )
    parser.add_argument(
        "--direction",
        choices=("radial", "vertical", "both"),
        default="both",
        help="Transport direction to plot (default: both).",
    )
    parser.add_argument(
        "--title",
        default=r"$L_z$ flux through cylindrical surfaces",
        help="Figure title.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of symlog.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    directions = list(DIRECTIONS) if args.direction == "both" else [args.direction]
    fluxes = {direction: _load_flux(data, direction) for direction in directions}

    fig, axes = plt.subplots(
        1,
        len(directions),
        figsize=(7.0 * len(directions), 4.5),
        squeeze=False,
    )
    for ax, direction in zip(axes[0], directions):
        _plot_direction(ax, fluxes[direction], direction, args.linear_y)

    title = args.title
    time = data.get("time")
    if time is not None:
        title += f", t = {float(time) / SECONDS_PER_MYR:.3f} Myr"
    fig.suptitle(title)
    fig.tight_layout()
    output = Path(args.output)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
