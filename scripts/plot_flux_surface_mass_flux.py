#!/usr/bin/env python3
"""Plot mass flux as a function of radius from flux_surface.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np


MSUN_PER_YEAR_LABEL = r"$M_\odot\,yr^{-1}$"
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_MYR = 1.0e6 * SECONDS_PER_YEAR


class TemperatureMassFlux(NamedTuple):
    labels: list[str]
    negative: np.ndarray
    positive: np.ndarray


class MassFluxData(NamedTuple):
    radius_kpc: np.ndarray
    net: np.ndarray
    negative: np.ndarray
    positive: np.ndarray
    time_myr: float | None
    temperature: TemperatureMassFlux | None


class AxisLimits(NamedTuple):
    xlim: tuple[float, float]
    ylim: tuple[float, float]


def _temperature_label(row: dict[str, Any]) -> str:
    t_min = row.get("temperature_min")
    t_max = row.get("temperature_max")
    if t_min is None or t_max is None:
        return "temperature bin"
    return f"{float(t_min):g}-{float(t_max):g} K"


def _load_temperature_mass_flux(
    rows: list[dict[str, Any]],
    order: np.ndarray,
) -> TemperatureMassFlux | None:
    first_bins = rows[0].get("mass_flux_msun_per_yr_bins_by_temperature")
    if first_bins is None:
        return None
    if not isinstance(first_bins, dict):
        raise ValueError("Temperature mass-flux bins must be an object.")

    first_negative = first_bins.get("negative")
    first_positive = first_bins.get("positive")
    if not first_negative or not first_positive:
        return None
    if len(first_negative) != len(first_positive):
        raise ValueError("Negative and positive temperature-bin counts differ.")

    labels = [_temperature_label(row) for row in first_negative]
    negative_rows = []
    positive_rows = []
    for row in rows:
        temp_bins = row.get("mass_flux_msun_per_yr_bins_by_temperature")
        if temp_bins is None:
            raise ValueError("Temperature-bin mass flux is missing from some radii.")
        negative = temp_bins.get("negative")
        positive = temp_bins.get("positive")
        if negative is None or positive is None:
            raise ValueError("Temperature-bin mass flux requires negative and positive bins.")
        if len(negative) != len(labels) or len(positive) != len(labels):
            raise ValueError("Temperature-bin counts differ between radii.")
        negative_rows.append(
            [float(item["mass_flux_msun_per_yr"]) for item in negative]
        )
        positive_rows.append(
            [float(item["mass_flux_msun_per_yr"]) for item in positive]
        )

    negative_by_radius = np.asarray(negative_rows, dtype=np.float64)
    positive_by_radius = np.asarray(positive_rows, dtype=np.float64)
    if not (
        np.all(np.isfinite(negative_by_radius))
        and np.all(np.isfinite(positive_by_radius))
    ):
        raise ValueError("Temperature-bin mass-flux values must be finite.")

    return TemperatureMassFlux(
        labels=labels,
        negative=negative_by_radius[order].T,
        positive=positive_by_radius[order].T,
    )


def _load_mass_flux(
    path: Path,
) -> MassFluxData:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    time = data.get("time")
    time_myr = None if time is None else float(time) / SECONDS_PER_MYR

    rows = data.get("derived", {}).get("mass_flux_msun_per_yr_by_radius")
    if rows is not None:
        if not rows:
            raise ValueError("No radius samples found.")
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
        order = np.argsort(radius)
        temperature = _load_temperature_mass_flux(rows, order)
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
        temperature = None

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
    return MassFluxData(
        radius_kpc=radius[order],
        net=mass_flux[order],
        negative=mass_flux_negative[order],
        positive=mass_flux_positive[order],
        time_myr=time_myr,
        temperature=temperature,
    )


def _output_with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _setup_axes(
    ax: plt.Axes,
    title: str,
    linear_y: bool,
    axis_limits: AxisLimits | None,
) -> None:
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xscale("log")
    if not linear_y:
        ax.set_yscale("symlog", linthresh=0.1)
    if axis_limits is not None:
        ax.set_xlim(axis_limits.xlim)
        ax.set_ylim(axis_limits.ylim)
    ax.set_xlabel("radius [kpc]")
    ax.set_ylabel(f"mass flux [{MSUN_PER_YEAR_LABEL}]")
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, which="both", alpha=0.25)


def _plot_lines(
    output: Path,
    radius_kpc: np.ndarray,
    lines: list[tuple[np.ndarray, str, str]],
    title: str,
    linear_y: bool,
    axis_limits: AxisLimits | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for values, label, linestyle in lines:
        ax.plot(
            radius_kpc,
            values,
            marker="o",
            linewidth=1.6,
            markersize=4,
            linestyle=linestyle,
            label=label,
        )
    _setup_axes(ax, title, linear_y, axis_limits)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _compute_axis_limits(
    radius_kpc: np.ndarray,
    y_values: list[np.ndarray],
) -> AxisLimits:
    radius = np.asarray(radius_kpc, dtype=np.float64)
    if np.any(radius <= 0.0):
        raise ValueError("Radius values must be positive for log-scaled plots.")
    x_min = float(np.min(radius))
    x_max = float(np.max(radius))
    if x_min == x_max:
        xlim = (x_min / 1.05, x_max * 1.05)
    else:
        x_pad = 10.0 ** (0.03 * np.log10(x_max / x_min))
        xlim = (x_min / x_pad, x_max * x_pad)

    y = np.concatenate([np.ravel(values) for values in y_values])
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 1.0)
    else:
        padding = 0.05 * (y_max - y_min)
    ylim = (y_min - padding, y_max + padding)
    if ylim[0] <= 0.0 <= ylim[1]:
        max_abs = max(abs(ylim[0]), abs(ylim[1]))
        ylim = (-max_abs, max_abs)
    return AxisLimits(xlim=xlim, ylim=ylim)


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

    data = _load_mass_flux(Path(args.input))
    title = args.title
    if data.time_myr is not None:
        title = f"{title}, t = {data.time_myr:.3f} Myr"

    output = Path(args.output)
    if data.temperature is None:
        axis_limits = _compute_axis_limits(
            data.radius_kpc,
            [data.positive, data.negative, data.net],
        )
        _plot_lines(
            output,
            data.radius_kpc,
            [
                (data.positive, "positive", "-"),
                (data.negative, "negative", "-"),
                (data.net, "net", "--"),
            ],
            title,
            args.linear_y,
            axis_limits,
        )
        print(f"wrote {output}")
        return 0

    temp = data.temperature
    net_by_temperature = temp.negative + temp.positive
    axis_limits = _compute_axis_limits(
        data.radius_kpc,
        [
            temp.negative,
            temp.positive,
            net_by_temperature,
        ],
    )
    outputs: list[Path] = []

    inflows_output = _output_with_suffix(output, "inflows_by_temperature")
    _plot_lines(
        inflows_output,
        data.radius_kpc,
        [
            (temp.negative[i], temp.labels[i], "-")
            for i in range(len(temp.labels))
        ],
        f"{title}: inflows by temperature",
        args.linear_y,
        axis_limits,
    )
    outputs.append(inflows_output)

    outflows_output = _output_with_suffix(output, "outflows_by_temperature")
    _plot_lines(
        outflows_output,
        data.radius_kpc,
        [
            (temp.positive[i], temp.labels[i], "-")
            for i in range(len(temp.labels))
        ],
        f"{title}: outflows by temperature",
        args.linear_y,
        axis_limits,
    )
    outputs.append(outflows_output)

    net_output = _output_with_suffix(output, "net_by_temperature")
    _plot_lines(
        net_output,
        data.radius_kpc,
        [
            (net_by_temperature[i], temp.labels[i], "-")
            for i in range(len(temp.labels))
        ],
        f"{title}: net flux by temperature",
        args.linear_y,
        axis_limits,
    )
    outputs.append(net_output)

    for i, label in enumerate(temp.labels):
        bin_output = _output_with_suffix(output, f"temperature_bin_{i:02d}")
        _plot_lines(
            bin_output,
            data.radius_kpc,
            [
                (temp.negative[i], "inflows", "-"),
                (temp.positive[i], "outflows", "-"),
                (net_by_temperature[i], "net", "--"),
            ],
            f"{title}: {label}",
            args.linear_y,
            axis_limits,
        )
        outputs.append(bin_output)

    for path in outputs:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
