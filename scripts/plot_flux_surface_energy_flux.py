#!/usr/bin/env python3
"""Plot energy fluxes as a function of radius from flux_surface.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np


ENERGY_COMPONENTS = {
    "hydro": ("hydro_energy_flux_sphere", "hydro energy flux"),
    "mhd": ("mhd_energy_flux_sphere", "MHD energy flux"),
}
LSUN_ERG_PER_S = 3.828e33
LSUN_LABEL = r"$L_\odot$"
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_MYR = 1.0e6 * SECONDS_PER_YEAR


class TemperatureEnergyFlux(NamedTuple):
    labels: list[str]
    negative: np.ndarray
    positive: np.ndarray


class EnergyFluxData(NamedTuple):
    radius_kpc: np.ndarray
    net: np.ndarray
    negative: np.ndarray
    positive: np.ndarray
    time_myr: float | None
    component_label: str
    temperature: TemperatureEnergyFlux | None


class AxisLimits(NamedTuple):
    xlim: tuple[float, float]
    ylim: tuple[float, float]


def _temperature_label(row: dict[str, Any]) -> str:
    t_min = row.get("temperature_min")
    t_max = row.get("temperature_max")
    if t_min is None or t_max is None:
        return "temperature bin"
    return f"{float(t_min):g}-{float(t_max):g} K"


def _to_lsun(value: float) -> float:
    return float(value) / LSUN_ERG_PER_S


def _load_temperature_energy_flux(
    rows: list[dict[str, Any]],
    order: np.ndarray,
    flux_key: str,
) -> TemperatureEnergyFlux | None:
    first_bins = rows[0].get("flux_bins_by_temperature")
    if first_bins is None:
        return None
    if not isinstance(first_bins, dict):
        raise ValueError("Temperature flux bins must be an object.")

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
        temp_bins = row.get("flux_bins_by_temperature")
        if temp_bins is None:
            raise ValueError("Temperature-bin flux is missing from some radii.")
        negative = temp_bins.get("negative")
        positive = temp_bins.get("positive")
        if negative is None or positive is None:
            raise ValueError("Temperature-bin flux requires negative and positive bins.")
        if len(negative) != len(labels) or len(positive) != len(labels):
            raise ValueError("Temperature-bin counts differ between radii.")
        negative_rows.append(
            [_to_lsun(item.get("fluxes", {}).get(flux_key, 0.0)) for item in negative]
        )
        positive_rows.append(
            [_to_lsun(item.get("fluxes", {}).get(flux_key, 0.0)) for item in positive]
        )

    negative_by_radius = np.asarray(negative_rows, dtype=np.float64)
    positive_by_radius = np.asarray(positive_rows, dtype=np.float64)
    if not (
        np.all(np.isfinite(negative_by_radius))
        and np.all(np.isfinite(positive_by_radius))
    ):
        raise ValueError("Temperature-bin energy-flux values must be finite.")

    return TemperatureEnergyFlux(
        labels=labels,
        negative=negative_by_radius[order].T,
        positive=positive_by_radius[order].T,
    )


def _flux_rows_from_single_height(data: dict[str, Any]) -> list[dict[str, Any]]:
    single_fluxes = data.get("fluxes")
    if single_fluxes is None:
        raise ValueError("JSON must contain fluxes_by_radius or fluxes.")
    radius_kpc = data.get("radius_kpc")
    if radius_kpc is None:
        raise ValueError("Single-radius JSON must contain radius_kpc.")
    return [
        {
            "radius_kpc": radius_kpc,
            "fluxes": single_fluxes,
            "flux_bins": data.get("flux_bins"),
            "flux_bins_by_temperature": data.get("flux_bins_by_temperature"),
        }
    ]


def _load_energy_flux(path: Path, component: str) -> EnergyFluxData:
    flux_key, component_label = ENERGY_COMPONENTS[component]
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    time = data.get("time")
    time_myr = None if time is None else float(time) / SECONDS_PER_MYR

    flux_rows = data.get("fluxes_by_radius")
    if flux_rows is None:
        flux_rows = _flux_rows_from_single_height(data)
    if not flux_rows:
        raise ValueError("No radius samples found.")

    radius = np.asarray([row["radius_kpc"] for row in flux_rows], dtype=np.float64)
    net = np.asarray(
        [_to_lsun(row["fluxes"][flux_key]) for row in flux_rows],
        dtype=np.float64,
    )
    negative = np.asarray(
        [
            _to_lsun(
                row.get("flux_bins", {})
                .get("negative", {})
                .get(flux_key, min(row["fluxes"][flux_key], 0.0))
            )
            for row in flux_rows
        ],
        dtype=np.float64,
    )
    positive = np.asarray(
        [
            _to_lsun(
                row.get("flux_bins", {})
                .get("positive", {})
                .get(flux_key, max(row["fluxes"][flux_key], 0.0))
            )
            for row in flux_rows
        ],
        dtype=np.float64,
    )

    if radius.size == 0:
        raise ValueError("No radius samples found.")
    if not (radius.shape == net.shape == negative.shape == positive.shape):
        raise ValueError("Radius and energy-flux arrays have different lengths.")
    if not (
        np.all(np.isfinite(radius))
        and np.all(np.isfinite(net))
        and np.all(np.isfinite(negative))
        and np.all(np.isfinite(positive))
    ):
        raise ValueError("Radius and energy-flux values must be finite.")

    order = np.argsort(radius)
    return EnergyFluxData(
        radius_kpc=radius[order],
        net=net[order],
        negative=negative[order],
        positive=positive[order],
        time_myr=time_myr,
        component_label=component_label,
        temperature=_load_temperature_energy_flux(flux_rows, order, flux_key),
    )


def _output_with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _setup_axes(
    ax: plt.Axes,
    title: str,
    component_label: str,
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
    ax.set_ylabel(f"{component_label} [{LSUN_LABEL}]")
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, which="both", alpha=0.25)


def _plot_lines(
    output: Path,
    data: EnergyFluxData,
    lines: list[tuple[np.ndarray, str, str]],
    title: str,
    linear_y: bool,
    axis_limits: AxisLimits | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for values, label, linestyle in lines:
        ax.plot(
            data.radius_kpc,
            values,
            marker="o",
            linewidth=1.6,
            markersize=4,
            linestyle=linestyle,
            label=label,
        )
    _setup_axes(ax, title, data.component_label, linear_y, axis_limits)
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


def _plot_energy_flux_set(
    data: EnergyFluxData,
    output: Path,
    title: str,
    linear_y: bool,
) -> list[Path]:
    if data.temperature is None:
        axis_limits = _compute_axis_limits(
            data.radius_kpc,
            [data.positive, data.negative, data.net],
        )
        _plot_lines(
            output,
            data,
            [
                (data.negative, "negative", "-"),
                (data.positive, "positive", "-"),
                (data.net, "net", "--"),
            ],
            title,
            linear_y,
            axis_limits,
        )
        return [output]

    temp = data.temperature
    net_by_temperature = temp.negative + temp.positive
    total_negative = np.sum(temp.negative, axis=0)
    total_positive = np.sum(temp.positive, axis=0)
    total_net = total_negative + total_positive
    total_axis_limits = _compute_axis_limits(
        data.radius_kpc,
        [total_negative, total_positive, total_net],
    )
    axis_limits = _compute_axis_limits(
        data.radius_kpc,
        [
            temp.negative,
            temp.positive,
            net_by_temperature,
        ],
    )
    outputs: list[Path] = []

    _plot_lines(
        output,
        data,
        [
            (total_negative, "negative", "-"),
            (total_positive, "positive", "-"),
            (total_net, "net", "--"),
        ],
        f"{title}: all temperature phases",
        linear_y,
        total_axis_limits,
    )
    outputs.append(output)

    negative_output = _output_with_suffix(output, "negative_by_temperature")
    _plot_lines(
        negative_output,
        data,
        [(temp.negative[i], temp.labels[i], "-") for i in range(len(temp.labels))],
        f"{title}: negative flux by temperature",
        linear_y,
        axis_limits,
    )
    outputs.append(negative_output)

    positive_output = _output_with_suffix(output, "positive_by_temperature")
    _plot_lines(
        positive_output,
        data,
        [(temp.positive[i], temp.labels[i], "-") for i in range(len(temp.labels))],
        f"{title}: positive flux by temperature",
        linear_y,
        axis_limits,
    )
    outputs.append(positive_output)

    net_output = _output_with_suffix(output, "net_by_temperature")
    _plot_lines(
        net_output,
        data,
        [(net_by_temperature[i], temp.labels[i], "-") for i in range(len(temp.labels))],
        f"{title}: net flux by temperature",
        linear_y,
        axis_limits,
    )
    outputs.append(net_output)

    for i, label in enumerate(temp.labels):
        bin_output = _output_with_suffix(output, f"temperature_bin_{i:02d}")
        _plot_lines(
            bin_output,
            data,
            [
                (temp.negative[i], "negative", "-"),
                (temp.positive[i], "positive", "-"),
                (net_by_temperature[i], "net", "--"),
            ],
            f"{title}: {label}",
            linear_y,
            axis_limits,
        )
        outputs.append(bin_output)

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot energy flux versus radius from a flux_surface.json file."
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
        default="energy_flux_vs_radius.png",
        help="Output image path (default: energy_flux_vs_radius.png).",
    )
    parser.add_argument(
        "--component",
        choices=("hydro", "mhd", "both"),
        default="both",
        help="Energy flux component to plot (default: both).",
    )
    parser.add_argument(
        "--title",
        default="Energy flux through spherical surfaces",
        help="Plot title.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of symlog.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    components = (
        list(ENERGY_COMPONENTS)
        if args.component == "both"
        else [args.component]
    )
    outputs: list[Path] = []
    for component in components:
        data = _load_energy_flux(Path(args.input), component)
        title = f"{args.title}: {data.component_label}"
        if data.time_myr is not None:
            title = f"{title}, t = {data.time_myr:.3f} Myr"
        component_output = (
            _output_with_suffix(output, component)
            if args.component == "both"
            else output
        )
        outputs.extend(
            _plot_energy_flux_set(
                data,
                component_output,
                title,
                args.linear_y,
            )
        )

    for path in outputs:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
