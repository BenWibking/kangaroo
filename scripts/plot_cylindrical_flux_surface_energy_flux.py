#!/usr/bin/env python3
"""Plot cylindrical energy fluxes as a function of half-height."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .plot_signed_flux import SIGN_LEGEND, fit_plot_title, flux_y_limits, plot_signed_flux
else:
    from plot_signed_flux import SIGN_LEGEND, fit_plot_title, flux_y_limits, plot_signed_flux


ENERGY_COMPONENTS = {
    "hydro": ("hydro_energy_flux_cylinder", "hydro energy flux"),
    "mhd": ("mhd_energy_flux_cylinder", "MHD energy flux"),
    "kinetic": ("kinetic_energy_flux_cylinder", "kinetic energy flux"),
    "enthalpy": ("thermal_enthalpy_flux_cylinder", "thermal enthalpy flux"),
    "poynting": ("poynting_flux_cylinder", "Poynting flux"),
}
ENERGY_BREAKDOWN = {
    "hydro": ("kinetic", "enthalpy"),
    "mhd": ("kinetic", "enthalpy", "poynting"),
}
GEOMETRIC_SECTIONS = ("walls", "endcaps")
LSUN_ERG_PER_S = 3.828e33
LSUN_LABEL = r"$L_\odot$"
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_MYR = 1.0e6 * SECONDS_PER_YEAR


class TemperatureEnergyFlux(NamedTuple):
    labels: list[str]
    negative: np.ndarray
    positive: np.ndarray


class EnergyFluxData(NamedTuple):
    height_kpc: np.ndarray
    net: np.ndarray
    negative: np.ndarray
    positive: np.ndarray
    time_myr: float | None
    component_label: str
    temperature: TemperatureEnergyFlux | None


class AxisLimits(NamedTuple):
    xlim: tuple[float, float]
    ylim: tuple[float, float] | None


def _temperature_label(row: dict[str, Any]) -> str:
    t_min = row.get("temperature_min")
    t_max = row.get("temperature_max")
    if t_min is None or t_max is None:
        return "temperature bin"
    return f"{float(t_min):g}-{float(t_max):g} K"


def _section_label(section: str) -> str:
    if section == "walls":
        return "walls"
    if section == "endcaps":
        return "endcaps"
    raise ValueError(f"unsupported geometric section: {section}")


def _to_lsun(value: float) -> float:
    return float(value) / LSUN_ERG_PER_S


def _flux_from_item(item: dict[str, Any], section: str, flux_key: str) -> float:
    fluxes_by_section = item.get("fluxes_by_geometric_section")
    if fluxes_by_section is not None:
        return _to_lsun(fluxes_by_section.get(section, {}).get(flux_key, 0.0))
    if section == "walls":
        return _to_lsun(item.get("fluxes", {}).get(flux_key, 0.0))
    return 0.0


def _load_temperature_energy_flux(
    rows: list[dict[str, Any]],
    order: np.ndarray,
    section: str,
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
            raise ValueError("Temperature-bin flux is missing from some heights.")
        negative = temp_bins.get("negative")
        positive = temp_bins.get("positive")
        if negative is None or positive is None:
            raise ValueError("Temperature-bin flux requires negative and positive bins.")
        if len(negative) != len(labels) or len(positive) != len(labels):
            raise ValueError("Temperature-bin counts differ between heights.")
        negative_rows.append(
            [_flux_from_item(item, section, flux_key) for item in negative]
        )
        positive_rows.append(
            [_flux_from_item(item, section, flux_key) for item in positive]
        )

    negative_by_height = np.asarray(negative_rows, dtype=np.float64)
    positive_by_height = np.asarray(positive_rows, dtype=np.float64)
    if not (
        np.all(np.isfinite(negative_by_height))
        and np.all(np.isfinite(positive_by_height))
    ):
        raise ValueError("Temperature-bin energy-flux values must be finite.")

    return TemperatureEnergyFlux(
        labels=labels,
        negative=negative_by_height[order].T,
        positive=positive_by_height[order].T,
    )


def _flux_rows_from_single_height(data: dict[str, Any]) -> list[dict[str, Any]]:
    single_fluxes = data.get("fluxes")
    if single_fluxes is None:
        raise ValueError("JSON must contain fluxes_by_height or fluxes.")
    height_kpc = data.get("height_kpc")
    if height_kpc is None:
        raise ValueError("Single-height JSON must contain height_kpc.")
    return [
        {
            "height_kpc": height_kpc,
            "fluxes": single_fluxes,
            "flux_bins": data.get("flux_bins"),
            "flux_bins_by_geometric_section": data.get(
                "flux_bins_by_geometric_section"
            ),
            "flux_bins_by_temperature": data.get("flux_bins_by_temperature"),
        }
    ]


def _load_energy_flux(data: dict[str, Any], section: str, component: str) -> EnergyFluxData:
    section = _section_label(section)
    flux_key, component_label = ENERGY_COMPONENTS[component]

    time = data.get("time")
    time_myr = None if time is None else float(time) / SECONDS_PER_MYR

    flux_rows = data.get("fluxes_by_height")
    if flux_rows is None:
        flux_rows = _flux_rows_from_single_height(data)
    if not flux_rows:
        raise ValueError("No height samples found.")

    height = np.asarray([row["height_kpc"] for row in flux_rows], dtype=np.float64)
    if flux_rows[0].get("flux_bins_by_geometric_section") is None:
        if section != "walls":
            raise ValueError("JSON does not contain geometric-section flux for endcaps.")
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
    else:
        negative = np.asarray(
            [
                _to_lsun(
                    row["flux_bins_by_geometric_section"]["negative"][section][
                        flux_key
                    ]
                )
                for row in flux_rows
            ],
            dtype=np.float64,
        )
        positive = np.asarray(
            [
                _to_lsun(
                    row["flux_bins_by_geometric_section"]["positive"][section][
                        flux_key
                    ]
                )
                for row in flux_rows
            ],
            dtype=np.float64,
        )
        net = negative + positive

    if height.size == 0:
        raise ValueError("No height samples found.")
    if not (height.shape == net.shape == negative.shape == positive.shape):
        raise ValueError("Height and energy-flux arrays have different lengths.")
    if not (
        np.all(np.isfinite(height))
        and np.all(np.isfinite(net))
        and np.all(np.isfinite(negative))
        and np.all(np.isfinite(positive))
    ):
        raise ValueError("Height and energy-flux values must be finite.")

    order = np.argsort(height)
    return EnergyFluxData(
        height_kpc=height[order],
        net=net[order],
        negative=negative[order],
        positive=positive[order],
        time_myr=time_myr,
        component_label=component_label,
        temperature=_load_temperature_energy_flux(
            flux_rows,
            order,
            section,
            flux_key,
        ),
    )


def _output_with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _section_output(path: Path, component: str, section: str, include_component: bool) -> Path:
    suffix = f"{component}_{section}" if include_component else section
    return _output_with_suffix(path, suffix)


def _setup_axes(
    ax: plt.Axes,
    title: str,
    component_label: str,
    linear_y: bool,
    axis_limits: AxisLimits | None,
) -> None:
    ax.set_xscale("log")
    if linear_y:
        ax.axhline(0.0, color="0.3", linewidth=0.8)
    else:
        ax.set_yscale("log")
    if axis_limits is not None:
        ax.set_xlim(axis_limits.xlim)
        if axis_limits.ylim is not None:
            ax.set_ylim(axis_limits.ylim)
    ax.set_xlabel("half-height [kpc]")
    quantity = component_label if linear_y else f"{component_label} magnitude"
    ax.set_ylabel(f"{quantity} [{LSUN_LABEL}]")
    fit_plot_title(ax.set_title(title), ax.bbox.width)
    ax.legend(fontsize="small", title=None if linear_y else SIGN_LEGEND)
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
        plot_signed_flux(
            ax,
            data.height_kpc,
            values,
            linear_y=linear_y,
            marker="o",
            linewidth=1.6,
            markersize=4,
            linestyle=linestyle,
            label=label,
        )
    _setup_axes(ax, title, data.component_label, linear_y, axis_limits)
    if not linear_y:
        ax.set_ylim(flux_y_limits([values for values, _, _ in lines], linear_y=False))
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _compute_axis_limits(
    height_kpc: np.ndarray,
    y_values: list[np.ndarray],
    *,
    linear_y: bool,
) -> AxisLimits:
    height = np.asarray(height_kpc, dtype=np.float64)
    if np.any(height <= 0.0):
        raise ValueError("Height values must be positive for log-scaled plots.")
    x_min = float(np.min(height))
    x_max = float(np.max(height))
    if x_min == x_max:
        xlim = (x_min / 1.05, x_max * 1.05)
    else:
        x_pad = 10.0 ** (0.03 * np.log10(x_max / x_min))
        xlim = (x_min / x_pad, x_max * x_pad)

    # Share y limits across temperature plots only in linear mode.
    return AxisLimits(xlim=xlim, ylim=flux_y_limits(y_values, linear_y=True) if linear_y else None)


def _energy_budget_lines(
    data: EnergyFluxData, breakdown: list[EnergyFluxData], temperature_bin: int | None = None,
) -> list[tuple[np.ndarray, str, str]]:
    def signed_fluxes(channel: EnergyFluxData) -> tuple[np.ndarray, np.ndarray]:
        if temperature_bin is not None:
            temp = channel.temperature
            if temp is None:
                raise ValueError(f"Temperature bins missing for {channel.component_label}.")
            return temp.negative[temperature_bin], temp.positive[temperature_bin]
        return channel.negative, channel.positive

    negative, positive = signed_fluxes(data)
    lines = [(negative, "total negative", "-"), (positive, "total positive", "-"),
             (negative + positive, "total net", "--")]
    for channel in breakdown:
        negative, positive = signed_fluxes(channel)
        lines.append((negative + positive, f"{channel.component_label} (net)", "-"))
    return lines


def _plot_energy_flux_set(
    data: EnergyFluxData,
    breakdown: list[EnergyFluxData],
    output: Path,
    title: str,
    linear_y: bool,
) -> list[Path]:
    total_lines = _energy_budget_lines(data, breakdown)
    total_axis_limits = _compute_axis_limits(
        data.height_kpc, [values for values, _, _ in total_lines], linear_y=linear_y,
    )
    if data.temperature is None:
        _plot_lines(
            output,
            data,
            total_lines,
            title,
            linear_y,
            total_axis_limits,
        )
        return [output]

    temp = data.temperature
    net_by_temperature = temp.negative + temp.positive
    bin_lines = [_energy_budget_lines(data, breakdown, i) for i in range(len(temp.labels))]
    axis_limits = _compute_axis_limits(
        data.height_kpc,
        [values for lines in bin_lines for values, _, _ in lines],
        linear_y=linear_y,
    )
    outputs: list[Path] = []

    _plot_lines(
        output,
        data,
        total_lines,
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
            bin_lines[i],
            f"{title}: {label}",
            linear_y,
            axis_limits,
        )
        outputs.append(bin_output)

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cylindrical energy flux versus half-height from a "
            "cylindrical_flux_surface.json file."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="cylindrical_flux_surface.json",
        help="Input cylindrical flux-surface JSON file (default: cylindrical_flux_surface.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="energy_flux_vs_height.png",
        help="Output image path (default: energy_flux_vs_height.png).",
    )
    parser.add_argument(
        "--component",
        choices=("hydro", "mhd", "both"),
        default="both",
        help="Energy flux component to plot (default: both).",
    )
    parser.add_argument(
        "--section",
        choices=("walls", "endcaps", "both"),
        default="both",
        help="Geometric section to plot (default: both).",
    )
    parser.add_argument(
        "--title",
        default="Energy flux through cylindrical surfaces",
        help="Plot title.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Plot signed values on a linear y-axis instead of log magnitudes.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    components = (
        list(ENERGY_BREAKDOWN)
        if args.component == "both"
        else [args.component]
    )
    sections = (
        list(GEOMETRIC_SECTIONS)
        if args.section == "both"
        else [args.section]
    )
    include_component = len(components) > 1

    with Path(args.input).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("fluxes_by_height") or _flux_rows_from_single_height(payload)
    recorded_components = rows[0].get("fluxes", {})
    outputs: list[Path] = []
    for component in components:
        for section in sections:
            data = _load_energy_flux(payload, section, component)
            breakdown = [
                _load_energy_flux(payload, section, term)
                for term in ENERGY_BREAKDOWN[component]
                if ENERGY_COMPONENTS[term][0] in recorded_components
            ]
            title = f"{args.title}: {data.component_label} ({section})"
            if data.time_myr is not None:
                title = f"{title}, t = {data.time_myr:.3f} Myr"
            outputs.extend(
                _plot_energy_flux_set(
                    data,
                    breakdown,
                    _section_output(output, component, section, include_component),
                    title,
                    args.linear_y,
                )
            )

    for path in outputs:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
