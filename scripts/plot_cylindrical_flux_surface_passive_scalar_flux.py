#!/usr/bin/env python3
"""Plot passive scalar flux from scalar_0 as a function of half-height."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np


FLUX_KEY = "passive_scalar_flux_cylinder"
GEOMETRIC_SECTIONS = ("walls", "endcaps")
MSUN_PER_YEAR_LABEL = r"$M_\odot\,yr^{-1}$"
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_MYR = 1.0e6 * SECONDS_PER_YEAR
MSUN_G = 1.98847e33
FLUX_TO_MSUN_PER_YEAR = SECONDS_PER_YEAR / MSUN_G


class TemperatureScalarFlux(NamedTuple):
    labels: list[str]
    negative: np.ndarray
    positive: np.ndarray


class ScalarFluxData(NamedTuple):
    height_kpc: np.ndarray
    net: np.ndarray
    negative: np.ndarray
    positive: np.ndarray
    time_myr: float | None
    scalar_name: str
    temperature: TemperatureScalarFlux | None


class AxisLimits(NamedTuple):
    xlim: tuple[float, float]
    ylim: tuple[float, float]


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


def _flux_from_item(item: dict[str, Any], section: str) -> float:
    fluxes_by_section = item.get("fluxes_by_geometric_section")
    if fluxes_by_section is not None:
        return (
            float(fluxes_by_section.get(section, {}).get(FLUX_KEY, 0.0))
            * FLUX_TO_MSUN_PER_YEAR
        )
    if section == "walls":
        return float(item.get("fluxes", {}).get(FLUX_KEY, 0.0)) * FLUX_TO_MSUN_PER_YEAR
    return 0.0


def _load_temperature_scalar_flux(
    rows: list[dict[str, Any]],
    order: np.ndarray,
    section: str,
) -> TemperatureScalarFlux | None:
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
        negative_rows.append([_flux_from_item(item, section) for item in negative])
        positive_rows.append([_flux_from_item(item, section) for item in positive])

    negative_by_height = np.asarray(negative_rows, dtype=np.float64)
    positive_by_height = np.asarray(positive_rows, dtype=np.float64)
    if not (
        np.all(np.isfinite(negative_by_height))
        and np.all(np.isfinite(positive_by_height))
    ):
        raise ValueError("Temperature-bin passive scalar flux values must be finite.")

    return TemperatureScalarFlux(
        labels=labels,
        negative=negative_by_height[order].T,
        positive=positive_by_height[order].T,
    )


def _load_scalar_flux(path: Path, section: str) -> ScalarFluxData:
    section = _section_label(section)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    time = data.get("time")
    time_myr = None if time is None else float(time) / SECONDS_PER_MYR
    scalar_name = data.get("fields", {}).get("scalar", "scalar_0")

    flux_rows = data.get("fluxes_by_height")
    if flux_rows is None:
        single_fluxes = data.get("fluxes")
        if single_fluxes is None:
            raise ValueError("JSON must contain fluxes_by_height or fluxes.")
        height_kpc = data.get("height_kpc")
        if height_kpc is None:
            raise ValueError("Single-height JSON must contain height_kpc.")
        flux_rows = [
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
    if not flux_rows:
        raise ValueError("No height samples found.")

    height = np.asarray([row["height_kpc"] for row in flux_rows], dtype=np.float64)
    if flux_rows[0].get("flux_bins_by_geometric_section") is None:
        if section != "walls":
            raise ValueError("JSON does not contain geometric-section flux for endcaps.")
        scalar_flux = np.asarray(
            [row["fluxes"][FLUX_KEY] * FLUX_TO_MSUN_PER_YEAR for row in flux_rows],
            dtype=np.float64,
        )
        scalar_flux_negative = np.asarray(
            [
                row.get("flux_bins", {})
                .get("negative", {})
                .get(FLUX_KEY, min(row["fluxes"][FLUX_KEY], 0.0))
                * FLUX_TO_MSUN_PER_YEAR
                for row in flux_rows
            ],
            dtype=np.float64,
        )
        scalar_flux_positive = np.asarray(
            [
                row.get("flux_bins", {})
                .get("positive", {})
                .get(FLUX_KEY, max(row["fluxes"][FLUX_KEY], 0.0))
                * FLUX_TO_MSUN_PER_YEAR
                for row in flux_rows
            ],
            dtype=np.float64,
        )
    else:
        scalar_flux_negative = np.asarray(
            [
                row["flux_bins_by_geometric_section"]["negative"][section][FLUX_KEY]
                * FLUX_TO_MSUN_PER_YEAR
                for row in flux_rows
            ],
            dtype=np.float64,
        )
        scalar_flux_positive = np.asarray(
            [
                row["flux_bins_by_geometric_section"]["positive"][section][FLUX_KEY]
                * FLUX_TO_MSUN_PER_YEAR
                for row in flux_rows
            ],
            dtype=np.float64,
        )
        scalar_flux = scalar_flux_negative + scalar_flux_positive

    if height.size == 0:
        raise ValueError("No height samples found.")
    if not (
        height.shape
        == scalar_flux.shape
        == scalar_flux_negative.shape
        == scalar_flux_positive.shape
    ):
        raise ValueError("Height and passive scalar flux arrays have different lengths.")
    if not (
        np.all(np.isfinite(height))
        and np.all(np.isfinite(scalar_flux))
        and np.all(np.isfinite(scalar_flux_negative))
        and np.all(np.isfinite(scalar_flux_positive))
    ):
        raise ValueError("Height and passive scalar flux values must be finite.")

    order = np.argsort(height)
    return ScalarFluxData(
        height_kpc=height[order],
        net=scalar_flux[order],
        negative=scalar_flux_negative[order],
        positive=scalar_flux_positive[order],
        time_myr=time_myr,
        scalar_name=str(scalar_name),
        temperature=_load_temperature_scalar_flux(flux_rows, order, section),
    )


def _output_with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _section_output(path: Path, section: str) -> Path:
    return _output_with_suffix(path, section)


def _setup_axes(
    ax: plt.Axes,
    title: str,
    scalar_name: str,
    linear_y: bool,
    axis_limits: AxisLimits | None,
) -> None:
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xscale("log")
    if not linear_y:
        ax.set_yscale("symlog")
    if axis_limits is not None:
        ax.set_xlim(axis_limits.xlim)
        ax.set_ylim(axis_limits.ylim)
    ax.set_xlabel("half-height [kpc]")
    ax.set_ylabel(f"{scalar_name} flux [{MSUN_PER_YEAR_LABEL}]")
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, which="both", alpha=0.25)


def _plot_lines(
    output: Path,
    data: ScalarFluxData,
    lines: list[tuple[np.ndarray, str, str]],
    title: str,
    linear_y: bool,
    axis_limits: AxisLimits | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for values, label, linestyle in lines:
        ax.plot(
            data.height_kpc,
            values,
            marker="o",
            linewidth=1.6,
            markersize=4,
            linestyle=linestyle,
            label=label,
        )
    _setup_axes(ax, title, data.scalar_name, linear_y, axis_limits)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _compute_axis_limits(
    height_kpc: np.ndarray,
    y_values: list[np.ndarray],
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


def _plot_scalar_flux_set(
    data: ScalarFluxData,
    output: Path,
    title: str,
    linear_y: bool,
) -> list[Path]:
    if data.temperature is None:
        axis_limits = _compute_axis_limits(
            data.height_kpc,
            [data.positive, data.negative, data.net],
        )
        _plot_lines(
            output,
            data,
            [
                (data.negative, "inflows", "-"),
                (data.positive, "outflows", "-"),
                (data.net, "net", "--"),
            ],
            title,
            linear_y,
            axis_limits,
        )
        return [output]

    temp = data.temperature
    net_by_temperature = temp.negative + temp.positive
    total_inflows = np.sum(temp.negative, axis=0)
    total_outflows = np.sum(temp.positive, axis=0)
    total_net = total_inflows + total_outflows
    total_axis_limits = _compute_axis_limits(
        data.height_kpc,
        [total_inflows, total_outflows, total_net],
    )
    axis_limits = _compute_axis_limits(
        data.height_kpc,
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
            (total_inflows, "inflows", "-"),
            (total_outflows, "outflows", "-"),
            (total_net, "net", "--"),
        ],
        f"{title}: all temperature phases",
        linear_y,
        total_axis_limits,
    )
    outputs.append(output)

    inflows_output = _output_with_suffix(output, "inflows_by_temperature")
    _plot_lines(
        inflows_output,
        data,
        [(temp.negative[i], temp.labels[i], "-") for i in range(len(temp.labels))],
        f"{title}: inflows by temperature",
        linear_y,
        axis_limits,
    )
    outputs.append(inflows_output)

    outflows_output = _output_with_suffix(output, "outflows_by_temperature")
    _plot_lines(
        outflows_output,
        data,
        [(temp.positive[i], temp.labels[i], "-") for i in range(len(temp.labels))],
        f"{title}: outflows by temperature",
        linear_y,
        axis_limits,
    )
    outputs.append(outflows_output)

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
                (temp.negative[i], "inflows", "-"),
                (temp.positive[i], "outflows", "-"),
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
        description=(
            "Plot passive scalar flux from scalar_0 versus half-height from a "
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
        default="passive_scalar_flux_vs_height.png",
        help="Output image path (default: passive_scalar_flux_vs_height.png).",
    )
    parser.add_argument(
        "--title",
        default="Passive scalar flux through cylindrical surfaces",
        help="Plot title.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of symlog.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    section_data = [
        (section, _load_scalar_flux(Path(args.input), section))
        for section in GEOMETRIC_SECTIONS
    ]
    outputs: list[Path] = []
    for section, data in section_data:
        title = f"{args.title} ({section})"
        if data.time_myr is not None:
            title = f"{title}, t = {data.time_myr:.3f} Myr"
        outputs.extend(
            _plot_scalar_flux_set(
                data,
                _section_output(output, section),
                title,
                args.linear_y,
            )
        )

    for path in outputs:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
