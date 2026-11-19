#!/usr/bin/env python3
"""Plot a saved density–temperature histogram without opening the plotfile."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def plot_histogram(input_path: Path, output: Path) -> None:
    data = json.loads(input_path.read_text())
    density = np.asarray(data["density_edges"], dtype=float)
    temperature = np.asarray(data["temperature_edges"], dtype=float)
    mass = np.asarray(data["mass_msun"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    try:
        ax.set(xscale="log", yscale="log", xlim=(density[0], density[-1]),
               ylim=(temperature[0], temperature[-1]), xlabel=r"Gas density [g cm$^{-3}$]",
               ylabel="Temperature [K]", title="Mass-weighted density–temperature phase diagram")
        positive = mass[mass > 0]
        if positive.size:
            lo, hi = float(positive.min()), float(positive.max())
            if lo == hi:
                lo, hi = lo / 10.0, hi * 10.0
            mesh = ax.pcolormesh(density, temperature, np.ma.masked_less_equal(mass.T, 0),
                                 norm=LogNorm(vmin=lo, vmax=hi), shading="flat")
            fig.colorbar(mesh, ax=ax, label=r"Gas mass per bin [$M_\odot$]")
        else:
            ax.text(0.5, 0.5, "No gas mass in the selected range", ha="center", transform=ax.transAxes)
        fig.savefig(output, dpi=200)
    finally:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("temperature_density.png"))
    args = parser.parse_args()
    plot_histogram(args.input, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
