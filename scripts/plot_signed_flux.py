"""Plot signed flux magnitudes on a logarithmic axis."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text


SIGN_LEGEND = "Solid: positive; dashed: negative"


def fit_plot_title(title: Text, width_px: float) -> None:
    """Wrap a title by its rendered width before laying out the figure."""
    max_width = width_px * 0.9  # Leave room for the plot margins.
    if title.get_window_extent().width <= max_width:
        return

    wrapped = []
    for paragraph in title.get_text().split("\n"):
        line = ""
        # Keep inline math (e.g. $L_z$) together, even if it contains spaces.
        for word in re.findall(r"(?:\$[^$]+\$|[^\s$]|\$)+", paragraph):
            candidate = f"{line} {word}" if line else word
            title.set_text(candidate)
            if line and title.get_window_extent().width > max_width:
                wrapped.append(line)
                line = word
            else:
                line = candidate
        wrapped.append(line)

    title.set_text("\n".join(wrapped))
    width = title.get_window_extent().width
    if width > max_width:  # A single unbroken token cannot be wrapped.
        title.set_fontsize(title.get_fontsize() * max_width / width)


def plot_signed_flux(
    ax: plt.Axes, x: np.ndarray, values: np.ndarray, *, label: str,
    linear_y: bool, linestyle: str = "-", **kwargs,
) -> None:
    if linear_y:
        ax.plot(x, values, label=label, linestyle=linestyle, **kwargs)
        return

    finite = np.isfinite(values)
    positive = finite & (values > 0)
    negative = finite & (values < 0)
    if not positive.any() and negative.any():
        ax.plot(x, np.where(negative, -values, np.nan), label=label, linestyle="--", **kwargs)
        return

    # Keep masked samples in place so the line breaks at zeros and sign changes.
    (line,) = ax.plot(x, np.where(positive, values, np.nan), label=label, linestyle="-", **kwargs)
    if negative.any():
        ax.plot(x, np.where(negative, -values, np.nan), color=line.get_color(), linestyle="--", **kwargs)


def flux_y_limits(values: list[np.ndarray], *, linear_y: bool) -> tuple[float, float]:
    y = np.concatenate([np.ravel(series) for series in values])
    y = y[np.isfinite(y)]
    if not linear_y:
        peak = float(np.max(np.abs(y), initial=0.0))
        if peak == 0.0:
            return (1.0e-3, 1.0)
        # Show three decades below the largest magnitude, with room for markers at the top.
        return (peak / 1000.0, peak * 1.1)

    y_min, y_max = float(y.min()), float(y.max())
    padding = 0.05 * (y_max - y_min) if y_min != y_max else max(abs(y_min) * 0.05, 1.0)
    lo, hi = y_min - padding, y_max + padding
    if lo <= 0.0 <= hi:
        max_abs = max(abs(lo), abs(hi))
        return (-max_abs, max_abs)
    return (lo, hi)
