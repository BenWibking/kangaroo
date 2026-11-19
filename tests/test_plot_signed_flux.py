import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from scripts import plot_cylindrical_flux_surface_mass_flux as mass_plot
from scripts.plot_signed_flux import fit_plot_title, flux_y_limits, plot_signed_flux


def test_signed_flux_breaks_lines_at_zero_and_sign_changes() -> None:
    fig, ax = plt.subplots()
    try:
        y = np.array([2.0, 3.0, -4.0, -5.0, 0.0, -6.0, 7.0])
        plot_signed_flux(ax, np.arange(1, len(y) + 1), y,
                         label="net", linear_y=False, linestyle="--")
        positive, negative = ax.lines
        assert positive.get_linestyle() == "-"
        assert negative.get_linestyle() == "--"
        assert positive.get_color() == negative.get_color()
        np.testing.assert_array_equal(positive.get_ydata(), [2, 3, np.nan, np.nan, np.nan, np.nan, 7])
        np.testing.assert_array_equal(negative.get_ydata(), [np.nan, np.nan, 4, 5, np.nan, 6, np.nan])
        assert ax.get_legend_handles_labels()[1] == ["net"]
    finally:
        plt.close(fig)


def test_negative_only_flux_and_linear_mode() -> None:
    fig, (log_ax, linear_ax) = plt.subplots(1, 2)
    try:
        x, y = np.array([1.0, 2.0]), np.array([-5.0, -2.0])
        plot_signed_flux(log_ax, x, y, label="inflows", linear_y=False)
        (line,) = log_ax.lines
        assert line.get_linestyle() == "--"
        np.testing.assert_array_equal(line.get_ydata(), [5, 2])
        assert log_ax.get_legend_handles_labels()[1] == ["inflows"]
        plot_signed_flux(linear_ax, x, y, label="inflows", linear_y=True)
        (line,) = linear_ax.lines
        np.testing.assert_array_equal(line.get_ydata(), y)
        lo, hi = flux_y_limits([y], linear_y=False)
        assert (lo, hi) == pytest.approx((0.005, 5.5))
        lo, hi = flux_y_limits([np.zeros(2)], linear_y=False)
        assert 0 < lo < hi
    finally:
        plt.close(fig)


@pytest.mark.parametrize("figure_title", [False, True])
def test_long_title_wraps_inside_figure(figure_title: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        text = (r"Energy flux through cylindrical surfaces: $L_z + L_x$ transport "
                "through endcaps at 554.450 Myr, with positive flux by temperature")
        title = fig.suptitle(text) if figure_title else ax.set_title(text)
        fit_plot_title(title, fig.bbox.width if figure_title else ax.bbox.width)
        fig.tight_layout()
        fig.canvas.draw()
        assert "\n" in title.get_text()
        assert r"$L_z + L_x$" in title.get_text()
        assert " ".join(title.get_text().splitlines()) == text
        bounds = title.get_window_extent()
        assert fig.bbox.x0 < bounds.x0 < bounds.x1 < fig.bbox.x1
    finally:
        plt.close(fig)


def test_temperature_plots_use_their_own_peak(tmp_path, monkeypatch) -> None:
    limits = {}

    def record_limits(fig, path, **kwargs):
        limits[path.name] = fig.axes[0].get_ylim()

    monkeypatch.setattr(Figure, "savefig", record_limits)
    temperature = mass_plot.TemperatureMassFlux(
        ["cold", "hot"],
        np.array([[-1e-6, -2e-6], [-10.0, -20.0]]),
        np.array([[1e-7, 2e-7], [1.0, 2.0]]),
    )
    negative = temperature.negative.sum(axis=0)
    positive = temperature.positive.sum(axis=0)
    data = mass_plot.MassFluxData(
        np.array([1.0, 2.0]), negative + positive, negative, positive, None, temperature,
    )
    mass_plot._plot_mass_flux_set(data, tmp_path / "flux.png", "Flux", linear_y=False)
    assert limits["flux_temperature_bin_00.png"] == pytest.approx((2e-9, 2.2e-6))
    assert limits["flux_temperature_bin_01.png"] == pytest.approx((0.02, 22.0))
