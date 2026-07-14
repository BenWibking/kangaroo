from __future__ import annotations

import math

from analysis import _core


def test_particle_value_counts_rejects_overflowing_record_count() -> None:
    assert _core.test_decode_particle_value_counts(1 << 60, 0) == 0
    assert _core.test_decode_particle_value_counts(1, 0) == 0
    assert _core.test_decode_particle_value_counts(1, 16) == 1


def test_axis_band_rejects_non_finite_geometry() -> None:
    empty = (0, -1)
    assert (
        _core.test_cells_intersecting_axis_band(
            0.0, 1.0, 0.0, math.inf, 0, 0, 3
        )
        == empty
    )
    assert (
        _core.test_cells_intersecting_axis_band(
            0.0, 1.0, math.nan, 1.0, 0, 0, 3
        )
        == empty
    )


def test_axis_band_handles_finite_scaling_overflow() -> None:
    assert _core.test_cells_intersecting_axis_band(
        1.0, 2.0, 0.0, math.ulp(0.0), 0, 0, 3
    ) == (0, -1)


def test_axis_band_preserves_normal_intersection_range() -> None:
    assert _core.test_cells_intersecting_axis_band(
        1.25, 2.25, 0.0, 1.0, 0, 0, 3
    ) == (0, 3)
