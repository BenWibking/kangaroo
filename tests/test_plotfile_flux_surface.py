from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from scripts.plotfile_flux_surface import (
    _flux_rows_and_derived,
    _intersecting_validation_blocks,
    _pick_field,
    _validate_selected_fields,
)


@dataclass(frozen=True)
class _Box:
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]


@dataclass(frozen=True)
class _Geom:
    dx: tuple[float, float, float]
    x0: tuple[float, float, float]
    index_origin: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class _Level:
    geom: _Geom
    boxes: list[_Box]


@dataclass(frozen=True)
class _Step:
    levels: list[_Level]


@dataclass(frozen=True)
class _RunMeta:
    steps: list[_Step]


@dataclass(frozen=True)
class _Dataset:
    step: int = 0
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(
                self,
                "metadata",
                {"var_names": ["rho", "E"]},
            )


def _runmeta_with_intersecting_and_far_levels() -> _RunMeta:
    return _RunMeta(
        steps=[
            _Step(
                levels=[
                    _Level(
                        geom=_Geom(dx=(1.0, 1.0, 1.0), x0=(0.0, -0.5, -0.5)),
                        boxes=[_Box((0, 0, 0), (0, 0, 0))],
                    ),
                    _Level(
                        geom=_Geom(dx=(0.5, 0.5, 0.5), x0=(0.0, -0.5, -0.5)),
                        boxes=[
                            _Box((10, 0, 0), (10, 0, 0)),
                            _Box((0, 0, 0), (0, 0, 0)),
                        ],
                    ),
                    _Level(
                        geom=_Geom(dx=(0.25, 0.25, 0.25), x0=(100.0, 100.0, 100.0)),
                        boxes=[_Box((0, 0, 0), (8, 8, 8))],
                    ),
                ]
            )
        ]
    )


def test_flux_surface_validation_samples_only_intersecting_levels() -> None:
    ds = _Dataset()
    runmeta = _runmeta_with_intersecting_and_far_levels()

    assert _intersecting_validation_blocks(ds, runmeta, radius=0.5) == [(0, 0), (1, 1)]

    _validate_selected_fields(
        ds,
        runmeta,
        {
            "density": ("rho", 1),
            "energy": ("E", 2),
        },
        radii=[0.5],
    )


def test_flux_surface_validation_rejects_fields_using_metadata_only() -> None:
    ds = _Dataset()
    runmeta = _runmeta_with_intersecting_and_far_levels()

    with pytest.raises(RuntimeError, match="metadata"):
        _validate_selected_fields(
            ds,
            runmeta,
            {
                "density": ("rho", 1),
                "energy": ("missing", 2),
            },
            radii=[0.5],
        )


def test_flux_surface_explicit_field_must_exist_in_metadata() -> None:
    with pytest.raises(RuntimeError, match="--list-fields"):
        _pick_field("density", "missing", ["rho", "E"])


def test_flux_surface_json_includes_negative_and_positive_bins() -> None:
    radii = np.array([1.0, 2.0], dtype=np.float64)
    values = np.array(
        [
            [[-2.0, -3.0, -4.0, -5.0], [7.0, 11.0, 13.0, 17.0]],
            [[-19.0, -23.0, -29.0, -31.0], [37.0, 41.0, 43.0, 47.0]],
        ],
        dtype=np.float64,
    )

    rows, derived = _flux_rows_and_derived(radii, values, pc_cm=1.0)

    assert rows[0]["flux_bins"]["negative"]["mass_flux_sphere"] == -2.0
    assert rows[0]["flux_bins"]["positive"]["mass_flux_sphere"] == 7.0
    assert rows[0]["fluxes"]["mass_flux_sphere"] == 5.0
    assert rows[1]["flux_bins"]["negative"]["mhd_energy_flux_sphere"] == -29.0
    assert rows[1]["flux_bins"]["positive"]["mhd_energy_flux_sphere"] == 43.0
    assert rows[1]["fluxes"]["mhd_energy_flux_sphere"] == 14.0
    assert derived["mass_flux_msun_per_yr"] is None
    assert derived["mass_flux_msun_per_yr_bins"] is None
    assert "mass_flux_msun_per_yr_bins" in derived["mass_flux_msun_per_yr_by_radius"][0]
