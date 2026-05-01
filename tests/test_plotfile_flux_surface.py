from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.plotfile_flux_surface import (
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
