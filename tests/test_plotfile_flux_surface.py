from __future__ import annotations

from dataclasses import dataclass

from scripts.plotfile_flux_surface import (
    _intersecting_validation_blocks,
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


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_task_chunk(self, **kwargs):
        self.calls.append(kwargs)
        return b"x"


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

    rt = _Runtime()
    _validate_selected_fields(
        ds,
        rt,
        runmeta,
        {
            "density": ("rho", 1),
            "energy": ("E", 2),
        },
        radius=0.5,
    )

    assert [(call["level"], call["block"], call["field"]) for call in rt.calls] == [
        (0, 0, 1),
        (1, 1, 1),
        (0, 0, 2),
        (1, 1, 2),
    ]
