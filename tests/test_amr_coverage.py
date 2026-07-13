from __future__ import annotations

from analysis.amr_coverage import (
    axis_ranges_by_level,
    covered_plane_boxes,
    covered_slab_boxes,
    covered_volume_boxes,
    intersecting_plane_blocks,
    intersecting_slab_blocks,
    plane_indices_by_level,
)
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta


def _levels() -> list[LevelMeta]:
    return [
        LevelMeta(
            geom=LevelGeom(
                dx=(1.0, 1.0, 1.0),
                x0=(0.0, 0.0, 0.0),
                index_origin=(10, 10, 10),
                ref_ratio=2,
            ),
            boxes=[
                BlockBox((10, 10, 10), (13, 17, 17)),
                BlockBox((14, 10, 10), (17, 17, 17)),
            ],
        ),
        LevelMeta(
            geom=LevelGeom(
                dx=(0.5, 0.5, 0.5),
                x0=(0.0, 0.0, 0.0),
                index_origin=(20, 20, 20),
                ref_ratio=1,
            ),
            boxes=[BlockBox((20, 20, 20), (27, 35, 35))],
        ),
    ]


def test_coverage_maps_physical_selections_across_levels() -> None:
    levels = _levels()

    plane_indices = plane_indices_by_level(levels, axis=2, coord=2.5)
    assert plane_indices == {0: 12, 1: 24}
    assert covered_plane_boxes(
        levels, level=0, axis=2, plane_indices=plane_indices
    ) == [((10, 10, 12), (13, 17, 12))]

    axis_ranges = axis_ranges_by_level(levels, axis=2, bounds=(1.0, 3.0))
    assert axis_ranges == {0: (11, 12), 1: (22, 25)}
    assert covered_slab_boxes(
        levels, level=0, axis=2, axis_ranges=axis_ranges
    ) == [((10, 10, 11), (13, 17, 12))]


def test_volume_coverage_is_origin_aware_and_empty_on_finest_level() -> None:
    levels = _levels()

    assert covered_volume_boxes(levels, level=0) == [
        ((10, 10, 10), (13, 17, 17))
    ]
    assert covered_volume_boxes(levels, level=1) == []


def test_block_selection_uses_the_same_plane_and_slab_seam() -> None:
    coarse = _levels()[0]
    rect = (0.0, 0.0, 3.9, 8.0)

    assert list(
        intersecting_plane_blocks(
            coarse, axis=2, plane_index=12, rect=rect
        )
    ) == [0]
    assert list(
        intersecting_slab_blocks(
            coarse, axis=2, axis_range=(11, 12), rect=rect
        )
    ) == [0]
