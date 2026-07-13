from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any


Index3 = tuple[int, int, int]
IndexBox = tuple[Index3, Index3]


def plane_indices_by_level(
    levels: Sequence[Any], *, axis: int, coord: float
) -> dict[int, int]:
    """Map one physical plane to the corresponding cell index on every level."""
    if not levels:
        return {}
    base_geom = levels[0].geom
    base_index = _cell_index(base_geom, axis, coord)
    base_origin = base_geom.index_origin[axis]
    return {
        level: (base_index - base_origin)
        * _refinement_ratio(levels, 0, level)
        + levels[level].geom.index_origin[axis]
        for level in range(len(levels))
    }


def axis_ranges_by_level(
    levels: Sequence[Any], *, axis: int, bounds: tuple[float, float]
) -> dict[int, tuple[int, int]]:
    """Map a half-open physical interval to inclusive cell ranges on every level."""
    return {
        level: _axis_index_range(level_meta.geom, axis, bounds)
        for level, level_meta in enumerate(levels)
    }


def covered_volume_boxes(levels: Sequence[Any], *, level: int) -> list[IndexBox]:
    """Return finer-level boxes coarsened into ``level`` index space."""
    return _coarsened_fine_boxes(levels, level=level)


def covered_plane_boxes(
    levels: Sequence[Any],
    *,
    level: int,
    axis: int,
    plane_indices: dict[int, int],
) -> list[IndexBox]:
    """Return coverage of one plane by all finer levels, in coarse index space."""
    coarse_index = plane_indices[level]
    covered = _coarsened_fine_boxes(
        levels,
        level=level,
        include=lambda fine, box: box.lo[axis]
        <= plane_indices[fine]
        <= box.hi[axis],
    )
    clamped: list[IndexBox] = []
    for lo, hi in covered:
        if not lo[axis] <= coarse_index <= hi[axis]:
            continue
        clipped_lo = list(lo)
        clipped_hi = list(hi)
        clipped_lo[axis] = coarse_index
        clipped_hi[axis] = coarse_index
        clamped.append((tuple(clipped_lo), tuple(clipped_hi)))
    return clamped


def covered_slab_boxes(
    levels: Sequence[Any],
    *,
    level: int,
    axis: int,
    axis_ranges: dict[int, tuple[int, int]],
) -> list[IndexBox]:
    """Return coverage within one axis interval by all finer levels."""
    coarse_lo, coarse_hi = axis_ranges[level]
    covered = _coarsened_fine_boxes(
        levels,
        level=level,
        include=lambda fine, box: not (
            box.hi[axis] < axis_ranges[fine][0]
            or box.lo[axis] > axis_ranges[fine][1]
        ),
    )
    clamped: list[IndexBox] = []
    for lo, hi in covered:
        if hi[axis] < coarse_lo or lo[axis] > coarse_hi:
            continue
        clipped_lo = list(lo)
        clipped_hi = list(hi)
        clipped_lo[axis] = max(clipped_lo[axis], coarse_lo)
        clipped_hi[axis] = min(clipped_hi[axis], coarse_hi)
        clamped.append((tuple(clipped_lo), tuple(clipped_hi)))
    return clamped


def intersecting_plane_blocks(
    level_meta: Any,
    *,
    axis: int,
    plane_index: int,
    rect: tuple[float, float, float, float],
) -> Iterable[int]:
    """Yield blocks intersecting an indexed plane and its in-plane rectangle."""
    return _intersecting_blocks(
        level_meta,
        axis=axis,
        axis_range=(plane_index, plane_index),
        rect=rect,
    )


def intersecting_slab_blocks(
    level_meta: Any,
    *,
    axis: int,
    axis_range: tuple[int, int],
    rect: tuple[float, float, float, float],
) -> Iterable[int]:
    """Yield blocks intersecting an indexed slab and its transverse rectangle."""
    return _intersecting_blocks(
        level_meta,
        axis=axis,
        axis_range=axis_range,
        rect=rect,
    )


def _refinement_ratio(levels: Sequence[Any], coarse: int, fine: int) -> int:
    ratio = 1
    for level in range(coarse, fine):
        ratio *= int(levels[level].geom.ref_ratio)
    return ratio


def _cell_index(geom: Any, axis: int, coord: float) -> int:
    origin = geom.index_origin[axis]
    dx = geom.dx[axis]
    if dx == 0.0:
        return origin
    return int(math.floor((coord - geom.x0[axis]) / dx)) + origin


def _axis_index_range(
    geom: Any, axis: int, bounds: tuple[float, float]
) -> tuple[int, int]:
    lo, hi = sorted(bounds)
    hi_adjusted = math.nextafter(hi, -math.inf)
    index_lo = _cell_index(geom, axis, lo)
    index_hi = _cell_index(geom, axis, hi_adjusted)
    return min(index_lo, index_hi), max(index_lo, index_hi)


def _coarsen_box(
    lo: Index3,
    hi: Index3,
    *,
    ratio: int,
    fine_origin: Index3,
    coarse_origin: Index3,
) -> IndexBox:
    coarse_lo = tuple(
        int(math.floor((lo[axis] - fine_origin[axis]) / ratio))
        + coarse_origin[axis]
        for axis in range(3)
    )
    coarse_hi = tuple(
        int(math.floor((hi[axis] - fine_origin[axis] + 1) / ratio))
        + coarse_origin[axis]
        - 1
        for axis in range(3)
    )
    return coarse_lo, coarse_hi


def _coarsened_fine_boxes(
    levels: Sequence[Any],
    *,
    level: int,
    include: Callable[[int, Any], bool] | None = None,
) -> list[IndexBox]:
    coarse_origin = levels[level].geom.index_origin
    covered: list[IndexBox] = []
    for fine in range(level + 1, len(levels)):
        ratio = _refinement_ratio(levels, level, fine)
        fine_origin = levels[fine].geom.index_origin
        for box in levels[fine].boxes:
            if include is not None and not include(fine, box):
                continue
            covered.append(
                _coarsen_box(
                    box.lo,
                    box.hi,
                    ratio=ratio,
                    fine_origin=fine_origin,
                    coarse_origin=coarse_origin,
                )
            )
    return covered


def _intersecting_blocks(
    level_meta: Any,
    *,
    axis: int,
    axis_range: tuple[int, int],
    rect: tuple[float, float, float, float],
) -> Iterable[int]:
    transverse_axes = [candidate for candidate in range(3) if candidate != axis]
    u_axis, v_axis = transverse_axes
    u0, v0, u1, v1 = rect
    umin, umax = sorted((u0, u1))
    vmin, vmax = sorted((v0, v1))
    axis_lo, axis_hi = axis_range
    geom = level_meta.geom

    for block, box in enumerate(level_meta.boxes):
        if box.hi[axis] < axis_lo or box.lo[axis] > axis_hi:
            continue
        block_u0, block_u1 = _physical_bounds(
            box.lo[u_axis], box.hi[u_axis], geom, u_axis
        )
        block_v0, block_v1 = _physical_bounds(
            box.lo[v_axis], box.hi[v_axis], geom, v_axis
        )
        if _overlaps(block_u0, block_u1, umin, umax) and _overlaps(
            block_v0, block_v1, vmin, vmax
        ):
            yield block


def _physical_bounds(lo: int, hi: int, geom: Any, axis: int) -> tuple[float, float]:
    origin = geom.index_origin[axis]
    dx = geom.dx[axis]
    x0 = geom.x0[axis]
    return x0 + (lo - origin) * dx, x0 + (hi + 1 - origin) * dx


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)
