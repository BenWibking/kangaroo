from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from .ctx import LoweringContext
from .plan import FieldRef


class VorticityMag:
    def __init__(
        self,
        vel_field: int | tuple[int, int, int],
        out_name: str = "vortmag",
        stencil_radius: int = 1,
    ) -> None:
        self.vel_field = vel_field
        self.out_name = out_name
        self.stencil_radius = stencil_radius

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        dom = ctx.domain(step=ds.step, level=ds.level)

        grad_fields: list[FieldRef] = []
        vort = ctx.output_field(self.out_name)

        s0 = ctx.stage("neighbor_fetch")
        s1 = ctx.stage("gradients", after=[s0])
        if isinstance(self.vel_field, tuple):
            component_fields = list(self.vel_field)
        else:
            component_fields = [self.vel_field]

        for comp, field_id in enumerate(component_fields):
            fetch_f = ctx.temp_field(f"nbrPatches_{comp}")
            s0.map_blocks(
                name=f"fetch_nbr_{comp}",
                kernel="amr_subbox_fetch_pack",
                domain=dom,
                inputs=[],
                outputs=[fetch_f],
                deps={"kind": "None"},
                params={
                    "input_field": field_id,
                    "input_version": 0,
                    "input_step": ds.step,
                    "input_level": ds.level,
                    "halo_cells": self.stencil_radius,
                },
            )
            grad_f = ctx.temp_field(f"gradU_{comp}")
            grad_fields.append(grad_f)
            s1.map_blocks(
                name=f"gradU_{comp}",
                kernel="gradU_stencil",
                domain=dom,
                inputs=[FieldRef(field_id), fetch_f],
                outputs=[grad_f],
                deps={"kind": "None"},
                params={
                    "order": 2,
                    "input_field": field_id,
                    "input_version": 0,
                    "input_step": ds.step,
                    "input_level": ds.level,
                    "stencil_radius": self.stencil_radius,
                },
            )

        s2 = ctx.stage("vortmag", after=[s1])
        s2.map_blocks(
            name="vortmag",
            kernel="vorticity_mag",
            domain=dom,
            inputs=grad_fields,
            outputs=[vort],
            deps={"kind": "None"},
            params={},
        )

        return ctx.fragment([s0, s1, s2])


def _axis_index(axis: str | int) -> int:
    if isinstance(axis, int):
        if axis in (0, 1, 2):
            return axis
        raise ValueError("axis must be 0, 1, or 2")
    axis_l = axis.lower()
    if axis_l == "x":
        return 0
    if axis_l == "y":
        return 1
    if axis_l == "z":
        return 2
    raise ValueError("axis must be 'x', 'y', 'z', or 0/1/2")


def _bounds_1d(
    lo: int, hi: int, x0: float, dx: float, origin: int
) -> tuple[float, float]:
    return x0 + (lo - origin) * dx, x0 + (hi + 1 - origin) * dx


def _overlaps_1d(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)


def _resolve_bytes_per_value(
    ctx: LoweringContext,
    *,
    field: int,
    bytes_per_value: int | None,
    level: int = 0,
) -> int:
    if bytes_per_value is not None:
        return int(bytes_per_value)

    ds = ctx.dataset
    runtime = getattr(ds, "runtime", None)
    if runtime is None:
        raise RuntimeError(
            "bytes_per_value was not provided and could not be inferred: dataset runtime is missing"
        )
    infer = getattr(ds, "infer_bytes_per_value", None)
    if callable(infer):
        try:
            return int(infer(runtime, field=field, level=level, step=ds.step))
        except Exception:
            pass
    return 8


def _coarsen_box(
    lo: tuple[int, int, int],
    hi: tuple[int, int, int],
    *,
    ratio: int,
    fine_origin: tuple[int, int, int],
    coarse_origin: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    clo = []
    chi = []
    for d in range(3):
        clo.append(int(math.floor((lo[d] - fine_origin[d]) / ratio)) + coarse_origin[d])
        chi.append(int(math.floor((hi[d] - fine_origin[d] + 1) / ratio)) + coarse_origin[d] - 1)
    return tuple(clo), tuple(chi)


class UniformSlice:
    def __init__(
        self,
        field: int,
        *,
        axis: str | int,
        coord: float,
        rect: tuple[float, float, float, float],
        resolution: tuple[int, int],
        out_name: str = "slice",
        bytes_per_value: int | None = None,
        reduce_fan_in: Optional[int] = None,
    ) -> None:
        self.field = field
        self.axis = axis
        self.coord = coord
        self.rect = rect
        self.resolution = resolution
        self.out_name = out_name
        self.bytes_per_value = bytes_per_value
        self.reduce_fan_in = reduce_fan_in

    def _intersecting_blocks_level(self, level_meta, *, plane_index: int | None = None) -> Iterable[int]:
        axis_idx = _axis_index(self.axis)
        u_axis, v_axis = [i for i in range(3) if i != axis_idx]

        u0, v0, u1, v1 = self.rect
        umin, umax = (u0, u1) if u0 <= u1 else (u1, u0)
        vmin, vmax = (v0, v1) if v0 <= v1 else (v1, v0)

        geom = level_meta.geom
        for i, box in enumerate(level_meta.boxes):
            if plane_index is not None:
                if plane_index < box.lo[axis_idx] or plane_index > box.hi[axis_idx]:
                    continue
            else:
                ax0, ax1 = _bounds_1d(
                    box.lo[axis_idx],
                    box.hi[axis_idx],
                    geom.x0[axis_idx],
                    geom.dx[axis_idx],
                    geom.index_origin[axis_idx],
                )
                if not (ax0 <= self.coord <= ax1):
                    continue

            u0_b, u1_b = _bounds_1d(
                box.lo[u_axis],
                box.hi[u_axis],
                geom.x0[u_axis],
                geom.dx[u_axis],
                geom.index_origin[u_axis],
            )
            v0_b, v1_b = _bounds_1d(
                box.lo[v_axis],
                box.hi[v_axis],
                geom.x0[v_axis],
                geom.dx[v_axis],
                geom.index_origin[v_axis],
            )
            if not (_overlaps_1d(u0_b, u1_b, umin, umax) and _overlaps_1d(v0_b, v1_b, vmin, vmax)):
                continue
            yield i

    def _reduce_fan_in(self, num_inputs: int) -> int:
        if self.reduce_fan_in is None:
            return max(2, int(math.sqrt(num_inputs)))
        return max(1, int(self.reduce_fan_in))

    def _coarsen_box(
        self,
        lo: tuple[int, int, int],
        hi: tuple[int, int, int],
        *,
        ratio: int,
        fine_origin: tuple[int, int, int],
        coarse_origin: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        clo = []
        chi = []
        for d in range(3):
            clo.append(int(math.floor((lo[d] - fine_origin[d]) / ratio)) + coarse_origin[d])
            chi.append(int(math.floor((hi[d] - fine_origin[d] + 1) / ratio)) + coarse_origin[d] - 1)
        return tuple(clo), tuple(chi)

    def _ref_ratio_between(self, levels, coarse: int, fine: int) -> int:
        ratio = 1
        for lev in range(coarse, fine):
            ratio *= int(levels[lev].geom.ref_ratio)
        return ratio

    def _cell_index(self, geom, axis: int, coord: float) -> int:
        x0 = geom.x0[axis]
        dx = geom.dx[axis]
        origin = geom.index_origin[axis]
        if dx == 0.0:
            return origin
        idx_f = (coord - x0) / dx
        return int(math.floor(idx_f)) + origin

    def _covered_boxes_for_level(
        self,
        ctx: LoweringContext,
        *,
        level: int,
        axis_idx: int,
        plane_index_by_level: dict[int, int],
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        ds = ctx.dataset
        levels = ctx.runmeta.steps[ds.step].levels
        coarse_geom = levels[level].geom
        coarse_k = plane_index_by_level[level]
        covered: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        coarse_origin = levels[level].geom.index_origin
        for fine in range(level + 1, len(levels)):
            fine_k = plane_index_by_level[fine]
            ratio = self._ref_ratio_between(levels, level, fine)
            fine_origin = levels[fine].geom.index_origin
            for box in levels[fine].boxes:
                if fine_k < box.lo[axis_idx] or fine_k > box.hi[axis_idx]:
                    continue
                covered.append(
                    self._coarsen_box(
                        box.lo,
                        box.hi,
                        ratio=ratio,
                        fine_origin=fine_origin,
                        coarse_origin=coarse_origin,
                    )
                )
        if not covered:
            return []
        clamped: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for lo, hi in covered:
            if coarse_k < lo[axis_idx] or coarse_k > hi[axis_idx]:
                continue
            clo = list(lo)
            chi = list(hi)
            clo[axis_idx] = coarse_k
            chi[axis_idx] = coarse_k
            clamped.append((tuple(clo), tuple(chi)))
        return clamped
        return covered

    def _lower_amr_cell_average(self, ctx: LoweringContext):
        ds = ctx.dataset
        axis_idx = _axis_index(self.axis)
        u_axis, v_axis = [i for i in range(3) if i != axis_idx]
        nx, ny = self.resolution
        bpv = _resolve_bytes_per_value(ctx, field=self.field, bytes_per_value=self.bytes_per_value)
        if nx <= 0 or ny <= 0:
            raise ValueError("resolution must be positive")

        levels = ctx.runmeta.steps[ds.step].levels
        base_level = 0
        base_origin = levels[base_level].geom.index_origin
        base_k = self._cell_index(levels[base_level].geom, axis_idx, self.coord)
        plane_index_by_level: dict[int, int] = {}
        for level_idx in range(len(levels)):
            ratio = self._ref_ratio_between(levels, base_level, level_idx)
            origin = levels[level_idx].geom.index_origin
            plane_index_by_level[level_idx] = (base_k - base_origin[axis_idx]) * ratio + origin[axis_idx]
        sum_fields: list[tuple[FieldRef, int]] = []
        area_fields: list[tuple[FieldRef, int]] = []
        stages: list = []
        producer_stage: dict[int, object] = {}

        out_sum_bytes = nx * ny * 8
        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            blocks = list(
                self._intersecting_blocks_level(
                    level_meta, plane_index=plane_index_by_level[level_idx]
                )
            )
            if not blocks:
                continue

            covered_boxes = self._covered_boxes_for_level(
                ctx,
                level=level_idx,
                axis_idx=axis_idx,
                plane_index_by_level=plane_index_by_level,
            )
            covered_payload = [[list(lo), list(hi)] for lo, hi in covered_boxes]

            sum_field = ctx.temp_field(f"{self.out_name}_sum_l{level_idx}")
            area_field = ctx.temp_field(f"{self.out_name}_area_l{level_idx}")

            stage = ctx.stage("uniform_slice")
            for block in blocks:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                stage.map_blocks(
                    name=f"uniform_slice_cellavg_b{block}",
                    kernel="uniform_slice_cellavg_accumulate",
                    domain=dom,
                    inputs=[FieldRef(self.field)],
                    outputs=[sum_field, area_field],
                    output_bytes=[out_sum_bytes, out_sum_bytes],
                    deps={"kind": "None"},
                    params={
                        "axis": axis_idx,
                        "coord": self.coord,
                        "plane_index": plane_index_by_level[level_idx],
                        "rect": list(self.rect),
                        "resolution": [nx, ny],
                        "plane_axes": [u_axis, v_axis],
                        "bytes_per_value": bpv,
                        "covered_boxes": covered_payload,
                    },
                )
            stages.append(stage)
            producer_stage[sum_field.field] = stage
            producer_stage[area_field.field] = stage

            num_inputs = len(blocks)
            fan_in = self._reduce_fan_in(num_inputs)
            input_sum = sum_field
            input_area = area_field
            reduce_idx = 0
            while num_inputs > 1:
                num_groups = (num_inputs + fan_in - 1) // fan_in
                out_sum = sum_field if num_groups == 1 else ctx.temp_field(
                    f"{self.out_name}_sum_reduce_{level_idx}_{reduce_idx}"
                )
                out_area = area_field if num_groups == 1 else ctx.temp_field(
                    f"{self.out_name}_area_reduce_{level_idx}_{reduce_idx}"
                )
                reduce_stage = ctx.stage("uniform_slice_reduce", plane="graph", after=[stages[-1]])
                sum_params = {
                    "graph_kind": "reduce",
                    "fan_in": fan_in,
                    "num_inputs": num_inputs,
                    "input_base": 0,
                    "output_base": 0,
                    "bytes_per_value": 8,
                }
                area_params = {
                    "graph_kind": "reduce",
                    "fan_in": fan_in,
                    "num_inputs": num_inputs,
                    "input_base": 0,
                    "output_base": 0,
                    "bytes_per_value": 8,
                }
                if reduce_idx == 0:
                    sum_params["input_blocks"] = list(blocks)
                    area_params["input_blocks"] = list(blocks)
                reduce_stage.map_blocks(
                    name=f"uniform_slice_sum_reduce_s{reduce_idx}",
                    kernel="uniform_slice_reduce",
                    domain=ctx.domain(step=ds.step, level=level_idx),
                    inputs=[input_sum],
                    outputs=[out_sum],
                    output_bytes=[out_sum_bytes],
                    deps={"kind": "None"},
                    params=sum_params,
                )
                reduce_stage2 = ctx.stage("uniform_slice_reduce", plane="graph", after=[reduce_stage])
                reduce_stage2.map_blocks(
                    name=f"uniform_slice_area_reduce_s{reduce_idx}",
                    kernel="uniform_slice_reduce",
                    domain=ctx.domain(step=ds.step, level=level_idx),
                    inputs=[input_area],
                    outputs=[out_area],
                    output_bytes=[out_sum_bytes],
                    deps={"kind": "None"},
                    params=area_params,
                )
                stages.append(reduce_stage)
                stages.append(reduce_stage2)
                producer_stage[out_sum.field] = reduce_stage
                producer_stage[out_area.field] = reduce_stage2
                input_sum = out_sum
                input_area = out_area
                num_inputs = num_groups
                reduce_idx += 1

            sum_fields.append((input_sum, level_idx))
            area_fields.append((input_area, level_idx))

        if not sum_fields:
            return ctx.fragment([])

        def reduce_pairwise(fields: list[tuple[FieldRef, int]], name: str) -> tuple[FieldRef, int]:
            if len(fields) == 1:
                return fields[0]
            current = fields
            reduce_round = 0
            while len(current) > 1:
                next_fields: list[tuple[FieldRef, int]] = []
                for i in range(0, len(current), 2):
                    if i + 1 >= len(current):
                        next_fields.append(current[i])
                        continue
                    left, left_level = current[i]
                    right, right_level = current[i + 1]
                    left_ref = FieldRef(left.field, version=left.version, domain=ctx.domain(step=ds.step, level=left_level))
                    right_ref = FieldRef(right.field, version=right.version, domain=ctx.domain(step=ds.step, level=right_level))
                    out_field = ctx.temp_field(f"{self.out_name}_{name}_add_{reduce_round}_{i}")
                    deps = [
                        s
                        for s in (
                            producer_stage.get(left.field),
                            producer_stage.get(right.field),
                        )
                        if s is not None
                    ]
                    add_stage = ctx.stage("uniform_slice_add", plane="graph", after=deps)
                    add_stage.map_blocks(
                        name=f"uniform_slice_add_{name}_{reduce_round}_{i}",
                        kernel="uniform_slice_add",
                        domain=ctx.domain(step=ds.step, level=ds.level),
                        inputs=[left_ref, right_ref],
                        outputs=[out_field],
                        output_bytes=[out_sum_bytes],
                        deps={"kind": "None"},
                        params={
                            "graph_kind": "reduce",
                            "fan_in": 1,
                            "num_inputs": 1,
                            "input_base": 0,
                            "output_base": 0,
                            "bytes_per_value": 8,
                        },
                    )
                    stages.append(add_stage)
                    producer_stage[out_field.field] = add_stage
                    next_fields.append((out_field, ds.level))
                current = next_fields
                reduce_round += 1
            return current[0]

        total_sum, total_sum_level = reduce_pairwise(sum_fields, "sum")
        total_area, total_area_level = reduce_pairwise(area_fields, "area")

        out_field = ctx.output_field(self.out_name)
        finalize_deps = [
            s
            for s in (
                producer_stage.get(total_sum.field),
                producer_stage.get(total_area.field),
            )
            if s is not None
        ]
        finalize = ctx.stage("uniform_slice_finalize", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="uniform_slice_finalize",
            kernel="uniform_slice_finalize",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(total_sum.field, version=total_sum.version, domain=ctx.domain(step=ds.step, level=total_sum_level)),
                FieldRef(total_area.field, version=total_area.version, domain=ctx.domain(step=ds.step, level=total_area_level)),
            ],
            outputs=[out_field],
            output_bytes=[nx * ny * bpv],
            deps={"kind": "None"},
            params={
                "graph_kind": "reduce",
                "fan_in": 1,
                "num_inputs": 1,
                "input_base": 0,
                "output_base": 0,
                "bytes_per_value": bpv,
                "pixel_area": abs((self.rect[2] - self.rect[0]) / nx) * abs((self.rect[3] - self.rect[1]) / ny),
            },
        )
        stages.append(finalize)
        producer_stage[out_field.field] = finalize
        return ctx.fragment(stages)

    def lower(self, ctx: LoweringContext):
        return self._lower_amr_cell_average(ctx)


class UniformProjection:
    def __init__(
        self,
        field: int,
        *,
        axis: str | int,
        axis_bounds: tuple[float, float],
        rect: tuple[float, float, float, float],
        resolution: tuple[int, int],
        out_name: str = "projection",
        bytes_per_value: int | None = None,
        reduce_fan_in: Optional[int] = None,
        amr_cell_average: bool = True,
    ) -> None:
        self.field = field
        self.axis = axis
        self.axis_bounds = axis_bounds
        self.rect = rect
        self.resolution = resolution
        self.out_name = out_name
        self.bytes_per_value = bytes_per_value
        self.reduce_fan_in = reduce_fan_in
        self.amr_cell_average = amr_cell_average

    def _intersecting_blocks_level(
        self, level_meta, *, axis_range: tuple[int, int]
    ) -> Iterable[int]:
        axis_idx = _axis_index(self.axis)
        u_axis, v_axis = [i for i in range(3) if i != axis_idx]

        u0, v0, u1, v1 = self.rect
        umin, umax = (u0, u1) if u0 <= u1 else (u1, u0)
        vmin, vmax = (v0, v1) if v0 <= v1 else (v1, v0)
        k_lo, k_hi = axis_range

        geom = level_meta.geom
        for i, box in enumerate(level_meta.boxes):
            if box.hi[axis_idx] < k_lo or box.lo[axis_idx] > k_hi:
                continue

            u0_b, u1_b = _bounds_1d(
                box.lo[u_axis],
                box.hi[u_axis],
                geom.x0[u_axis],
                geom.dx[u_axis],
                geom.index_origin[u_axis],
            )
            v0_b, v1_b = _bounds_1d(
                box.lo[v_axis],
                box.hi[v_axis],
                geom.x0[v_axis],
                geom.dx[v_axis],
                geom.index_origin[v_axis],
            )
            if not (_overlaps_1d(u0_b, u1_b, umin, umax) and _overlaps_1d(v0_b, v1_b, vmin, vmax)):
                continue
            yield i

    def _reduce_fan_in(self, num_inputs: int) -> int:
        if self.reduce_fan_in is None:
            return max(2, int(math.sqrt(num_inputs)))
        return max(1, int(self.reduce_fan_in))

    def _coarsen_box(
        self,
        lo: tuple[int, int, int],
        hi: tuple[int, int, int],
        *,
        ratio: int,
        fine_origin: tuple[int, int, int],
        coarse_origin: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        clo = []
        chi = []
        for d in range(3):
            clo.append(int(math.floor((lo[d] - fine_origin[d]) / ratio)) + coarse_origin[d])
            chi.append(int(math.floor((hi[d] - fine_origin[d] + 1) / ratio)) + coarse_origin[d] - 1)
        return tuple(clo), tuple(chi)

    def _ref_ratio_between(self, levels, coarse: int, fine: int) -> int:
        ratio = 1
        for lev in range(coarse, fine):
            ratio *= int(levels[lev].geom.ref_ratio)
        return ratio

    def _cell_index(self, geom, axis: int, coord: float) -> int:
        x0 = geom.x0[axis]
        dx = geom.dx[axis]
        origin = geom.index_origin[axis]
        if dx == 0.0:
            return origin
        idx_f = (coord - x0) / dx
        return int(math.floor(idx_f)) + origin

    def _axis_index_range(self, geom, axis: int, bounds: tuple[float, float]) -> tuple[int, int]:
        b0, b1 = bounds
        lo = b0 if b0 <= b1 else b1
        hi = b1 if b0 <= b1 else b0
        hi_adj = math.nextafter(hi, -math.inf)
        k_lo = self._cell_index(geom, axis, lo)
        k_hi = self._cell_index(geom, axis, hi_adj)
        return (k_lo, k_hi) if k_lo <= k_hi else (k_hi, k_lo)

    def _covered_boxes_for_level(
        self,
        ctx: LoweringContext,
        *,
        level: int,
        axis_idx: int,
        axis_range_by_level: dict[int, tuple[int, int]],
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        ds = ctx.dataset
        levels = ctx.runmeta.steps[ds.step].levels
        coarse_origin = levels[level].geom.index_origin
        coarse_k_lo, coarse_k_hi = axis_range_by_level[level]
        covered: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for fine in range(level + 1, len(levels)):
            fine_k_lo, fine_k_hi = axis_range_by_level[fine]
            ratio = self._ref_ratio_between(levels, level, fine)
            fine_origin = levels[fine].geom.index_origin
            for box in levels[fine].boxes:
                if box.hi[axis_idx] < fine_k_lo or box.lo[axis_idx] > fine_k_hi:
                    continue
                covered.append(
                    self._coarsen_box(
                        box.lo,
                        box.hi,
                        ratio=ratio,
                        fine_origin=fine_origin,
                        coarse_origin=coarse_origin,
                    )
                )
        if not covered:
            return []
        clamped: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for lo, hi in covered:
            if hi[axis_idx] < coarse_k_lo or lo[axis_idx] > coarse_k_hi:
                continue
            clo = list(lo)
            chi = list(hi)
            clo[axis_idx] = max(clo[axis_idx], coarse_k_lo)
            chi[axis_idx] = min(chi[axis_idx], coarse_k_hi)
            clamped.append((tuple(clo), tuple(chi)))
        return clamped

    def _lower_amr_projection(self, ctx: LoweringContext):
        ds = ctx.dataset
        axis_idx = _axis_index(self.axis)
        nx, ny = self.resolution
        bpv = _resolve_bytes_per_value(ctx, field=self.field, bytes_per_value=self.bytes_per_value)
        if nx <= 0 or ny <= 0:
            raise ValueError("resolution must be positive")

        levels = ctx.runmeta.steps[ds.step].levels
        axis_range_by_level: dict[int, tuple[int, int]] = {}
        for level_idx in range(len(levels)):
            geom = levels[level_idx].geom
            axis_range_by_level[level_idx] = self._axis_index_range(
                geom, axis_idx, self.axis_bounds
            )

        sum_fields: list[tuple[FieldRef, int]] = []
        stages: list = []
        producer_stage: dict[int, object] = {}
        out_sum_bytes = nx * ny * 8

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            axis_range = axis_range_by_level[level_idx]
            blocks = list(self._intersecting_blocks_level(level_meta, axis_range=axis_range))
            if not blocks:
                continue

            covered_boxes = self._covered_boxes_for_level(
                ctx,
                level=level_idx,
                axis_idx=axis_idx,
                axis_range_by_level=axis_range_by_level,
            )
            covered_payload = [[list(lo), list(hi)] for lo, hi in covered_boxes]

            sum_field = ctx.temp_field(f"{self.out_name}_sum_l{level_idx}")

            stage = ctx.stage("uniform_projection")
            for block in blocks:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                stage.map_blocks(
                    name=f"uniform_projection_b{block}",
                    kernel="uniform_projection_accumulate",
                    domain=dom,
                    inputs=[FieldRef(self.field)],
                    outputs=[sum_field],
                    output_bytes=[out_sum_bytes],
                    deps={"kind": "None"},
                    params={
                        "axis": axis_idx,
                        "axis_bounds": [float(self.axis_bounds[0]), float(self.axis_bounds[1])],
                        "rect": list(self.rect),
                        "resolution": [nx, ny],
                        "bytes_per_value": bpv,
                        "covered_boxes": covered_payload,
                    },
                )
            stages.append(stage)
            producer_stage[sum_field.field] = stage

            num_inputs = len(blocks)
            fan_in = self._reduce_fan_in(num_inputs)
            input_sum = sum_field
            reduce_idx = 0
            while num_inputs > 1:
                num_groups = (num_inputs + fan_in - 1) // fan_in
                out_sum = sum_field if num_groups == 1 else ctx.temp_field(
                    f"{self.out_name}_sum_reduce_{level_idx}_{reduce_idx}"
                )
                reduce_stage = ctx.stage("uniform_slice_reduce", plane="graph", after=[stages[-1]])
                sum_params = {
                    "graph_kind": "reduce",
                    "fan_in": fan_in,
                    "num_inputs": num_inputs,
                    "input_base": 0,
                    "output_base": 0,
                    "bytes_per_value": 8,
                }
                if reduce_idx == 0:
                    sum_params["input_blocks"] = list(blocks)
                reduce_stage.map_blocks(
                    name=f"uniform_projection_sum_reduce_s{reduce_idx}",
                    kernel="uniform_slice_reduce",
                    domain=ctx.domain(step=ds.step, level=level_idx),
                    inputs=[input_sum],
                    outputs=[out_sum],
                    output_bytes=[out_sum_bytes],
                    deps={"kind": "None"},
                    params=sum_params,
                )
                stages.append(reduce_stage)
                producer_stage[out_sum.field] = reduce_stage
                input_sum = out_sum
                num_inputs = num_groups
                reduce_idx += 1

            sum_fields.append((input_sum, level_idx))

        if not sum_fields:
            return ctx.fragment([])

        def reduce_pairwise(fields: list[tuple[FieldRef, int]]) -> tuple[FieldRef, int]:
            if len(fields) == 1:
                return fields[0]
            current = fields
            reduce_round = 0
            while len(current) > 1:
                next_fields: list[tuple[FieldRef, int]] = []
                for i in range(0, len(current), 2):
                    if i + 1 >= len(current):
                        next_fields.append(current[i])
                        continue
                    left, left_level = current[i]
                    right, right_level = current[i + 1]
                    left_ref = FieldRef(left.field, version=left.version, domain=ctx.domain(step=ds.step, level=left_level))
                    right_ref = FieldRef(right.field, version=right.version, domain=ctx.domain(step=ds.step, level=right_level))
                    out_field = ctx.temp_field(f"{self.out_name}_sum_add_{reduce_round}_{i}")
                    deps = [
                        s
                        for s in (
                            producer_stage.get(left.field),
                            producer_stage.get(right.field),
                        )
                        if s is not None
                    ]
                    add_stage = ctx.stage("uniform_projection_add", plane="graph", after=deps)
                    add_stage.map_blocks(
                        name=f"uniform_projection_add_{reduce_round}_{i}",
                        kernel="uniform_slice_add",
                        domain=ctx.domain(step=ds.step, level=ds.level),
                        inputs=[left_ref, right_ref],
                        outputs=[out_field],
                        output_bytes=[out_sum_bytes],
                        deps={"kind": "None"},
                        params={
                            "graph_kind": "reduce",
                            "fan_in": 1,
                            "num_inputs": 1,
                            "input_base": 0,
                            "output_base": 0,
                            "bytes_per_value": 8,
                        },
                    )
                    stages.append(add_stage)
                    producer_stage[out_field.field] = add_stage
                    next_fields.append((out_field, ds.level))
                current = next_fields
                reduce_round += 1
            return current[0]

        total_sum, total_sum_level = reduce_pairwise(sum_fields)
        out_field = ctx.output_field(self.out_name)
        finalize_deps = [s for s in (producer_stage.get(total_sum.field),) if s is not None]
        finalize = ctx.stage("uniform_projection_output", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="uniform_projection_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_sum.field,
                    version=total_sum.version,
                    domain=ctx.domain(step=ds.step, level=total_sum_level),
                )
            ],
            outputs=[out_field],
            output_bytes=[out_sum_bytes],
            deps={"kind": "None"},
            params={
                "graph_kind": "reduce",
                "fan_in": 1,
                "num_inputs": 1,
                "input_base": 0,
                "output_base": 0,
                "bytes_per_value": 8,
            },
        )
        stages.append(finalize)
        producer_stage[out_field.field] = finalize
        return ctx.fragment(stages)

    def lower(self, ctx: LoweringContext):
        if not self.amr_cell_average:
            raise ValueError("UniformProjection requires amr_cell_average semantics")
        return self._lower_amr_projection(ctx)


class ParticleCICProjection:
    def __init__(
        self,
        *,
        particle_type: str,
        axis: str | int,
        axis_bounds: tuple[float, float],
        rect: tuple[float, float, float, float],
        resolution: tuple[int, int],
        mass_max: float | None = None,
        out_name: str = "particle_cic_projection",
        reduce_fan_in: Optional[int] = None,
    ) -> None:
        self.particle_type = particle_type
        self.axis = axis
        self.axis_bounds = axis_bounds
        self.rect = rect
        self.resolution = resolution
        self.mass_max = mass_max
        self.out_name = out_name
        self.reduce_fan_in = reduce_fan_in

    def _intersecting_blocks_level(
        self, level_meta, *, axis_range: tuple[int, int]
    ) -> Iterable[int]:
        axis_idx = _axis_index(self.axis)
        u_axis, v_axis = [i for i in range(3) if i != axis_idx]

        u0, v0, u1, v1 = self.rect
        umin, umax = (u0, u1) if u0 <= u1 else (u1, u0)
        vmin, vmax = (v0, v1) if v0 <= v1 else (v1, v0)
        k_lo, k_hi = axis_range

        geom = level_meta.geom
        for i, box in enumerate(level_meta.boxes):
            if box.hi[axis_idx] < k_lo or box.lo[axis_idx] > k_hi:
                continue

            u0_b, u1_b = _bounds_1d(
                box.lo[u_axis],
                box.hi[u_axis],
                geom.x0[u_axis],
                geom.dx[u_axis],
                geom.index_origin[u_axis],
            )
            v0_b, v1_b = _bounds_1d(
                box.lo[v_axis],
                box.hi[v_axis],
                geom.x0[v_axis],
                geom.dx[v_axis],
                geom.index_origin[v_axis],
            )
            if not (_overlaps_1d(u0_b, u1_b, umin, umax) and _overlaps_1d(v0_b, v1_b, vmin, vmax)):
                continue
            yield i

    def _reduce_fan_in(self, num_inputs: int) -> int:
        if self.reduce_fan_in is None:
            return max(2, int(math.sqrt(num_inputs)))
        return max(1, int(self.reduce_fan_in))

    def _ref_ratio_between(self, levels, coarse: int, fine: int) -> int:
        ratio = 1
        for lev in range(coarse, fine):
            ratio *= int(levels[lev].geom.ref_ratio)
        return ratio

    def _cell_index(self, geom, axis: int, coord: float) -> int:
        x0 = geom.x0[axis]
        dx = geom.dx[axis]
        origin = geom.index_origin[axis]
        if dx == 0.0:
            return origin
        idx_f = (coord - x0) / dx
        return int(math.floor(idx_f)) + origin

    def _axis_index_range(self, geom, axis: int, bounds: tuple[float, float]) -> tuple[int, int]:
        b0, b1 = bounds
        lo = b0 if b0 <= b1 else b1
        hi = b1 if b0 <= b1 else b0
        hi_adj = math.nextafter(hi, -math.inf)
        k_lo = self._cell_index(geom, axis, lo)
        k_hi = self._cell_index(geom, axis, hi_adj)
        return (k_lo, k_hi) if k_lo <= k_hi else (k_hi, k_lo)

    def _covered_boxes_for_level(
        self,
        ctx: LoweringContext,
        *,
        level: int,
        axis_idx: int,
        axis_range_by_level: dict[int, tuple[int, int]],
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        ds = ctx.dataset
        levels = ctx.runmeta.steps[ds.step].levels
        coarse_origin = levels[level].geom.index_origin
        coarse_k_lo, coarse_k_hi = axis_range_by_level[level]
        covered: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for fine in range(level + 1, len(levels)):
            fine_k_lo, fine_k_hi = axis_range_by_level[fine]
            ratio = self._ref_ratio_between(levels, level, fine)
            fine_origin = levels[fine].geom.index_origin
            for box in levels[fine].boxes:
                if box.hi[axis_idx] < fine_k_lo or box.lo[axis_idx] > fine_k_hi:
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
        if not covered:
            return []
        clamped: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for lo, hi in covered:
            if hi[axis_idx] < coarse_k_lo or lo[axis_idx] > coarse_k_hi:
                continue
            clo = list(lo)
            chi = list(hi)
            clo[axis_idx] = max(clo[axis_idx], coarse_k_lo)
            chi[axis_idx] = min(chi[axis_idx], coarse_k_hi)
            clamped.append((tuple(clo), tuple(chi)))
        return clamped

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        axis_idx = _axis_index(self.axis)
        nx, ny = self.resolution
        if nx <= 0 or ny <= 0:
            raise ValueError("resolution must be positive")

        levels = ctx.runmeta.steps[ds.step].levels
        axis_range_by_level: dict[int, tuple[int, int]] = {}
        for level_idx in range(len(levels)):
            geom = levels[level_idx].geom
            axis_range_by_level[level_idx] = self._axis_index_range(
                geom, axis_idx, self.axis_bounds
            )

        sum_fields: list[tuple[FieldRef, int]] = []
        stages: list = []
        producer_stage: dict[int, object] = {}
        out_sum_bytes = nx * ny * 8

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            axis_range = axis_range_by_level[level_idx]
            blocks = list(self._intersecting_blocks_level(level_meta, axis_range=axis_range))
            if not blocks:
                continue

            covered_boxes = self._covered_boxes_for_level(
                ctx,
                level=level_idx,
                axis_idx=axis_idx,
                axis_range_by_level=axis_range_by_level,
            )
            covered_payload = [[list(lo), list(hi)] for lo, hi in covered_boxes]

            sum_field = ctx.temp_field(f"{self.out_name}_sum_l{level_idx}")

            stage = ctx.stage("particle_cic_projection")
            for block in blocks:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                params = {
                    "particle_type": self.particle_type,
                    "axis": axis_idx,
                    "axis_bounds": [float(self.axis_bounds[0]), float(self.axis_bounds[1])],
                    "rect": list(self.rect),
                    "resolution": [nx, ny],
                    "covered_boxes": covered_payload,
                }
                if self.mass_max is not None:
                    params["mass_max"] = float(self.mass_max)
                stage.map_blocks(
                    name=f"particle_cic_projection_b{block}",
                    kernel="particle_cic_projection_accumulate",
                    domain=dom,
                    inputs=[],
                    outputs=[sum_field],
                    output_bytes=[out_sum_bytes],
                    deps={"kind": "None"},
                    params=params,
                )
            stages.append(stage)
            producer_stage[sum_field.field] = stage

            num_inputs = len(blocks)
            fan_in = self._reduce_fan_in(num_inputs)
            input_sum = sum_field
            reduce_idx = 0
            while num_inputs > 1:
                num_groups = (num_inputs + fan_in - 1) // fan_in
                out_sum = sum_field if num_groups == 1 else ctx.temp_field(
                    f"{self.out_name}_sum_reduce_{level_idx}_{reduce_idx}"
                )
                reduce_stage = ctx.stage("particle_cic_projection_reduce", plane="graph", after=[stages[-1]])
                sum_params = {
                    "graph_kind": "reduce",
                    "fan_in": fan_in,
                    "num_inputs": num_inputs,
                    "input_base": 0,
                    "output_base": 0,
                    "bytes_per_value": 8,
                }
                if reduce_idx == 0:
                    sum_params["input_blocks"] = list(blocks)
                reduce_stage.map_blocks(
                    name=f"particle_cic_projection_sum_reduce_s{reduce_idx}",
                    kernel="uniform_slice_reduce",
                    domain=ctx.domain(step=ds.step, level=level_idx),
                    inputs=[input_sum],
                    outputs=[out_sum],
                    output_bytes=[out_sum_bytes],
                    deps={"kind": "None"},
                    params=sum_params,
                )
                stages.append(reduce_stage)
                producer_stage[out_sum.field] = reduce_stage
                input_sum = out_sum
                num_inputs = num_groups
                reduce_idx += 1

            sum_fields.append((input_sum, level_idx))

        if not sum_fields:
            return ctx.fragment([])

        def reduce_pairwise(fields: list[tuple[FieldRef, int]]) -> tuple[FieldRef, int]:
            if len(fields) == 1:
                return fields[0]
            current = fields
            reduce_round = 0
            while len(current) > 1:
                next_fields: list[tuple[FieldRef, int]] = []
                for i in range(0, len(current), 2):
                    if i + 1 >= len(current):
                        next_fields.append(current[i])
                        continue
                    left, left_level = current[i]
                    right, right_level = current[i + 1]
                    left_ref = FieldRef(left.field, version=left.version, domain=ctx.domain(step=ds.step, level=left_level))
                    right_ref = FieldRef(right.field, version=right.version, domain=ctx.domain(step=ds.step, level=right_level))
                    out_field = ctx.temp_field(f"{self.out_name}_sum_add_{reduce_round}_{i}")
                    deps = [
                        s
                        for s in (
                            producer_stage.get(left.field),
                            producer_stage.get(right.field),
                        )
                        if s is not None
                    ]
                    add_stage = ctx.stage("particle_cic_projection_add", plane="graph", after=deps)
                    add_stage.map_blocks(
                        name=f"particle_cic_projection_add_{reduce_round}_{i}",
                        kernel="uniform_slice_add",
                        domain=ctx.domain(step=ds.step, level=ds.level),
                        inputs=[left_ref, right_ref],
                        outputs=[out_field],
                        output_bytes=[out_sum_bytes],
                        deps={"kind": "None"},
                        params={
                            "graph_kind": "reduce",
                            "fan_in": 1,
                            "num_inputs": 1,
                            "input_base": 0,
                            "output_base": 0,
                            "bytes_per_value": 8,
                        },
                    )
                    stages.append(add_stage)
                    producer_stage[out_field.field] = add_stage
                    next_fields.append((out_field, ds.level))
                current = next_fields
                reduce_round += 1
            return current[0]

        total_sum, total_sum_level = reduce_pairwise(sum_fields)
        out_field = ctx.output_field(self.out_name)
        finalize_deps = [s for s in (producer_stage.get(total_sum.field),) if s is not None]
        finalize = ctx.stage("particle_cic_projection_output", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="particle_cic_projection_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_sum.field,
                    version=total_sum.version,
                    domain=ctx.domain(step=ds.step, level=total_sum_level),
                )
            ],
            outputs=[out_field],
            output_bytes=[out_sum_bytes],
            deps={"kind": "None"},
            params={
                "graph_kind": "reduce",
                "fan_in": 1,
                "num_inputs": 1,
                "input_base": 0,
                "output_base": 0,
                "bytes_per_value": 8,
            },
        )
        stages.append(finalize)
        return ctx.fragment(stages)


class ParticleCICGrid:
    def __init__(
        self,
        *,
        particle_type: str,
        axis: str | int,
        axis_bounds: tuple[float, float],
        rect: tuple[float, float, float, float],
        mass_max: float | None = None,
        out_name: str = "particle_cic_grid",
    ) -> None:
        self.particle_type = particle_type
        self.axis = axis
        self.axis_bounds = axis_bounds
        self.rect = rect
        self.mass_max = mass_max
        self.out_name = out_name

    def _intersecting_blocks_level(
        self, level_meta, *, axis_range: tuple[int, int]
    ) -> Iterable[int]:
        axis_idx = _axis_index(self.axis)
        u_axis, v_axis = [i for i in range(3) if i != axis_idx]

        u0, v0, u1, v1 = self.rect
        umin, umax = (u0, u1) if u0 <= u1 else (u1, u0)
        vmin, vmax = (v0, v1) if v0 <= v1 else (v1, v0)
        k_lo, k_hi = axis_range

        geom = level_meta.geom
        for i, box in enumerate(level_meta.boxes):
            if box.hi[axis_idx] < k_lo or box.lo[axis_idx] > k_hi:
                continue

            u0_b, u1_b = _bounds_1d(
                box.lo[u_axis],
                box.hi[u_axis],
                geom.x0[u_axis],
                geom.dx[u_axis],
                geom.index_origin[u_axis],
            )
            v0_b, v1_b = _bounds_1d(
                box.lo[v_axis],
                box.hi[v_axis],
                geom.x0[v_axis],
                geom.dx[v_axis],
                geom.index_origin[v_axis],
            )
            if not (_overlaps_1d(u0_b, u1_b, umin, umax) and _overlaps_1d(v0_b, v1_b, vmin, vmax)):
                continue
            yield i

    def _ref_ratio_between(self, levels, coarse: int, fine: int) -> int:
        ratio = 1
        for lev in range(coarse, fine):
            ratio *= int(levels[lev].geom.ref_ratio)
        return ratio

    def _cell_index(self, geom, axis: int, coord: float) -> int:
        x0 = geom.x0[axis]
        dx = geom.dx[axis]
        origin = geom.index_origin[axis]
        if dx == 0.0:
            return origin
        idx_f = (coord - x0) / dx
        return int(math.floor(idx_f)) + origin

    def _axis_index_range(self, geom, axis: int, bounds: tuple[float, float]) -> tuple[int, int]:
        b0, b1 = bounds
        lo = b0 if b0 <= b1 else b1
        hi = b1 if b0 <= b1 else b0
        hi_adj = math.nextafter(hi, -math.inf)
        k_lo = self._cell_index(geom, axis, lo)
        k_hi = self._cell_index(geom, axis, hi_adj)
        return (k_lo, k_hi) if k_lo <= k_hi else (k_hi, k_lo)

    def _covered_boxes_for_level(
        self,
        ctx: LoweringContext,
        *,
        level: int,
        axis_idx: int,
        axis_range_by_level: dict[int, tuple[int, int]],
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        ds = ctx.dataset
        levels = ctx.runmeta.steps[ds.step].levels
        coarse_origin = levels[level].geom.index_origin
        coarse_k_lo, coarse_k_hi = axis_range_by_level[level]
        covered: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for fine in range(level + 1, len(levels)):
            fine_k_lo, fine_k_hi = axis_range_by_level[fine]
            ratio = self._ref_ratio_between(levels, level, fine)
            fine_origin = levels[fine].geom.index_origin
            for box in levels[fine].boxes:
                if box.hi[axis_idx] < fine_k_lo or box.lo[axis_idx] > fine_k_hi:
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
        if not covered:
            return []
        clamped: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for lo, hi in covered:
            if hi[axis_idx] < coarse_k_lo or lo[axis_idx] > coarse_k_hi:
                continue
            clo = list(lo)
            chi = list(hi)
            clo[axis_idx] = max(clo[axis_idx], coarse_k_lo)
            chi[axis_idx] = min(chi[axis_idx], coarse_k_hi)
            clamped.append((tuple(clo), tuple(chi)))
        return clamped

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        axis_idx = _axis_index(self.axis)

        levels = ctx.runmeta.steps[ds.step].levels
        axis_range_by_level: dict[int, tuple[int, int]] = {}
        for level_idx in range(len(levels)):
            geom = levels[level_idx].geom
            axis_range_by_level[level_idx] = self._axis_index_range(
                geom, axis_idx, self.axis_bounds
            )

        stages: list = []
        grid_field = ctx.temp_field(self.out_name)

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            axis_range = axis_range_by_level[level_idx]
            blocks = list(self._intersecting_blocks_level(level_meta, axis_range=axis_range))
            if not blocks:
                continue

            covered_boxes = self._covered_boxes_for_level(
                ctx,
                level=level_idx,
                axis_idx=axis_idx,
                axis_range_by_level=axis_range_by_level,
            )
            covered_payload = [[list(lo), list(hi)] for lo, hi in covered_boxes]

            stage = ctx.stage("particle_cic_grid")
            for block in blocks:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                params = {
                    "particle_type": self.particle_type,
                    "axis": axis_idx,
                    "axis_bounds": [float(self.axis_bounds[0]), float(self.axis_bounds[1])],
                    "covered_boxes": covered_payload,
                }
                if self.mass_max is not None:
                    params["mass_max"] = float(self.mass_max)
                stage.map_blocks(
                    name=f"particle_cic_grid_b{block}",
                    kernel="particle_cic_grid_accumulate",
                    domain=dom,
                    inputs=[],
                    outputs=[grid_field],
                    output_bytes=[0],
                    deps={"kind": "None"},
                    params=params,
                )
            stages.append(stage)

        if not stages:
            return ctx.fragment([])

        return ctx.fragment(stages)


class Histogram1D:
    def __init__(
        self,
        field: int,
        *,
        hist_range: tuple[float, float],
        bins: int,
        out_name: str = "histogram1d",
        weight_field: int | None = None,
        bytes_per_value: int | None = None,
        reduce_fan_in: Optional[int] = None,
    ) -> None:
        self.field = field
        self.hist_range = hist_range
        self.bins = bins
        self.out_name = out_name
        self.weight_field = weight_field
        self.bytes_per_value = bytes_per_value
        self.reduce_fan_in = reduce_fan_in

    def _reduce_fan_in(self, num_inputs: int) -> int:
        if self.reduce_fan_in is None:
            return max(2, int(math.sqrt(num_inputs)))
        return max(1, int(self.reduce_fan_in))

    def _ref_ratio_between(self, levels, coarse: int, fine: int) -> int:
        ratio = 1
        for lev in range(coarse, fine):
            ratio *= int(levels[lev].geom.ref_ratio)
        return ratio

    def _covered_boxes_for_level(
        self,
        ctx: LoweringContext,
        *,
        level: int,
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        ds = ctx.dataset
        levels = ctx.runmeta.steps[ds.step].levels
        coarse_origin = levels[level].geom.index_origin
        covered: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for fine in range(level + 1, len(levels)):
            ratio = self._ref_ratio_between(levels, level, fine)
            fine_origin = levels[fine].geom.index_origin
            for box in levels[fine].boxes:
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

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        if self.bins <= 0:
            raise ValueError("bins must be positive")
        lo, hi = self.hist_range
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError("hist_range must be finite and increasing")

        bpv = _resolve_bytes_per_value(ctx, field=self.field, bytes_per_value=self.bytes_per_value)
        levels = ctx.runmeta.steps[ds.step].levels
        out_bytes = self.bins * 8

        hist_fields: list[tuple[FieldRef, int]] = []
        stages: list = []
        producer_stage: dict[int, object] = {}

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            blocks = list(range(len(level_meta.boxes)))
            if not blocks:
                continue

            covered_boxes = self._covered_boxes_for_level(ctx, level=level_idx)
            covered_payload = [[list(c_lo), list(c_hi)] for c_lo, c_hi in covered_boxes]
            hist_field = ctx.temp_field(f"{self.out_name}_sum_l{level_idx}")

            stage = ctx.stage("histogram1d")
            for block in blocks:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                inputs = [FieldRef(self.field)]
                if self.weight_field is not None:
                    inputs.append(FieldRef(self.weight_field))
                stage.map_blocks(
                    name=f"histogram1d_b{block}",
                    kernel="histogram1d_accumulate",
                    domain=dom,
                    inputs=inputs,
                    outputs=[hist_field],
                    output_bytes=[out_bytes],
                    deps={"kind": "None"},
                    params={
                        "range": [float(lo), float(hi)],
                        "bins": int(self.bins),
                        "bytes_per_value": int(bpv),
                        "covered_boxes": covered_payload,
                    },
                )
            stages.append(stage)
            producer_stage[hist_field.field] = stage

            num_inputs = len(blocks)
            fan_in = self._reduce_fan_in(num_inputs)
            input_hist = hist_field
            reduce_idx = 0
            while num_inputs > 1:
                num_groups = (num_inputs + fan_in - 1) // fan_in
                out_hist = hist_field if num_groups == 1 else ctx.temp_field(
                    f"{self.out_name}_sum_reduce_{level_idx}_{reduce_idx}"
                )
                reduce_stage = ctx.stage("histogram1d_reduce", plane="graph", after=[stages[-1]])
                reduce_params = {
                    "graph_kind": "reduce",
                    "fan_in": fan_in,
                    "num_inputs": num_inputs,
                    "input_base": 0,
                    "output_base": 0,
                    "bytes_per_value": 8,
                }
                if reduce_idx == 0:
                    reduce_params["input_blocks"] = list(blocks)
                reduce_stage.map_blocks(
                    name=f"histogram1d_reduce_s{reduce_idx}",
                    kernel="uniform_slice_reduce",
                    domain=ctx.domain(step=ds.step, level=level_idx),
                    inputs=[input_hist],
                    outputs=[out_hist],
                    output_bytes=[out_bytes],
                    deps={"kind": "None"},
                    params=reduce_params,
                )
                stages.append(reduce_stage)
                producer_stage[out_hist.field] = reduce_stage
                input_hist = out_hist
                num_inputs = num_groups
                reduce_idx += 1

            hist_fields.append((input_hist, level_idx))

        if not hist_fields:
            return ctx.fragment([])

        def reduce_pairwise(fields: list[tuple[FieldRef, int]]) -> tuple[FieldRef, int]:
            if len(fields) == 1:
                return fields[0]
            current = fields
            reduce_round = 0
            while len(current) > 1:
                next_fields: list[tuple[FieldRef, int]] = []
                for i in range(0, len(current), 2):
                    if i + 1 >= len(current):
                        next_fields.append(current[i])
                        continue
                    left, left_level = current[i]
                    right, right_level = current[i + 1]
                    left_ref = FieldRef(
                        left.field,
                        version=left.version,
                        domain=ctx.domain(step=ds.step, level=left_level),
                    )
                    right_ref = FieldRef(
                        right.field,
                        version=right.version,
                        domain=ctx.domain(step=ds.step, level=right_level),
                    )
                    out_field = ctx.temp_field(f"{self.out_name}_sum_add_{reduce_round}_{i}")
                    deps = [
                        s
                        for s in (
                            producer_stage.get(left.field),
                            producer_stage.get(right.field),
                        )
                        if s is not None
                    ]
                    add_stage = ctx.stage("histogram1d_add", plane="graph", after=deps)
                    add_stage.map_blocks(
                        name=f"histogram1d_add_{reduce_round}_{i}",
                        kernel="uniform_slice_add",
                        domain=ctx.domain(step=ds.step, level=ds.level),
                        inputs=[left_ref, right_ref],
                        outputs=[out_field],
                        output_bytes=[out_bytes],
                        deps={"kind": "None"},
                        params={
                            "graph_kind": "reduce",
                            "fan_in": 1,
                            "num_inputs": 1,
                            "input_base": 0,
                            "output_base": 0,
                            "bytes_per_value": 8,
                        },
                    )
                    stages.append(add_stage)
                    producer_stage[out_field.field] = add_stage
                    next_fields.append((out_field, ds.level))
                current = next_fields
                reduce_round += 1
            return current[0]

        total_hist, total_hist_level = reduce_pairwise(hist_fields)
        out_field = ctx.output_field(self.out_name)
        finalize_deps = [s for s in (producer_stage.get(total_hist.field),) if s is not None]
        finalize = ctx.stage("histogram1d_output", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="histogram1d_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_hist.field,
                    version=total_hist.version,
                    domain=ctx.domain(step=ds.step, level=total_hist_level),
                )
            ],
            outputs=[out_field],
            output_bytes=[out_bytes],
            deps={"kind": "None"},
            params={
                "graph_kind": "reduce",
                "fan_in": 1,
                "num_inputs": 1,
                "input_base": 0,
                "output_base": 0,
                "bytes_per_value": 8,
            },
        )
        stages.append(finalize)
        return ctx.fragment(stages)


class Histogram2D:
    def __init__(
        self,
        x_field: int,
        y_field: int,
        *,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        bins: tuple[int, int],
        out_name: str = "histogram2d",
        weight_field: int | None = None,
        weight_mode: str = "input",
        bytes_per_value: int | None = None,
        reduce_fan_in: Optional[int] = None,
    ) -> None:
        self.x_field = x_field
        self.y_field = y_field
        self.x_range = x_range
        self.y_range = y_range
        self.bins = bins
        self.out_name = out_name
        self.weight_field = weight_field
        self.weight_mode = weight_mode
        self.bytes_per_value = bytes_per_value
        self.reduce_fan_in = reduce_fan_in

    def _reduce_fan_in(self, num_inputs: int) -> int:
        if self.reduce_fan_in is None:
            return max(2, int(math.sqrt(num_inputs)))
        return max(1, int(self.reduce_fan_in))

    def _ref_ratio_between(self, levels, coarse: int, fine: int) -> int:
        ratio = 1
        for lev in range(coarse, fine):
            ratio *= int(levels[lev].geom.ref_ratio)
        return ratio

    def _covered_boxes_for_level(
        self,
        ctx: LoweringContext,
        *,
        level: int,
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        ds = ctx.dataset
        levels = ctx.runmeta.steps[ds.step].levels
        coarse_origin = levels[level].geom.index_origin
        covered: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for fine in range(level + 1, len(levels)):
            ratio = self._ref_ratio_between(levels, level, fine)
            fine_origin = levels[fine].geom.index_origin
            for box in levels[fine].boxes:
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

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        nx, ny = self.bins
        if nx <= 0 or ny <= 0:
            raise ValueError("bins must be positive")
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        if not (math.isfinite(x0) and math.isfinite(x1) and math.isfinite(y0) and math.isfinite(y1)):
            raise ValueError("histogram ranges must be finite")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("histogram ranges must be increasing")

        bpv = _resolve_bytes_per_value(ctx, field=self.x_field, bytes_per_value=self.bytes_per_value)
        levels = ctx.runmeta.steps[ds.step].levels
        out_bytes = nx * ny * 8

        hist_fields: list[tuple[FieldRef, int]] = []
        stages: list = []
        producer_stage: dict[int, object] = {}

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            blocks = list(range(len(level_meta.boxes)))
            if not blocks:
                continue

            covered_boxes = self._covered_boxes_for_level(ctx, level=level_idx)
            covered_payload = [[list(c_lo), list(c_hi)] for c_lo, c_hi in covered_boxes]
            hist_field = ctx.temp_field(f"{self.out_name}_sum_l{level_idx}")

            stage = ctx.stage("histogram2d")
            for block in blocks:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                inputs = [FieldRef(self.x_field), FieldRef(self.y_field)]
                if self.weight_field is not None:
                    inputs.append(FieldRef(self.weight_field))
                stage.map_blocks(
                    name=f"histogram2d_b{block}",
                    kernel="histogram2d_accumulate",
                    domain=dom,
                    inputs=inputs,
                    outputs=[hist_field],
                    output_bytes=[out_bytes],
                    deps={"kind": "None"},
                    params={
                        "x_range": [float(x0), float(x1)],
                        "y_range": [float(y0), float(y1)],
                        "bins": [int(nx), int(ny)],
                        "bytes_per_value": int(bpv),
                        "weight_mode": self.weight_mode,
                        "covered_boxes": covered_payload,
                    },
                )
            stages.append(stage)
            producer_stage[hist_field.field] = stage

            num_inputs = len(blocks)
            fan_in = self._reduce_fan_in(num_inputs)
            input_hist = hist_field
            reduce_idx = 0
            while num_inputs > 1:
                num_groups = (num_inputs + fan_in - 1) // fan_in
                out_hist = hist_field if num_groups == 1 else ctx.temp_field(
                    f"{self.out_name}_sum_reduce_{level_idx}_{reduce_idx}"
                )
                reduce_stage = ctx.stage("histogram2d_reduce", plane="graph", after=[stages[-1]])
                reduce_params = {
                    "graph_kind": "reduce",
                    "fan_in": fan_in,
                    "num_inputs": num_inputs,
                    "input_base": 0,
                    "output_base": 0,
                    "bytes_per_value": 8,
                }
                if reduce_idx == 0:
                    reduce_params["input_blocks"] = list(blocks)
                reduce_stage.map_blocks(
                    name=f"histogram2d_reduce_s{reduce_idx}",
                    kernel="uniform_slice_reduce",
                    domain=ctx.domain(step=ds.step, level=level_idx),
                    inputs=[input_hist],
                    outputs=[out_hist],
                    output_bytes=[out_bytes],
                    deps={"kind": "None"},
                    params=reduce_params,
                )
                stages.append(reduce_stage)
                producer_stage[out_hist.field] = reduce_stage
                input_hist = out_hist
                num_inputs = num_groups
                reduce_idx += 1

            hist_fields.append((input_hist, level_idx))

        if not hist_fields:
            return ctx.fragment([])

        def reduce_pairwise(fields: list[tuple[FieldRef, int]]) -> tuple[FieldRef, int]:
            if len(fields) == 1:
                return fields[0]
            current = fields
            reduce_round = 0
            while len(current) > 1:
                next_fields: list[tuple[FieldRef, int]] = []
                for i in range(0, len(current), 2):
                    if i + 1 >= len(current):
                        next_fields.append(current[i])
                        continue
                    left, left_level = current[i]
                    right, right_level = current[i + 1]
                    left_ref = FieldRef(
                        left.field,
                        version=left.version,
                        domain=ctx.domain(step=ds.step, level=left_level),
                    )
                    right_ref = FieldRef(
                        right.field,
                        version=right.version,
                        domain=ctx.domain(step=ds.step, level=right_level),
                    )
                    out_field = ctx.temp_field(f"{self.out_name}_sum_add_{reduce_round}_{i}")
                    deps = [
                        s
                        for s in (
                            producer_stage.get(left.field),
                            producer_stage.get(right.field),
                        )
                        if s is not None
                    ]
                    add_stage = ctx.stage("histogram2d_add", plane="graph", after=deps)
                    add_stage.map_blocks(
                        name=f"histogram2d_add_{reduce_round}_{i}",
                        kernel="uniform_slice_add",
                        domain=ctx.domain(step=ds.step, level=ds.level),
                        inputs=[left_ref, right_ref],
                        outputs=[out_field],
                        output_bytes=[out_bytes],
                        deps={"kind": "None"},
                        params={
                            "graph_kind": "reduce",
                            "fan_in": 1,
                            "num_inputs": 1,
                            "input_base": 0,
                            "output_base": 0,
                            "bytes_per_value": 8,
                        },
                    )
                    stages.append(add_stage)
                    producer_stage[out_field.field] = add_stage
                    next_fields.append((out_field, ds.level))
                current = next_fields
                reduce_round += 1
            return current[0]

        total_hist, total_hist_level = reduce_pairwise(hist_fields)
        out_field = ctx.output_field(self.out_name)
        finalize_deps = [s for s in (producer_stage.get(total_hist.field),) if s is not None]
        finalize = ctx.stage("histogram2d_output", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="histogram2d_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_hist.field,
                    version=total_hist.version,
                    domain=ctx.domain(step=ds.step, level=total_hist_level),
                )
            ],
            outputs=[out_field],
            output_bytes=[out_bytes],
            deps={"kind": "None"},
            params={
                "graph_kind": "reduce",
                "fan_in": 1,
                "num_inputs": 1,
                "input_base": 0,
                "output_base": 0,
                "bytes_per_value": 8,
            },
        )
        stages.append(finalize)
        return ctx.fragment(stages)


def histogram_edges_1d(hist_range: tuple[float, float], bins: int) -> list[float]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    lo, hi = hist_range
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("hist_range bounds must be finite")
    if hi <= lo:
        raise ValueError("hist_range must be increasing")
    dx = (hi - lo) / bins
    return [lo + i * dx for i in range(bins + 1)]


def histogram_edges_2d(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bins: tuple[int, int],
) -> tuple[list[float], list[float]]:
    nx, ny = bins
    return histogram_edges_1d(x_range, nx), histogram_edges_1d(y_range, ny)


def cdf_from_histogram(counts: Sequence[float], *, normalize: bool = True) -> list[float]:
    total = float(sum(float(v) for v in counts))
    out: list[float] = []
    accum = 0.0
    for value in counts:
        accum += float(value)
        out.append(accum)
    if normalize and total > 0.0:
        out = [v / total for v in out]
    return out


def cdf_from_samples(samples: Sequence[float]) -> tuple[list[float], list[float]]:
    if not samples:
        return [], []
    xs = sorted(float(v) for v in samples)
    n = len(xs)
    cdf = [(i + 1) / n for i in range(n)]
    return xs, cdf
