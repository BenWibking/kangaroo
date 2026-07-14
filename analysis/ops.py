from __future__ import annotations

import math
from typing import Optional, Sequence

from .amr_coverage import (
    axis_ranges_by_level,
    covered_plane_boxes,
    covered_slab_boxes,
    covered_volume_boxes,
    intersecting_plane_blocks,
    intersecting_slab_blocks,
    plane_indices_by_level,
)
from .buffer import (
    BlockShape,
    BufferSpec,
    DType,
    DynamicShape,
    DynamicUpperBound,
    FixedShape,
    InitPolicy,
)
from .ctx import LoweringContext
from .kernel_params import (
    AmrSubboxPackParams,
    CylindricalFluxParams,
    FluxSurfaceParams,
    GradStencilParams,
    Histogram1DParams,
    Histogram2DParams,
    ParticleCicGridParams,
    ParticleCicProjectionParams,
    SliceFinalizeParams,
    ToomreProfileParams,
    UniformProjectionParams,
    UniformSliceCellParams,
)
from .plan import DependencyRule, FieldRef, GraphReduceSpec, OutputRef
from .reduction import (
    GraphReductionBuilder,
    ReducedField,
    resolve_reduce_fan_in,
)


def _fixed_f64(field: FieldRef, elements: int | float) -> OutputRef:
    return OutputRef(
        field,
        BufferSpec(DType.F64, FixedShape((int(elements),)), InitPolicy.ZERO),
    )


def _fixed_f64_shape(field: FieldRef, extents: tuple[int, ...]) -> OutputRef:
    return OutputRef(
        field,
        BufferSpec(DType.F64, FixedShape(extents), InitPolicy.ZERO),
    )


def _fixed_real(field: FieldRef, elements: int, dtype: DType) -> OutputRef:
    return OutputRef(field, BufferSpec(dtype, FixedShape((int(elements),))))


def _fixed_real_shape(field: FieldRef, extents: tuple[int, ...], dtype: DType) -> OutputRef:
    return OutputRef(field, BufferSpec(dtype, FixedShape(extents)))


def _block_f64(field: FieldRef, components: int = 1) -> OutputRef:
    return OutputRef(field, BufferSpec(DType.F64, BlockShape(components), InitPolicy.ZERO))


def _amr_patch_payload(field: FieldRef) -> OutputRef:
    return OutputRef(
        field,
        BufferSpec(
            DType.OPAQUE,
            DynamicShape(DynamicUpperBound.amr_subbox_pack()),
        ),
    )


def _covered_boxes_payload(boxes) -> tuple:
    return tuple((tuple(lo), tuple(hi)) for lo, hi in boxes)


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
                outputs=[_amr_patch_payload(fetch_f)],
                deps=DependencyRule(),
                params=AmrSubboxPackParams(
                    input_field=field_id,
                    input_step=ds.step,
                    input_level=ds.level,
                    halo_cells=self.stencil_radius,
                ),
            )
            grad_f = ctx.temp_field(f"gradU_{comp}")
            grad_fields.append(grad_f)
            s1.map_blocks(
                name=f"gradU_{comp}",
                kernel="gradU_stencil",
                domain=dom,
                inputs=[FieldRef(field_id), fetch_f],
                outputs=[_block_f64(grad_f, 3)],
                deps=DependencyRule(),
                params=GradStencilParams(
                    input_field=field_id,
                    input_step=ds.step,
                    input_level=ds.level,
                    stencil_radius=self.stencil_radius,
                ),
            )

        s2 = ctx.stage("vortmag", after=[s1])
        s2.map_blocks(
            name="vortmag",
            kernel="vorticity_mag",
            domain=dom,
            inputs=grad_fields,
            outputs=[_block_f64(vort)],
            deps=DependencyRule(),
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


def _resolve_real_dtype(dtype: DType | str) -> DType:
    resolved = DType(dtype)
    if resolved not in (DType.F32, DType.F64):
        raise ValueError("numeric grid outputs require dtype='f32' or dtype='f64'")
    return resolved


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
        dtype: DType | str = DType.F64,
        reduce_fan_in: Optional[int] = None,
    ) -> None:
        self.field = field
        self.axis = axis
        self.coord = coord
        self.rect = rect
        self.resolution = resolution
        self.out_name = out_name
        self.dtype = _resolve_real_dtype(dtype)
        self.reduce_fan_in = reduce_fan_in

    def _lower_amr_cell_average(self, ctx: LoweringContext):
        ds = ctx.dataset
        axis_idx = _axis_index(self.axis)
        u_axis, v_axis = [i for i in range(3) if i != axis_idx]
        nx, ny = self.resolution
        dtype = self.dtype
        if nx <= 0 or ny <= 0:
            raise ValueError("resolution must be positive")

        levels = ctx.runmeta.steps[ds.step].levels
        plane_index_by_level = plane_indices_by_level(
            levels, axis=axis_idx, coord=self.coord
        )
        sum_fields: list[ReducedField] = []
        area_fields: list[ReducedField] = []
        reductions = GraphReductionBuilder(ctx)

        out_shape = (ny, nx)
        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            blocks = list(
                intersecting_plane_blocks(
                    level_meta,
                    axis=axis_idx,
                    plane_index=plane_index_by_level[level_idx],
                    rect=self.rect,
                )
            )
            if not blocks:
                continue

            covered_boxes = covered_plane_boxes(
                levels,
                level=level_idx,
                axis=axis_idx,
                plane_indices=plane_index_by_level,
            )
            covered_payload = _covered_boxes_payload(covered_boxes)

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
                    outputs=[_fixed_f64_shape(sum_field, out_shape), _fixed_f64_shape(area_field, out_shape)],
                    deps=DependencyRule(),
                    params=UniformSliceCellParams(
                        axis=axis_idx,
                        coord=self.coord,
                        plane_index=plane_index_by_level[level_idx],
                        rect=tuple(self.rect),
                        resolution=(nx, ny),
                        covered_boxes=covered_payload,
                    ),
                )
            reductions.add_stage(stage, outputs=[sum_field, area_field])
            output_buffer = BufferSpec(
                DType.F64, FixedShape(out_shape), InitPolicy.ZERO
            )
            reduced_sum = reductions.reduce_blocks(
                value=ReducedField(sum_field, level_idx),
                input_blocks=blocks,
                step=ds.step,
                fan_in=resolve_reduce_fan_in(self.reduce_fan_in, len(blocks)),
                kernel="uniform_slice_reduce",
                output_buffer=output_buffer,
                stage_name="uniform_slice_reduce",
                template_name="uniform_slice_sum_reduce_s{round}",
                temporary_name=f"{self.out_name}_sum_reduce_{level_idx}_{{round}}",
                after=stage,
                normalize_single=True,
            )
            reduced_area = reductions.reduce_blocks(
                value=ReducedField(area_field, level_idx),
                input_blocks=blocks,
                step=ds.step,
                fan_in=resolve_reduce_fan_in(self.reduce_fan_in, len(blocks)),
                kernel="uniform_slice_reduce",
                output_buffer=output_buffer,
                stage_name="uniform_slice_reduce",
                template_name="uniform_slice_area_reduce_s{round}",
                temporary_name=f"{self.out_name}_area_reduce_{level_idx}_{{round}}",
                after=reductions.producer(reduced_sum.field) or stage,
                normalize_single=True,
            )
            sum_fields.append(reduced_sum)
            area_fields.append(reduced_area)

        if not sum_fields:
            return ctx.fragment([])

        output_buffer = BufferSpec(DType.F64, FixedShape(out_shape), InitPolicy.ZERO)
        total_sum = reductions.reduce_pairwise(
            sum_fields,
            step=ds.step,
            target_level=ds.level,
            kernel="uniform_slice_add",
            output_buffer=output_buffer,
            stage_name="uniform_slice_add",
            template_name="uniform_slice_add_sum_{round}_{index}",
            temporary_name=f"{self.out_name}_sum_add_{{round}}_{{index}}",
        )
        total_area = reductions.reduce_pairwise(
            area_fields,
            step=ds.step,
            target_level=ds.level,
            kernel="uniform_slice_add",
            output_buffer=output_buffer,
            stage_name="uniform_slice_add",
            template_name="uniform_slice_add_area_{round}_{index}",
            temporary_name=f"{self.out_name}_area_add_{{round}}_{{index}}",
        )

        out_field = ctx.output_field(self.out_name)
        finalize_deps = reductions.dependencies([total_sum.field, total_area.field])
        finalize = ctx.stage("uniform_slice_finalize", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="uniform_slice_finalize",
            kernel="uniform_slice_finalize",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_sum.field.field,
                    version=total_sum.field.version,
                    domain=ctx.domain(step=ds.step, level=total_sum.level),
                ),
                FieldRef(
                    total_area.field.field,
                    version=total_area.field.version,
                    domain=ctx.domain(step=ds.step, level=total_area.level),
                ),
            ],
            outputs=[_fixed_real_shape(out_field, out_shape, dtype)],
            deps=DependencyRule(),
            params=SliceFinalizeParams(
                pixel_area=abs((self.rect[2] - self.rect[0]) / nx)
                * abs((self.rect[3] - self.rect[1]) / ny)
            ),
            graph_reduce=GraphReduceSpec(fan_in=1, num_inputs=1),
        )
        reductions.add_stage(finalize, outputs=[out_field])
        return ctx.fragment(reductions.stages)

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
        reduce_fan_in: Optional[int] = None,
        amr_cell_average: bool = True,
    ) -> None:
        self.field = field
        self.axis = axis
        self.axis_bounds = axis_bounds
        self.rect = rect
        self.resolution = resolution
        self.out_name = out_name
        self.reduce_fan_in = reduce_fan_in
        self.amr_cell_average = amr_cell_average

    def _lower_amr_projection(self, ctx: LoweringContext):
        ds = ctx.dataset
        axis_idx = _axis_index(self.axis)
        nx, ny = self.resolution
        if nx <= 0 or ny <= 0:
            raise ValueError("resolution must be positive")

        levels = ctx.runmeta.steps[ds.step].levels
        axis_range_by_level = axis_ranges_by_level(
            levels, axis=axis_idx, bounds=self.axis_bounds
        )

        sum_fields: list[ReducedField] = []
        reductions = GraphReductionBuilder(ctx)
        out_shape = (ny, nx)

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            axis_range = axis_range_by_level[level_idx]
            blocks = list(
                intersecting_slab_blocks(
                    level_meta,
                    axis=axis_idx,
                    axis_range=axis_range,
                    rect=self.rect,
                )
            )
            if not blocks:
                continue

            covered_boxes = covered_slab_boxes(
                levels,
                level=level_idx,
                axis=axis_idx,
                axis_ranges=axis_range_by_level,
            )
            covered_payload = _covered_boxes_payload(covered_boxes)

            sum_field = ctx.temp_field(f"{self.out_name}_sum_l{level_idx}")

            stage = ctx.stage("uniform_projection")
            for block in blocks:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                stage.map_blocks(
                    name=f"uniform_projection_b{block}",
                    kernel="uniform_projection_accumulate",
                    domain=dom,
                    inputs=[FieldRef(self.field)],
                    outputs=[_fixed_f64_shape(sum_field, out_shape)],
                    deps=DependencyRule(),
                    params=UniformProjectionParams(
                        axis=axis_idx,
                        axis_bounds=(float(self.axis_bounds[0]), float(self.axis_bounds[1])),
                        rect=tuple(self.rect),
                        resolution=(nx, ny),
                        covered_boxes=covered_payload,
                    ),
                )
            reductions.add_stage(stage, outputs=[sum_field])
            sum_fields.append(
                reductions.reduce_blocks(
                    value=ReducedField(sum_field, level_idx),
                    input_blocks=blocks,
                    step=ds.step,
                    fan_in=resolve_reduce_fan_in(self.reduce_fan_in, len(blocks)),
                    kernel="uniform_slice_reduce",
                    output_buffer=BufferSpec(
                        DType.F64, FixedShape(out_shape), InitPolicy.ZERO
                    ),
                    stage_name="uniform_slice_reduce",
                    template_name="uniform_projection_sum_reduce_s{round}",
                    temporary_name=(
                        f"{self.out_name}_sum_reduce_{level_idx}_{{round}}"
                    ),
                    after=stage,
                    normalize_single=True,
                )
            )

        if not sum_fields:
            return ctx.fragment([])

        total_sum = reductions.reduce_pairwise(
            sum_fields,
            step=ds.step,
            target_level=ds.level,
            kernel="uniform_slice_add",
            output_buffer=BufferSpec(DType.F64, FixedShape(out_shape), InitPolicy.ZERO),
            stage_name="uniform_projection_add",
            template_name="uniform_projection_add_{round}_{index}",
            temporary_name=f"{self.out_name}_sum_add_{{round}}_{{index}}",
            order_by_home=True,
            preserve_location=True,
        )
        out_field = ctx.output_field(self.out_name)
        finalize_deps = reductions.dependencies([total_sum.field])
        finalize = ctx.stage("uniform_projection_output", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="uniform_projection_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_sum.field.field,
                    version=total_sum.field.version,
                    domain=ctx.domain(
                        step=ds.step,
                        level=total_sum.level,
                        blocks=[total_sum.block],
                    ),
                )
            ],
            outputs=[_fixed_f64_shape(out_field, out_shape)],
            deps=DependencyRule(),
            graph_reduce=GraphReduceSpec(fan_in=1, num_inputs=1),
        )
        reductions.add_stage(finalize, outputs=[out_field])
        return ctx.fragment(reductions.stages)

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

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        axis_idx = _axis_index(self.axis)
        nx, ny = self.resolution
        if nx <= 0 or ny <= 0:
            raise ValueError("resolution must be positive")

        levels = ctx.runmeta.steps[ds.step].levels
        axis_range_by_level = axis_ranges_by_level(
            levels, axis=axis_idx, bounds=self.axis_bounds
        )

        sum_fields: list[ReducedField] = []
        reductions = GraphReductionBuilder(ctx)
        out_shape = (ny, nx)
        previous_level_tail = None

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            axis_range = axis_range_by_level[level_idx]
            blocks = list(
                intersecting_slab_blocks(
                    level_meta,
                    axis=axis_idx,
                    axis_range=axis_range,
                    rect=self.rect,
                )
            )
            if not blocks:
                continue

            covered_boxes = covered_slab_boxes(
                levels,
                level=level_idx,
                axis=axis_idx,
                axis_ranges=axis_range_by_level,
            )
            covered_payload = _covered_boxes_payload(covered_boxes)

            sum_field = ctx.temp_field(f"{self.out_name}_sum_l{level_idx}")

            stage = ctx.stage(
                "particle_cic_projection",
                after=[previous_level_tail] if previous_level_tail is not None else None,
            )
            dom = ctx.domain(step=ds.step, level=level_idx, blocks=blocks)
            params = ParticleCicProjectionParams(
                particle_type=self.particle_type,
                level_index=int(level_idx),
                axis=axis_idx,
                axis_bounds=(float(self.axis_bounds[0]), float(self.axis_bounds[1])),
                rect=tuple(self.rect),
                resolution=(nx, ny),
                mass_max=float(self.mass_max) if self.mass_max is not None else math.nan,
                covered_boxes=covered_payload,
            )
            stage.map_blocks(
                name=f"particle_cic_projection_l{level_idx}",
                kernel="particle_cic_projection_accumulate",
                domain=dom,
                inputs=[],
                outputs=[_fixed_f64_shape(sum_field, out_shape)],
                deps=DependencyRule(),
                params=params,
            )
            reductions.add_stage(stage, outputs=[sum_field])
            sum_fields.append(
                reductions.reduce_blocks(
                    value=ReducedField(sum_field, level_idx),
                    input_blocks=blocks,
                    step=ds.step,
                    fan_in=resolve_reduce_fan_in(self.reduce_fan_in, len(blocks)),
                    kernel="uniform_slice_reduce",
                    output_buffer=BufferSpec(
                        DType.F64, FixedShape(out_shape), InitPolicy.ZERO
                    ),
                    stage_name="particle_cic_projection_reduce",
                    template_name="particle_cic_projection_sum_reduce_s{round}",
                    temporary_name=(
                        f"{self.out_name}_sum_reduce_{level_idx}_{{round}}"
                    ),
                    after=stage,
                    normalize_single=True,
                )
            )
            previous_level_tail = reductions.stages[-1]

        if not sum_fields:
            return ctx.fragment([])

        total_sum = reductions.reduce_pairwise(
            sum_fields,
            step=ds.step,
            target_level=ds.level,
            kernel="uniform_slice_add",
            output_buffer=BufferSpec(DType.F64, FixedShape(out_shape), InitPolicy.ZERO),
            stage_name="particle_cic_projection_add",
            template_name="particle_cic_projection_add_{round}_{index}",
            temporary_name=f"{self.out_name}_sum_add_{{round}}_{{index}}",
            order_by_home=True,
            preserve_location=True,
        )
        out_field = ctx.output_field(self.out_name)
        finalize_deps = reductions.dependencies([total_sum.field])
        finalize = ctx.stage("particle_cic_projection_output", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="particle_cic_projection_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_sum.field.field,
                    version=total_sum.field.version,
                    domain=ctx.domain(
                        step=ds.step,
                        level=total_sum.level,
                        blocks=[total_sum.block],
                    ),
                )
            ],
            outputs=[_fixed_f64_shape(out_field, out_shape)],
            deps=DependencyRule(),
            graph_reduce=GraphReduceSpec(fan_in=1, num_inputs=1),
        )
        reductions.add_stage(finalize, outputs=[out_field])
        return ctx.fragment(reductions.stages)


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

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        axis_idx = _axis_index(self.axis)

        levels = ctx.runmeta.steps[ds.step].levels
        axis_range_by_level = axis_ranges_by_level(
            levels, axis=axis_idx, bounds=self.axis_bounds
        )

        stages: list = []
        grid_field = ctx.temp_field(self.out_name)

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            axis_range = axis_range_by_level[level_idx]
            blocks = list(
                intersecting_slab_blocks(
                    level_meta,
                    axis=axis_idx,
                    axis_range=axis_range,
                    rect=self.rect,
                )
            )
            if not blocks:
                continue

            covered_boxes = covered_slab_boxes(
                levels,
                level=level_idx,
                axis=axis_idx,
                axis_ranges=axis_range_by_level,
            )
            covered_payload = _covered_boxes_payload(covered_boxes)

            stage = ctx.stage("particle_cic_grid")
            for block in blocks:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                params = ParticleCicGridParams(
                    particle_type=self.particle_type,
                    level_index=int(level_idx),
                    axis=axis_idx,
                    axis_bounds=(float(self.axis_bounds[0]), float(self.axis_bounds[1])),
                    mass_max=float(self.mass_max) if self.mass_max is not None else math.nan,
                    covered_boxes=covered_payload,
                )
                stage.map_blocks(
                    name=f"particle_cic_grid_b{block}",
                    kernel="particle_cic_grid_accumulate",
                    domain=dom,
                    inputs=[],
                    outputs=[_block_f64(grid_field)],
                    deps=DependencyRule(),
                    params=params,
                )
            stages.append(stage)

        if not stages:
            return ctx.fragment([])

        return ctx.fragment(stages)


class FluxSurfaceIntegral:
    def __init__(
        self,
        *,
        density: int,
        momentum: tuple[int, int, int],
        energy: int,
        passive_scalar: int,
        magnetic_field: tuple[int, int, int],
        radius: float | Sequence[float],
        temperature: int | None = None,
        temperature_bins: Sequence[float] | None = None,
        out_name: str = "flux_surface_integral",
        gamma: float = 5.0 / 3.0,
        reduce_fan_in: Optional[int] = None,
    ) -> None:
        self.density = int(density)
        self.momentum = tuple(int(v) for v in momentum)
        self.energy = int(energy)
        self.passive_scalar = int(passive_scalar)
        self.magnetic_field = tuple(int(v) for v in magnetic_field)
        self.radii = self._coerce_radii(radius)
        self.temperature = None if temperature is None else int(temperature)
        self.temperature_bins = self._coerce_temperature_bins(temperature_bins)
        self.out_name = out_name
        self.gamma = float(gamma)
        self.reduce_fan_in = reduce_fan_in

    def _coerce_radii(self, radius: float | Sequence[float]) -> tuple[float, ...]:
        if isinstance(radius, (str, bytes)):
            raise TypeError("radius must be a number or a sequence of numbers")
        try:
            values = tuple(float(v) for v in radius)  # type: ignore[arg-type]
        except TypeError:
            values = (float(radius),)
        return values

    def _coerce_temperature_bins(
        self, temperature_bins: Sequence[float] | None
    ) -> tuple[float, ...] | None:
        if temperature_bins is None:
            return None
        if isinstance(temperature_bins, (str, bytes)):
            raise TypeError("temperature_bins must be a sequence of numbers")
        values = tuple(float(v) for v in temperature_bins)
        return values

    def _block_radius_bounds2(self, level_meta, block) -> tuple[float, float]:
        geom = level_meta.geom
        lo2 = 0.0
        hi2 = 0.0
        for axis in range(3):
            x0, x1 = _bounds_1d(
                block.lo[axis],
                block.hi[axis],
                geom.x0[axis],
                geom.dx[axis],
                geom.index_origin[axis],
            )
            if x1 < 0.0:
                lo2 += x1 * x1
            elif x0 > 0.0:
                lo2 += x0 * x0
            hi2 += max(abs(x0), abs(x1)) ** 2
        return lo2, hi2

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        if not self.radii:
            raise ValueError("radius must contain at least one value")
        if any(not math.isfinite(radius) or radius <= 0.0 for radius in self.radii):
            raise ValueError("radius values must be finite and positive")
        if self.temperature_bins is not None:
            if self.temperature is None:
                raise ValueError("temperature must be provided when temperature_bins are set")
            if len(self.temperature_bins) < 2:
                raise ValueError("temperature_bins must contain at least two edges")
            if any(not math.isfinite(edge) for edge in self.temperature_bins):
                raise ValueError("temperature_bins values must be finite")
            if any(
                right <= left
                for left, right in zip(self.temperature_bins, self.temperature_bins[1:])
            ):
                raise ValueError("temperature_bins values must be strictly increasing")
        if not math.isfinite(self.gamma) or self.gamma <= 1.0:
            raise ValueError("gamma must be finite and greater than 1")
        if len(self.momentum) != 3:
            raise ValueError("momentum must contain three fields")
        if len(self.magnetic_field) != 3:
            raise ValueError("magnetic_field must contain three cell-centered fields")

        input_fields = [
            self.density,
            *self.momentum,
            self.energy,
            self.passive_scalar,
            *self.magnetic_field,
        ]
        if self.temperature is not None:
            input_fields.append(self.temperature)
        levels = ctx.runmeta.steps[ds.step].levels
        num_temperature_bins = (
            len(self.temperature_bins) - 1 if self.temperature_bins is not None else 1
        )
        out_shape_parts: list[int] = []
        if len(self.radii) > 1:
            out_shape_parts.append(len(self.radii))
        out_shape_parts.append(2)
        if num_temperature_bins > 1:
            out_shape_parts.append(num_temperature_bins)
        out_shape_parts.append(4)
        out_shape = tuple(out_shape_parts)
        radii2 = [radius * radius for radius in self.radii]
        radius_intersects = [False] * len(self.radii)

        flux_fields: list[ReducedField] = []
        accumulate_stage = ctx.stage("flux_surface_integral")
        reductions = GraphReductionBuilder(ctx)
        reductions.add_stage(accumulate_stage)

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            block_radius_indices: list[tuple[int, list[int]]] = []
            for block_idx, block in enumerate(level_meta.boxes):
                lo2, hi2 = self._block_radius_bounds2(level_meta, block)
                active_radius_indices: list[int] = []
                for radius_idx, radius2 in enumerate(radii2):
                    if lo2 <= radius2 <= hi2:
                        radius_intersects[radius_idx] = True
                        active_radius_indices.append(radius_idx)
                if active_radius_indices:
                    block_radius_indices.append((block_idx, active_radius_indices))
            if not block_radius_indices:
                continue
            blocks = [block_idx for block_idx, _ in block_radius_indices]

            covered_boxes = covered_volume_boxes(levels, level=level_idx)
            covered_payload = _covered_boxes_payload(covered_boxes)
            flux_field = ctx.temp_field(f"{self.out_name}_sum_l{level_idx}")

            for block, active_radius_indices in block_radius_indices:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                active_radii = [self.radii[radius_idx] for radius_idx in active_radius_indices]
                accumulate_stage.map_blocks(
                    name=f"flux_surface_integral_b{block}",
                    kernel="flux_surface_integral_accumulate",
                    domain=dom,
                    inputs=[FieldRef(fid) for fid in input_fields],
                    outputs=[_fixed_f64_shape(flux_field, out_shape)],
                    deps=DependencyRule(),
                    params=FluxSurfaceParams(
                        radii=tuple(active_radii),
                        radius_indices=tuple(active_radius_indices),
                        num_radii=len(self.radii),
                        temperature_bins=tuple(self.temperature_bins or ()),
                        gamma=self.gamma,
                        covered_boxes=covered_payload,
                    ),
                )
            reductions.add_stage(accumulate_stage, outputs=[flux_field])
            flux_fields.append(
                reductions.reduce_blocks(
                    value=ReducedField(flux_field, level_idx),
                    input_blocks=blocks,
                    step=ds.step,
                    fan_in=resolve_reduce_fan_in(self.reduce_fan_in, len(blocks)),
                    kernel="uniform_slice_reduce",
                    output_buffer=BufferSpec(
                        DType.F64, FixedShape(out_shape), InitPolicy.ZERO
                    ),
                    stage_name="flux_surface_integral_reduce",
                    template_name="flux_surface_integral_reduce_s{round}",
                    singleton_template_name="flux_surface_integral_reduce_single",
                    temporary_name=(
                        f"{self.out_name}_sum_reduce_{level_idx}_{{round}}"
                    ),
                    after=accumulate_stage,
                    normalize_single=True,
                )
            )

        if not flux_fields:
            if len(self.radii) > 1:
                raise ValueError(
                    f"radius values do not intersect any mesh block: {list(self.radii)}"
                )
            raise ValueError("radius does not intersect any mesh block")
        if not all(radius_intersects):
            missing = [
                radius
                for radius, intersects in zip(self.radii, radius_intersects)
                if not intersects
            ]
            raise ValueError(f"radius values do not intersect any mesh block: {missing}")

        total_flux = reductions.reduce_pairwise(
            flux_fields,
            step=ds.step,
            target_level=ds.level,
            kernel="uniform_slice_add",
            output_buffer=BufferSpec(DType.F64, FixedShape(out_shape), InitPolicy.ZERO),
            stage_name="flux_surface_integral_add",
            template_name="flux_surface_integral_add_{round}_{index}",
            temporary_name=f"{self.out_name}_sum_add_{{round}}_{{index}}",
        )
        out_field = ctx.output_field(self.out_name)
        finalize_deps = reductions.dependencies([total_flux.field])
        finalize = ctx.stage("flux_surface_integral_output", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="flux_surface_integral_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_flux.field.field,
                    version=total_flux.field.version,
                    domain=ctx.domain(step=ds.step, level=total_flux.level),
                )
            ],
            outputs=[_fixed_f64_shape(out_field, out_shape)],
            deps=DependencyRule(),
            graph_reduce=GraphReduceSpec(fan_in=1, num_inputs=1),
        )
        reductions.add_stage(finalize, outputs=[out_field])
        return ctx.fragment(reductions.stages)


class ToomreQProfile:
    """Accumulate cylindrical annular moments for gas Toomre-Q profiles."""

    NUM_COMPONENTS = 7

    def __init__(
        self,
        *,
        density: int,
        momentum: tuple[int, int],
        internal_energy: int,
        magnetic_field: tuple[int, int, int],
        potential: int,
        radial_edges: Sequence[float],
        z_bounds: tuple[float, float],
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        out_name: str = "toomre_q_profile",
        gamma: float = 5.0 / 3.0,
        reduce_fan_in: Optional[int] = None,
    ) -> None:
        self.density = int(density)
        self.momentum = tuple(int(v) for v in momentum)
        self.internal_energy = int(internal_energy)
        self.magnetic_field = tuple(int(v) for v in magnetic_field)
        self.potential = int(potential)
        self.radial_edges = tuple(float(v) for v in radial_edges)
        self.z_bounds = tuple(float(v) for v in z_bounds)
        self.center = tuple(float(v) for v in center)
        self.out_name = str(out_name)
        self.gamma = float(gamma)
        self.reduce_fan_in = reduce_fan_in

    def _reduce_fan_in(self, num_inputs: int) -> int:
        return resolve_reduce_fan_in(self.reduce_fan_in, num_inputs)

    def _ref_ratio_between(self, levels, coarse: int, fine: int) -> int:
        ratio = 1
        for level in range(coarse, fine):
            ratio *= int(levels[level].geom.ref_ratio)
        return ratio

    def _covered_boxes_for_level(
        self,
        ctx: LoweringContext,
        *,
        level: int,
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        levels = ctx.runmeta.steps[ctx.dataset.step].levels
        return covered_volume_boxes(levels, level=level)

    def _block_intersects_profile(self, level_meta, block) -> bool:
        geom = level_meta.geom
        z0, z1 = _bounds_1d(
            block.lo[2],
            block.hi[2],
            geom.x0[2],
            geom.dx[2],
            geom.index_origin[2],
        )
        if z1 <= self.z_bounds[0] or z0 >= self.z_bounds[1]:
            return False

        lo2 = 0.0
        hi2 = 0.0
        for axis in (0, 1):
            x0, x1 = _bounds_1d(
                block.lo[axis],
                block.hi[axis],
                geom.x0[axis],
                geom.dx[axis],
                geom.index_origin[axis],
            )
            x0 -= self.center[axis]
            x1 -= self.center[axis]
            if x1 < 0.0:
                lo2 += x1 * x1
            elif x0 > 0.0:
                lo2 += x0 * x0
            hi2 += max(abs(x0), abs(x1)) ** 2
        rmin2 = self.radial_edges[0] ** 2
        rmax2 = self.radial_edges[-1] ** 2
        return lo2 < rmax2 and hi2 >= rmin2

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        if len(self.momentum) != 2:
            raise ValueError("momentum must contain x and y fields")
        if len(self.magnetic_field) != 3:
            raise ValueError("magnetic_field must contain three cell-centered fields")
        if len(self.radial_edges) < 2:
            raise ValueError("radial_edges must contain at least two values")
        if (
            any(not math.isfinite(value) for value in self.radial_edges)
            or self.radial_edges[0] < 0.0
            or any(
                right <= left
                for left, right in zip(self.radial_edges, self.radial_edges[1:])
            )
        ):
            raise ValueError(
                "radial_edges must be finite, non-negative, and strictly increasing"
            )
        bins = len(self.radial_edges) - 1
        if len(self.z_bounds) != 2:
            raise ValueError("z_bounds must contain two values")
        if (
            not math.isfinite(self.z_bounds[0])
            or not math.isfinite(self.z_bounds[1])
            or self.z_bounds[1] <= self.z_bounds[0]
        ):
            raise ValueError("z_bounds must be finite and increasing")
        if len(self.center) != 3 or any(not math.isfinite(v) for v in self.center):
            raise ValueError("center must contain three finite coordinates")
        if not math.isfinite(self.gamma) or self.gamma <= 1.0:
            raise ValueError("gamma must be finite and greater than 1")

        levels = ctx.runmeta.steps[ds.step].levels
        active_by_level: dict[int, list[int]] = {}
        for level_idx, level_meta in enumerate(levels):
            active = [
                block_idx
                for block_idx, block in enumerate(level_meta.boxes)
                if self._block_intersects_profile(level_meta, block)
            ]
            if active:
                active_by_level[level_idx] = active
        if not active_by_level:
            raise ValueError("radial_edges and z_bounds do not intersect any mesh block")

        output_buffer = BufferSpec(
            DType.F64,
            FixedShape((bins, self.NUM_COMPONENTS)),
            InitPolicy.ZERO,
        )
        neighbor_field = ctx.temp_field(f"{self.out_name}_potential_neighbors")
        gradient_field = ctx.temp_field(f"{self.out_name}_potential_gradient")
        fetch_stage = ctx.stage("toomre_q_potential_fetch")
        gradient_stage = ctx.stage("toomre_q_potential_gradient", after=[fetch_stage])
        accumulate_stage = ctx.stage("toomre_q_profile", after=[gradient_stage])
        reductions = GraphReductionBuilder(ctx)
        reductions.add_stage(fetch_stage)
        reductions.add_stage(gradient_stage)
        reductions.add_stage(accumulate_stage)
        profile_fields: list[ReducedField] = []

        input_fields = [
            self.density,
            *self.momentum,
            self.internal_energy,
            *self.magnetic_field,
        ]

        for level_idx in range(len(levels) - 1, -1, -1):
            blocks = active_by_level.get(level_idx, [])
            if not blocks:
                continue
            level_meta = levels[level_idx]
            covered_boxes = self._covered_boxes_for_level(ctx, level=level_idx)
            covered_payload = _covered_boxes_payload(covered_boxes)
            profile_field = ctx.temp_field(f"{self.out_name}_sum_l{level_idx}")

            for block in blocks:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                fetch_stage.map_blocks(
                    name=f"toomre_q_potential_fetch_l{level_idx}_b{block}",
                    kernel="amr_subbox_fetch_pack",
                    domain=dom,
                    inputs=[],
                    outputs=[_amr_patch_payload(neighbor_field)],
                    deps=DependencyRule(),
                    params=AmrSubboxPackParams(
                        input_field=self.potential,
                        input_version=0,
                        input_step=ds.step,
                        input_level=level_idx,
                        halo_cells=1,
                    ),
                )
                gradient_stage.map_blocks(
                    name=f"toomre_q_potential_gradient_l{level_idx}_b{block}",
                    kernel="gradU_stencil",
                    domain=dom,
                    inputs=[FieldRef(self.potential), neighbor_field],
                    outputs=[_block_f64(gradient_field, 3)],
                    deps=DependencyRule(),
                    params=GradStencilParams(
                        input_field=self.potential,
                        input_version=0,
                        input_step=ds.step,
                        input_level=level_idx,
                        stencil_radius=1,
                    ),
                )
                accumulate_stage.map_blocks(
                    name=f"toomre_q_profile_l{level_idx}_b{block}",
                    kernel="toomre_profile_accumulate",
                    domain=dom,
                    inputs=[FieldRef(fid) for fid in input_fields] + [gradient_field],
                    outputs=[OutputRef(profile_field, output_buffer)],
                    deps=DependencyRule(),
                    params=ToomreProfileParams(
                        radial_edges=self.radial_edges,
                        z_bounds=(float(self.z_bounds[0]), float(self.z_bounds[1])),
                        center=tuple(float(v) for v in self.center),
                        covered_boxes=covered_payload,
                    ),
                )

            reductions.add_stage(accumulate_stage, outputs=[profile_field])
            profile_fields.append(
                reductions.reduce_blocks(
                    value=ReducedField(profile_field, level_idx),
                    input_blocks=blocks,
                    step=ds.step,
                    fan_in=self._reduce_fan_in(len(blocks)),
                    kernel="uniform_slice_reduce",
                    output_buffer=output_buffer,
                    stage_name="toomre_q_profile_reduce",
                    template_name="toomre_q_profile_reduce_s{round}",
                    singleton_template_name="toomre_q_profile_reduce_single",
                    temporary_name=f"{self.out_name}_sum_reduce_{level_idx}_{{round}}",
                    after=accumulate_stage,
                    normalize_single=True,
                )
            )

        total_profile = reductions.reduce_pairwise(
            profile_fields,
            step=ds.step,
            target_level=ds.level,
            kernel="uniform_slice_add",
            output_buffer=output_buffer,
            stage_name="toomre_q_profile_add",
            template_name="toomre_q_profile_add_{round}_{index}",
            temporary_name=f"{self.out_name}_sum_add_{{round}}_{{index}}",
        )
        out_field = ctx.output_field(self.out_name)
        finalize = ctx.stage("toomre_q_profile_output", plane="graph", after=reductions.dependencies([total_profile.field]))
        finalize.map_blocks(
            name="toomre_q_profile_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_profile.field.field,
                    version=total_profile.field.version,
                    domain=ctx.domain(step=ds.step, level=total_profile.level, blocks=[0]),
                )
            ],
            outputs=[OutputRef(out_field, output_buffer)],
            deps=DependencyRule(),
            graph_reduce=GraphReduceSpec(fan_in=1, num_inputs=1),
        )
        reductions.add_stage(finalize, outputs=[out_field])
        return ctx.fragment(reductions.stages)


class CylindricalFluxSurfaceIntegral(FluxSurfaceIntegral):
    def __init__(
        self,
        *,
        density: int,
        momentum: tuple[int, int, int],
        energy: int,
        passive_scalar: int,
        magnetic_field: tuple[int, int, int],
        radius: float,
        height: float | Sequence[float],
        temperature: int | None = None,
        temperature_bins: Sequence[float] | None = None,
        out_name: str = "cylindrical_flux_surface_integral",
        gamma: float = 5.0 / 3.0,
        reduce_fan_in: Optional[int] = None,
    ) -> None:
        super().__init__(
            density=density,
            momentum=momentum,
            energy=energy,
            passive_scalar=passive_scalar,
            magnetic_field=magnetic_field,
            radius=radius,
            temperature=temperature,
            temperature_bins=temperature_bins,
            out_name=out_name,
            gamma=gamma,
            reduce_fan_in=reduce_fan_in,
        )
        self.radius = self.radii[0]
        if len(self.radii) != 1:
            raise ValueError("radius must be a single fixed value")
        self.heights = self._coerce_heights(height)

    def _coerce_heights(self, height: float | Sequence[float]) -> tuple[float, ...]:
        if isinstance(height, (str, bytes)):
            raise TypeError("height must be a number or a sequence of numbers")
        try:
            values = tuple(float(v) for v in height)  # type: ignore[arg-type]
        except TypeError:
            values = (float(height),)
        return values

    def _block_cylinder_bounds(self, level_meta, block) -> tuple[float, float, float, float]:
        geom = level_meta.geom
        x0, x1 = _bounds_1d(
            block.lo[0],
            block.hi[0],
            geom.x0[0],
            geom.dx[0],
            geom.index_origin[0],
        )
        y0, y1 = _bounds_1d(
            block.lo[1],
            block.hi[1],
            geom.x0[1],
            geom.dx[1],
            geom.index_origin[1],
        )
        z0, z1 = _bounds_1d(
            block.lo[2],
            block.hi[2],
            geom.x0[2],
            geom.dx[2],
            geom.index_origin[2],
        )
        lo2 = 0.0
        if x1 < 0.0:
            lo2 += x1 * x1
        elif x0 > 0.0:
            lo2 += x0 * x0
        if y1 < 0.0:
            lo2 += y1 * y1
        elif y0 > 0.0:
            lo2 += y0 * y0
        hi2 = max(abs(x0), abs(x1)) ** 2 + max(abs(y0), abs(y1)) ** 2
        z_abs_min = 0.0 if z0 <= 0.0 <= z1 else min(abs(z0), abs(z1))
        z_abs_max = max(abs(z0), abs(z1))
        return lo2, hi2, z_abs_min, z_abs_max

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        if not self.heights:
            raise ValueError("height must contain at least one value")
        if any(not math.isfinite(height) or height <= 0.0 for height in self.heights):
            raise ValueError("height values must be finite and positive")
        if any(not math.isfinite(radius) or radius <= 0.0 for radius in self.radii):
            raise ValueError("radius values must be finite and positive")
        if self.temperature_bins is not None:
            if self.temperature is None:
                raise ValueError("temperature must be provided when temperature_bins are set")
            if len(self.temperature_bins) < 2:
                raise ValueError("temperature_bins must contain at least two edges")
            if any(not math.isfinite(edge) for edge in self.temperature_bins):
                raise ValueError("temperature_bins values must be finite")
            if any(
                right <= left
                for left, right in zip(self.temperature_bins, self.temperature_bins[1:])
            ):
                raise ValueError("temperature_bins values must be strictly increasing")
        if not math.isfinite(self.gamma) or self.gamma <= 1.0:
            raise ValueError("gamma must be finite and greater than 1")
        if len(self.momentum) != 3:
            raise ValueError("momentum must contain three fields")
        if len(self.magnetic_field) != 3:
            raise ValueError("magnetic_field must contain three cell-centered fields")

        input_fields = [
            self.density,
            *self.momentum,
            self.energy,
            self.passive_scalar,
            *self.magnetic_field,
        ]
        if self.temperature is not None:
            input_fields.append(self.temperature)
        levels = ctx.runmeta.steps[ds.step].levels
        num_temperature_bins = (
            len(self.temperature_bins) - 1 if self.temperature_bins is not None else 1
        )
        num_geometric_sections = 2
        out_shape_parts: list[int] = []
        if len(self.heights) > 1:
            out_shape_parts.append(len(self.heights))
        out_shape_parts.append(2)
        if num_temperature_bins > 1:
            out_shape_parts.append(num_temperature_bins)
        if len(out_shape_parts) < 3:
            out_shape_parts.extend((num_geometric_sections, 4))
        else:
            out_shape_parts.append(num_geometric_sections * 4)
        out_shape = tuple(out_shape_parts)
        radius2 = self.radius * self.radius
        height_intersects = [False] * len(self.heights)

        flux_fields: list[ReducedField] = []
        accumulate_stage = ctx.stage("cylindrical_flux_surface_integral")
        reductions = GraphReductionBuilder(ctx)
        reductions.add_stage(accumulate_stage)

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            block_height_indices: list[tuple[int, list[int]]] = []
            for block_idx, block in enumerate(level_meta.boxes):
                lo2, hi2, z_abs_min, z_abs_max = self._block_cylinder_bounds(level_meta, block)
                if not (lo2 <= radius2 <= hi2):
                    continue
                active_height_indices: list[int] = []
                for height_idx, height in enumerate(self.heights):
                    if z_abs_min <= height and z_abs_max >= 0.0:
                        height_intersects[height_idx] = True
                        active_height_indices.append(height_idx)
                if active_height_indices:
                    block_height_indices.append((block_idx, active_height_indices))
            if not block_height_indices:
                continue
            blocks = [block_idx for block_idx, _ in block_height_indices]

            covered_boxes = covered_volume_boxes(levels, level=level_idx)
            covered_payload = _covered_boxes_payload(covered_boxes)
            flux_field = ctx.temp_field(f"{self.out_name}_sum_l{level_idx}")

            for block, active_height_indices in block_height_indices:
                dom = ctx.domain(step=ds.step, level=level_idx, blocks=[block])
                active_heights = [self.heights[idx] for idx in active_height_indices]
                accumulate_stage.map_blocks(
                    name=f"cylindrical_flux_surface_integral_b{block}",
                    kernel="cylindrical_flux_surface_integral_accumulate",
                    domain=dom,
                    inputs=[FieldRef(fid) for fid in input_fields],
                    outputs=[_fixed_f64_shape(flux_field, out_shape)],
                    deps=DependencyRule(),
                    params=CylindricalFluxParams(
                        radius=self.radius,
                        heights=tuple(active_heights),
                        height_indices=tuple(active_height_indices),
                        num_heights=len(self.heights),
                        temperature_bins=tuple(self.temperature_bins or ()),
                        gamma=self.gamma,
                        covered_boxes=covered_payload,
                    ),
                )
            reductions.add_stage(accumulate_stage, outputs=[flux_field])
            flux_fields.append(
                reductions.reduce_blocks(
                    value=ReducedField(flux_field, level_idx),
                    input_blocks=blocks,
                    step=ds.step,
                    fan_in=resolve_reduce_fan_in(self.reduce_fan_in, len(blocks)),
                    kernel="uniform_slice_reduce",
                    output_buffer=BufferSpec(
                        DType.F64, FixedShape(out_shape), InitPolicy.ZERO
                    ),
                    stage_name="cylindrical_flux_surface_integral_reduce",
                    template_name="cylindrical_flux_surface_integral_reduce_s{round}",
                    singleton_template_name=(
                        "cylindrical_flux_surface_integral_reduce_single"
                    ),
                    temporary_name=(
                        f"{self.out_name}_sum_reduce_{level_idx}_{{round}}"
                    ),
                    after=accumulate_stage,
                    normalize_single=True,
                )
            )

        if not flux_fields:
            if len(self.heights) > 1:
                raise ValueError(
                    f"height values do not intersect any mesh block: {list(self.heights)}"
                )
            raise ValueError("cylindrical surface does not intersect any mesh block")
        if not all(height_intersects):
            missing = [
                height
                for height, intersects in zip(self.heights, height_intersects)
                if not intersects
            ]
            raise ValueError(f"height values do not intersect any mesh block: {missing}")

        total_flux = reductions.reduce_pairwise(
            flux_fields,
            step=ds.step,
            target_level=ds.level,
            kernel="uniform_slice_add",
            output_buffer=BufferSpec(DType.F64, FixedShape(out_shape), InitPolicy.ZERO),
            stage_name="cylindrical_flux_surface_integral_add",
            template_name=(
                "cylindrical_flux_surface_integral_add_{round}_{index}"
            ),
            temporary_name=f"{self.out_name}_sum_add_{{round}}_{{index}}",
        )
        out_field = ctx.output_field(self.out_name)
        finalize_deps = reductions.dependencies([total_flux.field])
        finalize = ctx.stage(
            "cylindrical_flux_surface_integral_output",
            plane="graph",
            after=finalize_deps,
        )
        finalize.map_blocks(
            name="cylindrical_flux_surface_integral_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_flux.field.field,
                    version=total_flux.field.version,
                    domain=ctx.domain(step=ds.step, level=total_flux.level),
                )
            ],
            outputs=[_fixed_f64_shape(out_field, out_shape)],
            deps=DependencyRule(),
            graph_reduce=GraphReduceSpec(fan_in=1, num_inputs=1),
        )
        reductions.add_stage(finalize, outputs=[out_field])
        return ctx.fragment(reductions.stages)


class Histogram1D:
    def __init__(
        self,
        field: int,
        *,
        hist_range: tuple[float, float],
        bins: int,
        out_name: str = "histogram1d",
        weight_field: int | None = None,
        reduce_fan_in: Optional[int] = None,
    ) -> None:
        self.field = field
        self.hist_range = hist_range
        self.bins = bins
        self.out_name = out_name
        self.weight_field = weight_field
        self.reduce_fan_in = reduce_fan_in

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        if self.bins <= 0:
            raise ValueError("bins must be positive")
        lo, hi = self.hist_range
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError("hist_range must be finite and increasing")

        levels = ctx.runmeta.steps[ds.step].levels
        out_shape = (self.bins,)

        hist_fields: list[ReducedField] = []
        reductions = GraphReductionBuilder(ctx)

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            blocks = list(range(len(level_meta.boxes)))
            if not blocks:
                continue

            covered_boxes = covered_volume_boxes(levels, level=level_idx)
            covered_payload = _covered_boxes_payload(covered_boxes)
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
                    outputs=[_fixed_f64_shape(hist_field, out_shape)],
                    deps=DependencyRule(),
                    params=Histogram1DParams(
                        range=(float(lo), float(hi)),
                        bins=int(self.bins),
                        covered_boxes=covered_payload,
                    ),
                )
            reductions.add_stage(stage, outputs=[hist_field])
            hist_fields.append(
                reductions.reduce_blocks(
                    value=ReducedField(hist_field, level_idx),
                    input_blocks=blocks,
                    step=ds.step,
                    fan_in=resolve_reduce_fan_in(self.reduce_fan_in, len(blocks)),
                    kernel="uniform_slice_reduce",
                    output_buffer=BufferSpec(
                        DType.F64, FixedShape(out_shape), InitPolicy.ZERO
                    ),
                    stage_name="histogram1d_reduce",
                    template_name="histogram1d_reduce_s{round}",
                    temporary_name=(
                        f"{self.out_name}_sum_reduce_{level_idx}_{{round}}"
                    ),
                    after=stage,
                )
            )

        if not hist_fields:
            return ctx.fragment([])

        total_hist = reductions.reduce_pairwise(
            hist_fields,
            step=ds.step,
            target_level=ds.level,
            kernel="uniform_slice_add",
            output_buffer=BufferSpec(DType.F64, FixedShape(out_shape), InitPolicy.ZERO),
            stage_name="histogram1d_add",
            template_name="histogram1d_add_{round}_{index}",
            temporary_name=f"{self.out_name}_sum_add_{{round}}_{{index}}",
        )
        out_field = ctx.output_field(self.out_name)
        finalize_deps = reductions.dependencies([total_hist.field])
        finalize = ctx.stage("histogram1d_output", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="histogram1d_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_hist.field.field,
                    version=total_hist.field.version,
                    domain=ctx.domain(step=ds.step, level=total_hist.level),
                )
            ],
            outputs=[_fixed_f64_shape(out_field, out_shape)],
            deps=DependencyRule(),
            graph_reduce=GraphReduceSpec(fan_in=1, num_inputs=1),
        )
        reductions.add_stage(finalize, outputs=[out_field])
        return ctx.fragment(reductions.stages)


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
        self.reduce_fan_in = reduce_fan_in

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

        levels = ctx.runmeta.steps[ds.step].levels
        hist_fields: list[ReducedField] = []
        reductions = GraphReductionBuilder(ctx)

        for level_idx in range(len(levels) - 1, -1, -1):
            level_meta = levels[level_idx]
            blocks = list(range(len(level_meta.boxes)))
            if not blocks:
                continue

            covered_boxes = covered_volume_boxes(levels, level=level_idx)
            covered_payload = _covered_boxes_payload(covered_boxes)
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
                    outputs=[_fixed_f64_shape(hist_field, (nx, ny))],
                    deps=DependencyRule(),
                    params=Histogram2DParams(
                        x_range=(float(x0), float(x1)),
                        y_range=(float(y0), float(y1)),
                        bins=(int(nx), int(ny)),
                        weight_mode=self.weight_mode,
                        covered_boxes=covered_payload,
                    ),
                )
            reductions.add_stage(stage, outputs=[hist_field])
            hist_fields.append(
                reductions.reduce_blocks(
                    value=ReducedField(hist_field, level_idx),
                    input_blocks=blocks,
                    step=ds.step,
                    fan_in=resolve_reduce_fan_in(self.reduce_fan_in, len(blocks)),
                    kernel="uniform_slice_reduce",
                    output_buffer=BufferSpec(
                        DType.F64, FixedShape((nx, ny)), InitPolicy.ZERO
                    ),
                    stage_name="histogram2d_reduce",
                    template_name="histogram2d_reduce_s{round}",
                    temporary_name=(
                        f"{self.out_name}_sum_reduce_{level_idx}_{{round}}"
                    ),
                    after=stage,
                )
            )

        if not hist_fields:
            return ctx.fragment([])

        total_hist = reductions.reduce_pairwise(
            hist_fields,
            step=ds.step,
            target_level=ds.level,
            kernel="uniform_slice_add",
            output_buffer=BufferSpec(
                DType.F64, FixedShape((nx, ny)), InitPolicy.ZERO
            ),
            stage_name="histogram2d_add",
            template_name="histogram2d_add_{round}_{index}",
            temporary_name=f"{self.out_name}_sum_add_{{round}}_{{index}}",
        )
        out_field = ctx.output_field(self.out_name)
        finalize_deps = reductions.dependencies([total_hist.field])
        finalize = ctx.stage("histogram2d_output", plane="graph", after=finalize_deps)
        finalize.map_blocks(
            name="histogram2d_output",
            kernel="uniform_slice_reduce",
            domain=ctx.domain(step=ds.step, level=ds.level),
            inputs=[
                FieldRef(
                    total_hist.field.field,
                    version=total_hist.field.version,
                    domain=ctx.domain(step=ds.step, level=total_hist.level),
                )
            ],
            outputs=[_fixed_f64_shape(out_field, (nx, ny))],
            deps=DependencyRule(),
            graph_reduce=GraphReduceSpec(fan_in=1, num_inputs=1),
        )
        reductions.add_stage(finalize, outputs=[out_field])
        return ctx.fragment(reductions.stages)


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
