from __future__ import annotations

import math
from typing import Any

import flatbuffers

from . import buffer as model_buffer
from . import kernel_params as model_params
from . import plan as model_plan
from .fb import (
    AmrSubboxPackParams,
    BufferSpec,
    CoveredBox,
    CoveredBoxList,
    CylindricalFluxParams,
    DependencyRule,
    Domain,
    DynamicUpperBound,
    FieldExprParams,
    FieldRef,
    FiniteOnlyParams,
    FluxSurfaceParams,
    GradStencilParams,
    GraphReduceSpec,
    Histogram1DParams,
    Histogram2DParams,
    KernelParams,
    OutputRef,
    ParticleCicGridParams,
    ParticleCicProjectionParams,
    ParticleFieldParams,
    ParticleHistogramParams,
    Plan,
    PlotfileLoadParams,
    ScalarParams,
    SliceFinalizeParams,
    Stage,
    TaskTemplate,
    TopKModesParams,
    UniformProjectionParams,
    UniformSliceCellParams,
    UniformSliceParams,
    ValuesParams,
)


_PLANE = {"chunk": 0, "graph": 1, "mixed": 2}
_DTYPE = {
    model_buffer.DType.OPAQUE: 0,
    model_buffer.DType.U8: 1,
    model_buffer.DType.I64: 2,
    model_buffer.DType.F32: 3,
    model_buffer.DType.F64: 4,
}
_INIT = {model_buffer.InitPolicy.UNINITIALIZED: 0, model_buffer.InitPolicy.ZERO: 1}
_DYNAMIC_KIND = {
    model_buffer.DynamicUpperBoundKind.LITERAL: 0,
    model_buffer.DynamicUpperBoundKind.LIKE_INPUT: 1,
    model_buffer.DynamicUpperBoundKind.BACKEND_CHUNK: 2,
    model_buffer.DynamicUpperBoundKind.AMR_SUBBOX_PACK: 3,
}


def _domain(value: model_plan.Domain) -> Domain.DomainT:
    blocks = None if value.blocks is None else [int(block) for block in value.blocks]
    return Domain.DomainT(
        step=value.step,
        level=value.level,
        blocks=blocks,
        allBlocks=value.blocks is None,
    )


def _field_ref(value: model_plan.FieldRef, fallback: model_plan.Domain) -> FieldRef.FieldRefT:
    return FieldRef.FieldRefT(
        field=value.field,
        version=value.version,
        domain=_domain(value.domain if value.domain is not None else fallback),
    )


def _buffer_spec(value: model_buffer.BufferSpec) -> BufferSpec.BufferSpecT:
    shape = value.shape
    fixed_extents: list[int] | None = None
    block_components = 1
    like_input_index = -1
    dynamic_bound = None
    if isinstance(shape, model_buffer.BlockShape):
        shape_kind = 0
        block_components = shape.components
    elif isinstance(shape, model_buffer.FixedShape):
        shape_kind = 1
        fixed_extents = list(shape.extents)
    elif isinstance(shape, model_buffer.LikeInputShape):
        shape_kind = 2
        like_input_index = shape.input_index
    elif isinstance(shape, model_buffer.DynamicShape):
        shape_kind = 3
        bound = shape.upper_bound
        dynamic_bound = DynamicUpperBound.DynamicUpperBoundT(
            kind=_DYNAMIC_KIND[bound.kind],
            value=bound.value if bound.value is not None else 0,
            inputIndex=bound.input_index if bound.input_index is not None else -1,
        )
    else:
        raise TypeError(f"unsupported buffer shape: {type(shape).__name__}")
    return BufferSpec.BufferSpecT(
        scalar=_DTYPE[value.dtype],
        shapeKind=shape_kind,
        fixedExtents=fixed_extents,
        blockComponents=block_components,
        likeInputIndex=like_input_index,
        dynamicUpperBound=dynamic_bound,
        init=_INIT[value.init],
    )


def _graph_reduce(value: model_plan.GraphReduceSpec | None) -> GraphReduceSpec.GraphReduceSpecT | None:
    if value is None:
        return None
    return GraphReduceSpec.GraphReduceSpecT(
        fanIn=value.fan_in,
        numInputs=value.num_inputs,
        inputBase=value.input_base,
        outputBase=value.output_base,
        inputBlocks=list(value.input_blocks),
        outputBlocks=list(value.output_blocks),
        groupOffsets=list(value.group_offsets),
    )


def _params(value: model_params.KernelParams) -> tuple[int, Any]:
    p = model_params
    if isinstance(value, p.NoKernelParams):
        return KernelParams.KernelParams.NONE, None
    if isinstance(value, p.AmrSubboxPackParams):
        return KernelParams.KernelParams.AmrSubboxPackParams, AmrSubboxPackParams.AmrSubboxPackParamsT(
            value.input_field, value.input_version, value.input_step, value.input_level, value.halo_cells
        )
    if isinstance(value, p.GradStencilParams):
        return KernelParams.KernelParams.GradStencilParams, GradStencilParams.GradStencilParamsT(
            value.input_field, value.input_version, value.input_step, value.input_level, value.stencil_radius
        )
    if isinstance(value, p.PlotfileLoadParams):
        return KernelParams.KernelParams.PlotfileLoadParams, PlotfileLoadParams.PlotfileLoadParamsT(
            value.plotfile, value.level, value.comp
        )
    if isinstance(value, p.FluxSurfaceParams):
        return KernelParams.KernelParams.FluxSurfaceParams, FluxSurfaceParams.FluxSurfaceParamsT(
            list(value.radii), list(value.radius_indices), list(value.temperature_bins), value.num_radii, value.gamma
        )
    if isinstance(value, p.CylindricalFluxParams):
        return KernelParams.KernelParams.CylindricalFluxParams, CylindricalFluxParams.CylindricalFluxParamsT(
            value.radius, list(value.heights), list(value.height_indices), list(value.temperature_bins), value.num_heights, value.gamma
        )
    if isinstance(value, p.UniformSliceCellParams):
        return KernelParams.KernelParams.UniformSliceCellParams, UniformSliceCellParams.UniformSliceCellParamsT(
            value.axis, value.coord,
            value.plane_index if value.plane_index is not None else 0, value.plane_index is not None,
            list(value.rect), list(value.resolution)
        )
    if isinstance(value, p.UniformProjectionParams):
        return KernelParams.KernelParams.UniformProjectionParams, UniformProjectionParams.UniformProjectionParamsT(
            value.axis, list(value.axis_bounds), list(value.rect), list(value.resolution)
        )
    if isinstance(value, p.FieldExprParams):
        return KernelParams.KernelParams.FieldExprParams, FieldExprParams.FieldExprParamsT(
            value.expression, list(value.variables)
        )
    if isinstance(value, p.UniformSliceParams):
        return KernelParams.KernelParams.UniformSliceParams, UniformSliceParams.UniformSliceParamsT(
            value.axis, value.coord, list(value.rect), list(value.resolution)
        )
    if isinstance(value, p.Histogram1DParams):
        return KernelParams.KernelParams.Histogram1DParams, Histogram1DParams.Histogram1DParamsT(
            list(value.range), value.bins
        )
    if isinstance(value, p.Histogram2DParams):
        return KernelParams.KernelParams.Histogram2DParams, Histogram2DParams.Histogram2DParamsT(
            list(value.x_range), list(value.y_range), list(value.bins), value.weight_mode
        )
    if isinstance(value, p.ParticleFieldParams):
        return KernelParams.KernelParams.ParticleFieldParams, ParticleFieldParams.ParticleFieldParamsT(
            value.particle_type, value.field_name
        )
    if isinstance(value, p.ParticleCicGridParams):
        return KernelParams.KernelParams.ParticleCicGridParams, ParticleCicGridParams.ParticleCicGridParamsT(
            value.particle_type, value.level_index, value.axis, list(value.axis_bounds),
            0.0 if math.isnan(value.mass_max) else value.mass_max, not math.isnan(value.mass_max)
        )
    if isinstance(value, p.ParticleCicProjectionParams):
        return KernelParams.KernelParams.ParticleCicProjectionParams, ParticleCicProjectionParams.ParticleCicProjectionParamsT(
            value.particle_type, value.level_index, value.axis, list(value.axis_bounds),
            list(value.rect), list(value.resolution),
            0.0 if math.isnan(value.mass_max) else value.mass_max, not math.isnan(value.mass_max)
        )
    if isinstance(value, p.ScalarParams):
        return KernelParams.KernelParams.ScalarParams, ScalarParams.ScalarParamsT(value.scalar)
    if isinstance(value, p.ValuesParams):
        return KernelParams.KernelParams.ValuesParams, ValuesParams.ValuesParamsT(list(value.values))
    if isinstance(value, p.FiniteOnlyParams):
        return KernelParams.KernelParams.FiniteOnlyParams, FiniteOnlyParams.FiniteOnlyParamsT(value.finite_only)
    if isinstance(value, p.ParticleHistogramParams):
        return KernelParams.KernelParams.ParticleHistogramParams, ParticleHistogramParams.ParticleHistogramParamsT(
            list(value.edges), value.density
        )
    if isinstance(value, p.TopKModesParams):
        return KernelParams.KernelParams.TopKModesParams, TopKModesParams.TopKModesParamsT(value.k)
    if isinstance(value, p.SliceFinalizeParams):
        return KernelParams.KernelParams.SliceFinalizeParams, SliceFinalizeParams.SliceFinalizeParamsT(value.pixel_area)
    raise TypeError(f"unsupported kernel params: {type(value).__name__}")


def encode_plan(plan: model_plan.Plan) -> bytes:
    topo = plan.topo_stages()
    stage_ids = {id(stage): index for index, stage in enumerate(topo)}
    shared_boxes: list[model_params.CoveredBoxes] = []
    shared_box_indices: dict[model_params.CoveredBoxes, int] = {}
    stages: list[Stage.StageT] = []

    for stage in topo:
        templates: list[TaskTemplate.TaskTemplateT] = []
        for template in stage.templates:
            boxes = model_params.covered_boxes(template.params)
            covered_ref = -1
            if boxes:
                covered_ref = shared_box_indices.get(boxes, -1)
                if covered_ref < 0:
                    covered_ref = len(shared_boxes)
                    shared_boxes.append(boxes)
                    shared_box_indices[boxes] = covered_ref
            params_type, params = _params(template.params)
            deps = template.deps
            templates.append(
                TaskTemplate.TaskTemplateT(
                    name=template.name,
                    plane=_PLANE[template.plane],
                    kernel=template.kernel,
                    domain=_domain(template.domain),
                    inputs=[_field_ref(ref, template.domain) for ref in template.inputs],
                    outputs=[OutputRef.OutputRefT(ref.field.field, ref.field.version, _buffer_spec(ref.buffer)) for ref in template.outputs],
                    deps=DependencyRule.DependencyRuleT(
                        kind=0 if deps.kind == "None" else 1,
                        width=deps.width,
                        faces=list(deps.faces),
                        haloInputs=list(deps.halo_inputs),
                    ),
                    coveredBoxesRef=covered_ref,
                    graphReduce=_graph_reduce(template.graph_reduce),
                    paramsType=params_type,
                    params=params,
                )
            )
        stages.append(
            Stage.StageT(
                name=stage.name,
                plane=_PLANE[stage.plane],
                after=[stage_ids[id(parent)] for parent in stage.after],
                templates=templates,
            )
        )

    covered = [
        CoveredBoxList.CoveredBoxListT(
            [CoveredBox.CoveredBoxT(list(lo), list(hi)) for lo, hi in boxes]
        )
        for boxes in shared_boxes
    ]
    builder = flatbuffers.Builder(1024)
    root = Plan.PlanT(stages=stages, sharedCoveredBoxes=covered).Pack(builder)
    builder.Finish(root, file_identifier=b"KPLN")
    return bytes(builder.Output())
