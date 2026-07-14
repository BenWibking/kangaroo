from __future__ import annotations

from analysis.buffer import BufferSpec, DType, FixedShape, InitPolicy
from analysis.ctx import LoweringContext
from analysis.plan import FieldRef
from analysis.reduction import (
    GraphReductionBuilder,
    ReducedField,
    graph_reduce_spec,
    reduce_group_plan,
)


class _Runtime:
    def __init__(self, num_localities: int = 1) -> None:
        self._next_field = 100
        self._num_localities = num_localities

    def alloc_field_id(self, name: str) -> int:
        del name
        field = self._next_field
        self._next_field += 1
        return field

    def num_localities(self) -> int:
        return self._num_localities

    def chunk_home_rank(self, step: int, level: int, block: int) -> int:
        del step, level
        return block % self._num_localities


def _ctx(num_localities: int = 1) -> LoweringContext:
    return LoweringContext(
        runtime=_Runtime(num_localities),
        runmeta=None,
        dataset=None,
    )


def _buffer() -> BufferSpec:
    return BufferSpec(DType.F64, FixedShape((4,)), InitPolicy.ZERO)


def test_reduce_group_plan_keeps_locality_groups_together() -> None:
    inputs, outputs, offsets = reduce_group_plan(
        _ctx(2),
        step=0,
        level=0,
        input_blocks=list(range(8)),
        fan_in=2,
    )

    assert inputs == [0, 2, 4, 6, 1, 3, 5, 7]
    assert outputs == [0, 4, 1, 5]
    assert offsets == [0, 2, 4, 6, 8]


def test_graph_topology_is_a_typed_plan_record() -> None:
    spec = graph_reduce_spec(fan_in=2, num_inputs=4)
    assert spec.fan_in == 2
    assert spec.num_inputs == 4


def test_block_reduction_builder_rejects_fan_in_one() -> None:
    ctx = _ctx()
    builder = GraphReductionBuilder(ctx)
    accumulate = ctx.stage("accumulate")
    source = FieldRef(10)
    builder.add_stage(accumulate, outputs=[source])

    try:
        builder.reduce_blocks(
            value=ReducedField(source, level=0),
            input_blocks=[0, 1],
            step=0,
            fan_in=1,
            kernel="sum",
            output_buffer=_buffer(),
            stage_name="reduce",
            template_name="reduce_{round}",
            temporary_name="temporary_{round}",
            after=accumulate,
        )
    except ValueError as exc:
        assert "fan_in must be >= 2" in str(exc)
    else:
        raise AssertionError("expected fan_in=1 to be rejected")

    assert builder.stages == [accumulate]


def test_block_reduction_builder_owns_topology_and_producer_edges() -> None:
    ctx = _ctx()
    builder = GraphReductionBuilder(ctx)
    accumulate = ctx.stage("accumulate")
    source = FieldRef(10)
    builder.add_stage(accumulate, outputs=[source])

    reduced = builder.reduce_blocks(
        value=ReducedField(source, level=0),
        input_blocks=list(range(16)),
        step=0,
        fan_in=4,
        kernel="sum",
        output_buffer=_buffer(),
        stage_name="reduce",
        template_name="reduce_{round}",
        temporary_name="temporary_{round}",
        after=accumulate,
    )

    reductions = builder.stages[1:]
    assert len(reductions) == 2
    assert reductions[0].after == [accumulate]
    assert reductions[1].after == [reductions[0]]
    assert reductions[0].templates[0].graph_reduce.output_blocks == (0, 4, 8, 12)
    assert reductions[1].templates[0].graph_reduce.input_blocks == (0, 4, 8, 12)
    assert reduced.block == 0
    assert builder.producer(reduced.field) is reductions[-1]


def test_block_reduction_normalizes_single_nonzero_block() -> None:
    ctx = _ctx()
    builder = GraphReductionBuilder(ctx)
    accumulate = ctx.stage("accumulate")
    source = FieldRef(10)
    builder.add_stage(accumulate, outputs=[source])

    reduced = builder.reduce_blocks(
        value=ReducedField(source, level=0, block=7),
        input_blocks=[7],
        step=0,
        fan_in=2,
        kernel="sum",
        output_buffer=_buffer(),
        stage_name="reduce",
        template_name="reduce_{round}",
        singleton_template_name="reduce_single",
        temporary_name="temporary_{round}",
        after=accumulate,
        normalize_single=True,
    )

    template = builder.stages[-1].templates[0]
    assert template.name == "reduce_single"
    assert template.graph_reduce.input_blocks == (7,)
    assert template.graph_reduce.output_blocks == (0,)
    assert template.graph_reduce.group_offsets == (0, 1)
    assert reduced == ReducedField(source, level=0, block=0)


def test_pairwise_builder_tracks_cross_level_dependencies() -> None:
    ctx = _ctx()
    builder = GraphReductionBuilder(ctx)
    left_stage = ctx.stage("left")
    right_stage = ctx.stage("right")
    left = FieldRef(10)
    right = FieldRef(11)
    builder.add_stage(left_stage, outputs=[left])
    builder.add_stage(right_stage, outputs=[right])

    reduced = builder.reduce_pairwise(
        [ReducedField(left, level=0), ReducedField(right, level=1)],
        step=0,
        target_level=0,
        kernel="add",
        output_buffer=_buffer(),
        stage_name="add",
        template_name="add_{round}_{index}",
        temporary_name="temporary_{round}_{index}",
    )

    stage = builder.producer(reduced.field)
    assert stage is not None
    assert stage.after == [left_stage, right_stage]
    assert [ref.domain.level for ref in stage.templates[0].inputs] == [0, 1]
    assert stage.templates[0].graph_reduce == graph_reduce_spec(
        fan_in=1, num_inputs=1
    )


def test_pairwise_builder_preserves_nonzero_input_block() -> None:
    ctx = _ctx()
    builder = GraphReductionBuilder(ctx)
    left = FieldRef(10)
    right = FieldRef(11)

    reduced = builder.reduce_pairwise(
        [
            ReducedField(left, level=0, block=7),
            ReducedField(right, level=1, block=7),
        ],
        step=0,
        target_level=0,
        kernel="add",
        output_buffer=_buffer(),
        stage_name="add",
        template_name="add_{round}_{index}",
        temporary_name="temporary_{round}_{index}",
        preserve_location=True,
    )

    stage = builder.producer(reduced.field)
    assert stage is not None
    template = stage.templates[0]
    assert [ref.domain.blocks for ref in template.inputs] == [[7], [7]]
    assert template.graph_reduce == graph_reduce_spec(
        fan_in=1,
        num_inputs=1,
        input_base=7,
        output_base=7,
        output_blocks=[7],
    )
    assert reduced.block == 7
