from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .buffer import BufferSpec
from .ctx import LoweringContext
from .plan import FieldRef, OutputRef, Stage


@dataclass(frozen=True)
class ReducedField:
    """A field plus the task location containing its reduced value."""

    field: FieldRef
    level: int
    block: int = 0


def default_reduce_fan_in(num_inputs: int) -> int:
    return max(2, min(8, int(math.sqrt(max(1, num_inputs)))))


def resolve_reduce_fan_in(configured: int | None, num_inputs: int) -> int:
    if configured is None:
        return default_reduce_fan_in(num_inputs)
    return max(1, int(configured))


def _contiguous_groups(input_blocks: Sequence[int], fan_in: int) -> list[list[int]]:
    width = max(1, int(fan_in))
    return [
        [int(block) for block in input_blocks[i : i + width]]
        for i in range(0, len(input_blocks), width)
    ]


def _fallback_home_rank(
    step: int, level: int, block: int, num_localities: int
) -> int:
    mask = (1 << 64) - 1
    value_hash = 0xCBF29CE484222325
    for value in (step, level, block):
        value_hash ^= (
            (int(value) & mask)
            + 0x9E3779B97F4A7C15
            + ((value_hash << 6) & mask)
            + (value_hash >> 2)
        )
        value_hash &= mask
    return int(value_hash % max(1, int(num_localities)))


def _num_localities(ctx: LoweringContext) -> int:
    fn = getattr(ctx.runtime, "num_localities", None)
    if callable(fn):
        try:
            return max(1, int(fn()))
        except Exception:
            return 1
    return 1


def _chunk_home_rank(
    ctx: LoweringContext,
    *,
    step: int,
    level: int,
    block: int,
    num_localities: int,
) -> int:
    fn = getattr(ctx.runtime, "chunk_home_rank", None)
    if callable(fn):
        try:
            return int(fn(int(step), int(level), int(block))) % max(
                1, int(num_localities)
            )
        except TypeError:
            return int(
                fn(step=int(step), level=int(level), block=int(block))
            ) % max(1, int(num_localities))
    return _fallback_home_rank(step, level, block, num_localities)


def reduce_group_plan(
    ctx: LoweringContext,
    *,
    step: int,
    level: int,
    input_blocks: Sequence[int],
    fan_in: int,
) -> tuple[list[int], list[int], list[int]]:
    """Plan one locality-aware reduction round."""

    blocks = [int(block) for block in input_blocks]
    if not blocks:
        return [], [], [0]

    groups = _contiguous_groups(blocks, fan_in)
    num_localities = _num_localities(ctx)
    if num_localities > 1 and len(blocks) > 1 and fan_in > 1:
        buckets: list[list[int]] = [[] for _ in range(num_localities)]
        for block in blocks:
            rank = _chunk_home_rank(
                ctx,
                step=step,
                level=level,
                block=block,
                num_localities=num_localities,
            )
            buckets[rank].append(block)
        locality_groups = [
            group
            for bucket in buckets
            for group in _contiguous_groups(bucket, fan_in)
            if group
        ]
        if locality_groups and len(locality_groups) < len(blocks):
            groups = locality_groups

    ordered_blocks = [block for group in groups for block in group]
    group_offsets = [0]
    for group in groups:
        group_offsets.append(group_offsets[-1] + len(group))
    output_blocks = [0] if len(groups) == 1 else [group[0] for group in groups]
    return ordered_blocks, output_blocks, group_offsets


def graph_reduce_params(
    *,
    fan_in: int,
    num_inputs: int,
    input_blocks: Sequence[int] | None = None,
    output_blocks: Sequence[int] | None = None,
    group_offsets: Sequence[int] | None = None,
    input_base: int = 0,
    output_base: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the executor's graph-reduction contract in one place."""

    params: dict[str, Any] = {
        "graph_kind": "reduce",
        "fan_in": int(fan_in),
        "num_inputs": int(num_inputs),
        "input_base": int(input_base),
        "output_base": int(output_base),
    }
    if input_blocks is not None:
        params["input_blocks"] = [int(block) for block in input_blocks]
    if output_blocks is not None:
        params["output_blocks"] = [int(block) for block in output_blocks]
    if group_offsets is not None:
        params["group_offsets"] = [int(offset) for offset in group_offsets]
    if extra:
        reserved = params.keys() & extra.keys()
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                f"reduction-specific params cannot override graph topology: {names}"
            )
        params.update(extra)
    return params


class GraphReductionBuilder:
    """Lower graph reductions while owning their stages and producer edges."""

    def __init__(self, ctx: LoweringContext) -> None:
        self.ctx = ctx
        self.stages: list[Stage] = []
        self._producers: dict[int, Stage] = {}

    def add_stage(self, stage: Stage, *, outputs: Sequence[FieldRef] = ()) -> None:
        if not any(existing is stage for existing in self.stages):
            self.stages.append(stage)
        for field in outputs:
            self._producers[field.field] = stage

    def producer(self, field: FieldRef) -> Stage | None:
        return self._producers.get(field.field)

    def dependencies(self, fields: Sequence[FieldRef]) -> list[Stage]:
        return [
            producer
            for field in fields
            if (producer := self.producer(field)) is not None
        ]

    def reduce_blocks(
        self,
        *,
        value: ReducedField,
        input_blocks: Sequence[int],
        step: int,
        fan_in: int,
        kernel: str,
        output_buffer: BufferSpec,
        stage_name: str,
        template_name: str,
        temporary_name: str,
        after: Stage,
        extra_params: Mapping[str, Any] | None = None,
        normalize_single: bool = False,
        singleton_template_name: str | None = None,
    ) -> ReducedField:
        current = value
        current_blocks = [int(block) for block in input_blocks]
        if not current_blocks:
            raise ValueError("block reduction requires at least one input block")
        fan_in = max(1, int(fan_in))
        round_index = 0
        tail = after

        while len(current_blocks) > 1 or (
            normalize_single and current_blocks and current_blocks[0] != 0
        ):
            round_fan_in = fan_in if len(current_blocks) > 1 else 1
            input_order, output_blocks, group_offsets = reduce_group_plan(
                self.ctx,
                step=step,
                level=current.level,
                input_blocks=current_blocks,
                fan_in=round_fan_in,
            )
            num_groups = len(output_blocks)
            output_field = (
                current.field
                if num_groups == 1
                else self.ctx.temp_field(temporary_name.format(round=round_index))
            )
            stage = self.ctx.stage(
                stage_name.format(round=round_index),
                plane="graph",
                after=[tail],
            )
            stage.map_blocks(
                name=(
                    singleton_template_name
                    if len(current_blocks) == 1
                    and singleton_template_name is not None
                    else template_name.format(round=round_index)
                ),
                kernel=kernel,
                domain=self.ctx.domain(step=step, level=current.level),
                inputs=[current.field],
                outputs=[OutputRef(output_field, output_buffer)],
                deps={"kind": "None"},
                params=graph_reduce_params(
                    fan_in=round_fan_in,
                    num_inputs=len(current_blocks),
                    input_blocks=input_order,
                    output_blocks=output_blocks,
                    group_offsets=group_offsets,
                    extra=extra_params,
                ),
            )
            self.add_stage(stage, outputs=[output_field])
            current = ReducedField(
                output_field,
                current.level,
                output_blocks[0] if num_groups == 1 else 0,
            )
            current_blocks = output_blocks
            tail = stage
            round_index += 1
            normalize_single = False

        if current_blocks:
            return ReducedField(current.field, current.level, current_blocks[0])
        return current

    def reduce_pairwise(
        self,
        values: Sequence[ReducedField],
        *,
        step: int,
        target_level: int,
        kernel: str,
        output_buffer: BufferSpec,
        stage_name: str,
        template_name: str,
        temporary_name: str,
        order_by_home: bool = False,
        preserve_location: bool = False,
        extra_params: Mapping[str, Any] | None = None,
    ) -> ReducedField:
        if not values:
            raise ValueError("pairwise reduction requires at least one value")
        current = list(values)
        round_index = 0
        while len(current) > 1:
            if order_by_home:
                current = self._order_values_by_home(step=step, values=current)
            next_values: list[ReducedField] = []
            for index in range(0, len(current), 2):
                if index + 1 >= len(current):
                    next_values.append(current[index])
                    continue
                left = current[index]
                right = current[index + 1]
                if preserve_location and left.block != right.block:
                    raise ValueError(
                        "location-preserving reductions require matching blocks"
                    )
                output_level = left.level if preserve_location else target_level
                output_block = left.block if preserve_location else 0
                explicit_blocks = preserve_location
                left_ref = FieldRef(
                    left.field.field,
                    version=left.field.version,
                    domain=self.ctx.domain(
                        step=step,
                        level=left.level,
                        blocks=[left.block] if explicit_blocks else None,
                    ),
                )
                right_ref = FieldRef(
                    right.field.field,
                    version=right.field.version,
                    domain=self.ctx.domain(
                        step=step,
                        level=right.level,
                        blocks=[right.block] if explicit_blocks else None,
                    ),
                )
                output_field = self.ctx.temp_field(
                    temporary_name.format(round=round_index, index=index)
                )
                stage = self.ctx.stage(
                    stage_name.format(round=round_index, index=index),
                    plane="graph",
                    after=self.dependencies([left.field, right.field]),
                )
                stage.map_blocks(
                    name=template_name.format(round=round_index, index=index),
                    kernel=kernel,
                    domain=self.ctx.domain(
                        step=step,
                        level=output_level,
                        blocks=[output_block] if explicit_blocks else None,
                    ),
                    inputs=[left_ref, right_ref],
                    outputs=[OutputRef(output_field, output_buffer)],
                    deps={"kind": "None"},
                    params=graph_reduce_params(
                        fan_in=1,
                        num_inputs=1,
                        output_base=output_block,
                        output_blocks=[output_block] if explicit_blocks else None,
                        extra=extra_params,
                    ),
                )
                self.add_stage(stage, outputs=[output_field])
                next_values.append(
                    ReducedField(output_field, output_level, output_block)
                )
            current = next_values
            round_index += 1
        return current[0]

    def _order_values_by_home(
        self, *, step: int, values: Sequence[ReducedField]
    ) -> list[ReducedField]:
        ordered = list(values)
        num_localities = _num_localities(self.ctx)
        if num_localities <= 1 or len(ordered) <= 2:
            return ordered

        buckets: list[list[ReducedField]] = [[] for _ in range(num_localities)]
        for value in ordered:
            rank = _chunk_home_rank(
                self.ctx,
                step=step,
                level=value.level,
                block=value.block,
                num_localities=num_localities,
            )
            buckets[rank].append(value)

        grouped: list[ReducedField] = []
        leftovers: list[ReducedField] = []
        for bucket in buckets:
            paired = len(bucket) - (len(bucket) % 2)
            grouped.extend(bucket[:paired])
            leftovers.extend(bucket[paired:])
        grouped.extend(leftovers)
        return grouped if len(grouped) == len(ordered) else ordered
