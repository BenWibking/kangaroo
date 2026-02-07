from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ctx import LoweringContext
from .ops import UniformProjection, UniformSlice, VorticityMag
from .plan import Plan, Stage


@dataclass(frozen=True)
class FieldHandle:
    """Symbolic field produced or consumed by the imperative pipeline API."""

    pipeline: "Pipeline"
    field: int
    name: str | None = None


class Pipeline:
    """Imperative facade that records a DAG and lowers to the existing Plan IR."""

    def __init__(self, *, runtime: Any, runmeta: Any, dataset: Any) -> None:
        self.runtime = runtime
        self.runmeta = runmeta
        self.dataset = dataset
        core_runtime = getattr(runtime, "_rt", runtime)
        self._ctx = LoweringContext(runtime=core_runtime, runmeta=runmeta, dataset=dataset)
        self._stages: list[Stage] = []
        self._frontier: list[Stage] = []
        self._name_counters: dict[str, int] = {}

    def _unique_name(self, base: str) -> str:
        idx = self._name_counters.get(base, 0)
        self._name_counters[base] = idx + 1
        if idx == 0:
            return base
        return f"{base}_{idx}"

    def _as_field_id(self, field: int | FieldHandle) -> int:
        if isinstance(field, FieldHandle):
            if field.pipeline is not self:
                raise ValueError("field handle belongs to a different pipeline")
            return field.field
        return int(field)

    def _root_stages(self, fragment: list[Stage]) -> list[Stage]:
        fragment_ids = {id(stage) for stage in fragment}
        roots: list[Stage] = []
        for stage in fragment:
            has_fragment_parent = any(id(parent) in fragment_ids for parent in stage.after)
            if not has_fragment_parent:
                roots.append(stage)
        return roots

    def _leaf_stages(self, fragment: list[Stage]) -> list[Stage]:
        leaves: list[Stage] = []
        for stage in fragment:
            if not any(stage in child.after for child in fragment):
                leaves.append(stage)
        return leaves

    def _sink_fields(self, fragment: list[Stage]) -> list[int]:
        produced: set[int] = set()
        consumed: set[int] = set()
        for stage in fragment:
            for tmpl in stage.templates:
                produced.update(ref.field for ref in tmpl.outputs)
                consumed.update(ref.field for ref in tmpl.inputs)
        sinks = sorted(produced - consumed)
        if not sinks:
            raise RuntimeError("failed to infer output field from fragment")
        return sinks

    def _append_fragment(self, fragment: list[Stage]) -> None:
        if not fragment:
            return

        roots = self._root_stages(fragment)
        if self._frontier:
            for stage in roots:
                for parent in self._frontier:
                    if parent not in stage.after:
                        stage.after.append(parent)

        self._stages.extend(fragment)
        self._frontier = self._leaf_stages(fragment)

    def field(self, name_or_id: str | int) -> FieldHandle:
        if isinstance(name_or_id, str):
            fid = int(self.dataset.field_id(name_or_id))
            return FieldHandle(self, fid, name_or_id)
        return FieldHandle(self, int(name_or_id), None)

    def vorticity_mag(
        self,
        vel_field: int | FieldHandle | tuple[int | FieldHandle, int | FieldHandle, int | FieldHandle],
        *,
        out: str | None = None,
        stencil_radius: int = 1,
    ) -> FieldHandle:
        out_name = out or self._unique_name("vortmag")
        if isinstance(vel_field, tuple):
            lowered_vel: int | tuple[int, int, int] = tuple(
                self._as_field_id(comp) for comp in vel_field
            )
        else:
            lowered_vel = self._as_field_id(vel_field)

        fragment = VorticityMag(
            vel_field=lowered_vel,
            out_name=out_name,
            stencil_radius=stencil_radius,
        ).lower(self._ctx)
        self._append_fragment(fragment)
        out_field = self._sink_fields(fragment)[-1]
        return FieldHandle(self, out_field, out_name)

    def uniform_slice(
        self,
        field: int | FieldHandle,
        *,
        axis: str | int,
        coord: float,
        rect: tuple[float, float, float, float],
        resolution: tuple[int, int],
        out: str | None = None,
        bytes_per_value: int = 4,
        reduce_fan_in: int | None = None,
        amr_cell_average: bool = False,
    ) -> FieldHandle:
        out_name = out or self._unique_name("slice")
        fragment = UniformSlice(
            field=self._as_field_id(field),
            axis=axis,
            coord=coord,
            rect=rect,
            resolution=resolution,
            out_name=out_name,
            bytes_per_value=bytes_per_value,
            reduce_fan_in=reduce_fan_in,
            amr_cell_average=amr_cell_average,
        ).lower(self._ctx)
        self._append_fragment(fragment)
        out_field = self._sink_fields(fragment)[-1]
        return FieldHandle(self, out_field, out_name)

    def uniform_projection(
        self,
        field: int | FieldHandle,
        *,
        axis: str | int,
        axis_bounds: tuple[float, float],
        rect: tuple[float, float, float, float],
        resolution: tuple[int, int],
        out: str | None = None,
        bytes_per_value: int = 4,
        reduce_fan_in: int | None = None,
        amr_cell_average: bool = True,
    ) -> FieldHandle:
        out_name = out or self._unique_name("projection")
        fragment = UniformProjection(
            field=self._as_field_id(field),
            axis=axis,
            axis_bounds=axis_bounds,
            rect=rect,
            resolution=resolution,
            out_name=out_name,
            bytes_per_value=bytes_per_value,
            reduce_fan_in=reduce_fan_in,
            amr_cell_average=amr_cell_average,
        ).lower(self._ctx)
        self._append_fragment(fragment)
        out_field = self._sink_fields(fragment)[-1]
        return FieldHandle(self, out_field, out_name)

    def plan(self) -> Plan:
        return Plan(stages=list(self._stages))

    def run(self) -> None:
        self.runtime.run(self.plan(), runmeta=self.runmeta, dataset=self.dataset)


def pipeline(*, runtime: Any, runmeta: Any, dataset: Any) -> Pipeline:
    return Pipeline(runtime=runtime, runmeta=runmeta, dataset=dataset)
