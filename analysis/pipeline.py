from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ctx import LoweringContext
from .ops import (
    Histogram1D,
    Histogram2D,
    UniformProjection,
    UniformSlice,
    VorticityMag,
    histogram_edges_1d,
    histogram_edges_2d,
)
from .plan import Domain, FieldRef, Plan, Stage
from .runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


@dataclass(frozen=True)
class FieldHandle:
    """Symbolic field produced or consumed by the imperative pipeline API."""

    pipeline: "Pipeline"
    field: int
    name: str | None = None


@dataclass(frozen=True)
class Histogram1DHandle:
    counts: FieldHandle
    hist_range: tuple[float, float]
    bins: int

    @property
    def edges(self) -> list[float]:
        return histogram_edges_1d(self.hist_range, self.bins)


@dataclass(frozen=True)
class Histogram2DHandle:
    counts: FieldHandle
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    bins: tuple[int, int]

    @property
    def edges(self) -> tuple[list[float], list[float]]:
        return histogram_edges_2d(self.x_range, self.y_range, self.bins)


@dataclass(frozen=True)
class ParticleArrayHandle:
    pipeline: "Pipeline"
    field: int
    chunk_count: int
    dtype: str = "float64"

    @property
    def values(self) -> np.ndarray:
        return self.pipeline._particle_materialize_array(self)


@dataclass(frozen=True)
class ParticleMaskHandle:
    pipeline: "Pipeline"
    field: int
    chunk_count: int

    @property
    def values(self) -> np.ndarray:
        return self.pipeline._particle_materialize_mask(self)


class Pipeline:
    """Imperative facade that records a DAG and lowers to the existing Plan IR."""

    def __init__(self, *, runtime: Any, runmeta: Any, dataset: Any) -> None:
        self.runtime = runtime
        self.runmeta = runmeta
        self.dataset = dataset
        bind_dataset = getattr(runtime, "_bind_dataset_handle", None)
        if callable(bind_dataset):
            bind_dataset(dataset)
        core_runtime = getattr(runtime, "_rt", runtime)
        self._ctx = LoweringContext(runtime=core_runtime, runmeta=runmeta, dataset=dataset)
        self._stages: list[Stage] = []
        self._frontier: list[Stage] = []
        self._name_counters: dict[str, int] = {}
        self._particle_stages: list[Stage] = []
        self._particle_frontier: list[Stage] = []
        self._particle_max_chunks: int = 1
        self._particle_executed: bool = False
        self._particle_cache: dict[int, list[np.ndarray]] = {}

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

    def _particle_chunk_count(self, particle_type: str) -> int:
        meta_chunks = getattr(self.runmeta, "particle_species", {})
        if particle_type in meta_chunks:
            return max(1, int(meta_chunks[particle_type]))
        if hasattr(self.dataset, "get_particle_chunk_count"):
            return max(1, int(self.dataset.get_particle_chunk_count(particle_type)))
        return 1

    def _particle_runmeta(self) -> RunMeta:
        n = max(1, int(self._particle_max_chunks))
        boxes = [BlockBox((i, 0, 0), (i, 0, 0)) for i in range(n)]
        return RunMeta(
            steps=[
                StepMeta(
                    step=0,
                    levels=[LevelMeta(geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1), boxes=boxes)],
                )
            ],
            particle_species=dict(getattr(self.runmeta, "particle_species", {})),
        )

    def _alloc_runtime_field(self, prefix: str) -> int:
        return int(self.runtime.alloc_field_id(f"{prefix}_{self._unique_name('particle')}"))

    def _append_particle_stage(self, stage: Stage, *, chunk_count: int) -> None:
        if self._particle_frontier:
            for parent in self._particle_frontier:
                if parent not in stage.after:
                    stage.after.append(parent)
        self._particle_stages.append(stage)
        self._particle_frontier = [stage]
        self._particle_max_chunks = max(self._particle_max_chunks, int(chunk_count))
        self._particle_executed = False
        self._particle_cache.clear()

    def _particle_reduce_fan_in(self, num_inputs: int) -> int:
        return max(2, int(np.sqrt(max(1, num_inputs))))

    def _append_particle_reduce_tree(
        self,
        *,
        input_field: int,
        chunk_count: int,
        kernel: str,
        output_bytes: int,
        params: dict[str, Any],
    ) -> int:
        num_inputs = int(chunk_count)
        fan_in = self._particle_reduce_fan_in(num_inputs)
        in_field = int(input_field)
        reduce_idx = 0
        while num_inputs > 1:
            num_groups = (num_inputs + fan_in - 1) // fan_in
            out_field = in_field if num_groups == 1 else self._alloc_runtime_field("particle_reduce")
            reduce_stage = Stage(name=self._unique_name(f"{kernel}_reduce"), plane="graph")
            reduce_params = {
                "graph_kind": "reduce",
                "fan_in": fan_in,
                "num_inputs": num_inputs,
                "input_base": 0,
                "output_base": 0,
                **params,
            }
            if reduce_idx == 0:
                reduce_params["input_blocks"] = list(range(chunk_count))
            reduce_stage.map_blocks(
                name=kernel,
                kernel=kernel,
                domain=Domain(step=0, level=0),
                inputs=[FieldRef(in_field)],
                outputs=[FieldRef(out_field)],
                output_bytes=[int(output_bytes)],
                deps={"kind": "None"},
                params=reduce_params,
            )
            self._append_particle_stage(reduce_stage, chunk_count=max(1, num_groups))
            in_field = out_field
            num_inputs = num_groups
            reduce_idx += 1
        return in_field

    def _particle_import_array(
        self, values: np.ndarray, *, dtype: str, chunk_count: int
    ) -> ParticleArrayHandle | ParticleMaskHandle:
        arr = np.asarray(values)
        fid = self._alloc_runtime_field("particle_input")
        if chunk_count <= 1:
            chunks = [arr]
        else:
            chunks = np.array_split(arr, chunk_count)
        for block, chunk in enumerate(chunks):
            if dtype == "float64":
                payload = np.ascontiguousarray(chunk, dtype=np.float64).tobytes(order="C")
            elif dtype == "int64":
                payload = np.ascontiguousarray(chunk, dtype=np.int64).tobytes(order="C")
            elif dtype == "mask_u8":
                payload = np.ascontiguousarray(chunk, dtype=np.uint8).tobytes(order="C")
            else:
                raise ValueError(f"unsupported particle import dtype '{dtype}'")
            self.dataset._h.set_chunk_ref(0, 0, fid, 0, block, payload)
        self._particle_max_chunks = max(self._particle_max_chunks, int(chunk_count))
        if dtype == "mask_u8":
            return ParticleMaskHandle(self, fid, int(chunk_count))
        out_dtype = "int64" if dtype == "int64" else "float64"
        return ParticleArrayHandle(self, fid, int(chunk_count), out_dtype)

    def _coerce_particle_array_handle(
        self, value: ParticleArrayHandle | np.ndarray | list[float], *, chunk_count: int | None = None
    ) -> ParticleArrayHandle:
        if isinstance(value, ParticleArrayHandle):
            if value.pipeline is not self:
                raise ValueError("particle handle belongs to a different pipeline")
            return value
        arr = np.asarray(value, dtype=np.float64)
        target_chunks = int(chunk_count or max(1, self._particle_max_chunks))
        return self._particle_import_array(arr, dtype="float64", chunk_count=target_chunks)  # type: ignore[return-value]

    def _coerce_particle_mask_handle(
        self, value: ParticleMaskHandle | np.ndarray | list[bool], *, chunk_count: int | None = None
    ) -> ParticleMaskHandle:
        if isinstance(value, ParticleMaskHandle):
            if value.pipeline is not self:
                raise ValueError("particle mask belongs to a different pipeline")
            return value
        arr = np.asarray(value, dtype=np.uint8)
        target_chunks = int(chunk_count or max(1, self._particle_max_chunks))
        return self._particle_import_array(arr, dtype="mask_u8", chunk_count=target_chunks)  # type: ignore[return-value]

    def _particle_materialize_chunks(self, field: int, *, chunk_count: int, dtype: str) -> list[np.ndarray]:
        key = (field << 16) ^ chunk_count
        cached = self._particle_cache.get(key)
        if cached is not None:
            return cached
        self._ensure_particle_executed()
        out: list[np.ndarray] = []
        for block in range(chunk_count):
            raw = self.runtime.get_task_chunk(
                step=0, level=0, field=field, version=0, block=block, dataset=self.dataset
            )
            if dtype == "float64":
                out.append(np.frombuffer(raw, dtype=np.float64).copy())
            elif dtype == "int64":
                out.append(np.frombuffer(raw, dtype=np.int64).copy())
            elif dtype == "mask_u8":
                out.append(np.frombuffer(raw, dtype=np.uint8).astype(bool, copy=False).copy())
            else:
                raise ValueError(f"unsupported particle dtype '{dtype}'")
        self._particle_cache[key] = out
        return out

    def _particle_materialize_array(self, handle: ParticleArrayHandle) -> np.ndarray:
        chunks = self._particle_materialize_chunks(
            handle.field, chunk_count=handle.chunk_count, dtype=("int64" if handle.dtype == "int64" else "float64")
        )
        if not chunks:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(chunks)

    def _particle_materialize_mask(self, handle: ParticleMaskHandle) -> np.ndarray:
        chunks = self._particle_materialize_chunks(handle.field, chunk_count=handle.chunk_count, dtype="mask_u8")
        if not chunks:
            return np.zeros(0, dtype=bool)
        return np.concatenate(chunks).astype(bool, copy=False)

    def _ensure_particle_executed(self) -> None:
        if self._particle_executed:
            return
        if not self._particle_stages:
            self._particle_executed = True
            return
        self.runtime.run(Plan(stages=list(self._particle_stages)), runmeta=self._particle_runmeta(), dataset=self.dataset)
        self._particle_executed = True

    def _particle_scalar_from_field(self, field: int, *, dtype: str) -> float | int:
        self._ensure_particle_executed()
        raw = self.runtime.get_task_chunk(step=0, level=0, field=field, version=0, block=0, dataset=self.dataset)
        if dtype == "float64":
            arr = np.frombuffer(raw, dtype=np.float64)
            return float(arr[0]) if arr.size else float("nan")
        if dtype == "int64":
            arr = np.frombuffer(raw, dtype=np.int64)
            return int(arr[0]) if arr.size else 0
        raise ValueError(f"unsupported scalar dtype '{dtype}'")

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
        bytes_per_value: int | None = None,
        reduce_fan_in: int | None = None,
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
        bytes_per_value: int | None = None,
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

    def histogram1d(
        self,
        field: int | FieldHandle,
        *,
        hist_range: tuple[float, float],
        bins: int,
        out: str | None = None,
        weights: int | FieldHandle | None = None,
        bytes_per_value: int | None = None,
        reduce_fan_in: int | None = None,
    ) -> Histogram1DHandle:
        out_name = out or self._unique_name("histogram1d")
        weight_field = self._as_field_id(weights) if weights is not None else None
        fragment = Histogram1D(
            field=self._as_field_id(field),
            hist_range=hist_range,
            bins=bins,
            out_name=out_name,
            weight_field=weight_field,
            bytes_per_value=bytes_per_value,
            reduce_fan_in=reduce_fan_in,
        ).lower(self._ctx)
        self._append_fragment(fragment)
        out_field = self._sink_fields(fragment)[-1]
        return Histogram1DHandle(
            counts=FieldHandle(self, out_field, out_name),
            hist_range=hist_range,
            bins=int(bins),
        )

    def histogram2d(
        self,
        x_field: int | FieldHandle,
        y_field: int | FieldHandle,
        *,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        bins: tuple[int, int],
        out: str | None = None,
        weights: int | FieldHandle | None = None,
        weight_mode: str = "input",
        bytes_per_value: int | None = None,
        reduce_fan_in: int | None = None,
    ) -> Histogram2DHandle:
        out_name = out or self._unique_name("histogram2d")
        weight_field = self._as_field_id(weights) if weights is not None else None
        fragment = Histogram2D(
            x_field=self._as_field_id(x_field),
            y_field=self._as_field_id(y_field),
            x_range=x_range,
            y_range=y_range,
            bins=bins,
            out_name=out_name,
            weight_field=weight_field,
            weight_mode=weight_mode,
            bytes_per_value=bytes_per_value,
            reduce_fan_in=reduce_fan_in,
        ).lower(self._ctx)
        self._append_fragment(fragment)
        out_field = self._sink_fields(fragment)[-1]
        return Histogram2DHandle(
            counts=FieldHandle(self, out_field, out_name),
            x_range=x_range,
            y_range=y_range,
            bins=(int(bins[0]), int(bins[1])),
        )

    # Particle operators (pipeline facade)
    def particle_field(self, particle_type: str, field: str) -> ParticleArrayHandle:
        chunk_count = self._particle_chunk_count(particle_type)
        out_fid = self._alloc_runtime_field("particle_field")
        stage = Stage(name=self._unique_name("particle_field"))
        stage.map_blocks(
            name="particle_load_field_chunk_f64",
            kernel="particle_load_field_chunk_f64",
            domain=Domain(step=0, level=0, blocks=list(range(chunk_count))),
            inputs=[],
            outputs=[FieldRef(out_fid)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={"particle_type": particle_type, "field_name": field},
        )
        self._append_particle_stage(stage, chunk_count=chunk_count)
        return ParticleArrayHandle(self, out_fid, chunk_count, "float64")

    def particle_equals(
        self, values: ParticleArrayHandle | np.ndarray, scalar: int | float
    ) -> ParticleMaskHandle:
        in_h = self._coerce_particle_array_handle(values)
        out_fid = self._alloc_runtime_field("particle_eq")
        stage = Stage(name=self._unique_name("particle_eq"))
        stage.map_blocks(
            name="particle_eq_mask",
            kernel="particle_eq_mask",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(out_fid)],
            output_bytes=[1],
            deps={"kind": "None"},
            params={"scalar": float(scalar)},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        return ParticleMaskHandle(self, out_fid, in_h.chunk_count)

    def particle_isin(
        self, values: ParticleArrayHandle | np.ndarray, scalars: list[int] | np.ndarray
    ) -> ParticleMaskHandle:
        in_h = self._coerce_particle_array_handle(values)
        out_fid = self._alloc_runtime_field("particle_isin")
        stage = Stage(name=self._unique_name("particle_isin"))
        stage.map_blocks(
            name="particle_isin_mask",
            kernel="particle_isin_mask",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(out_fid)],
            output_bytes=[1],
            deps={"kind": "None"},
            params={"values": [float(x) for x in np.asarray(scalars).ravel()]},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        return ParticleMaskHandle(self, out_fid, in_h.chunk_count)

    def particle_isfinite(self, values: ParticleArrayHandle | np.ndarray) -> ParticleMaskHandle:
        in_h = self._coerce_particle_array_handle(values)
        out_fid = self._alloc_runtime_field("particle_isfinite")
        stage = Stage(name=self._unique_name("particle_isfinite"))
        stage.map_blocks(
            name="particle_isfinite_mask",
            kernel="particle_isfinite_mask",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(out_fid)],
            output_bytes=[1],
            deps={"kind": "None"},
            params={},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        return ParticleMaskHandle(self, out_fid, in_h.chunk_count)

    def particle_abs_lt(
        self, values: ParticleArrayHandle | np.ndarray, scalar: float
    ) -> ParticleMaskHandle:
        in_h = self._coerce_particle_array_handle(values)
        out_fid = self._alloc_runtime_field("particle_abs_lt")
        stage = Stage(name=self._unique_name("particle_abs_lt"))
        stage.map_blocks(
            name="particle_abs_lt_mask",
            kernel="particle_abs_lt_mask",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(out_fid)],
            output_bytes=[1],
            deps={"kind": "None"},
            params={"scalar": float(scalar)},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        return ParticleMaskHandle(self, out_fid, in_h.chunk_count)

    def particle_le(
        self, values: ParticleArrayHandle | np.ndarray, scalar: float
    ) -> ParticleMaskHandle:
        in_h = self._coerce_particle_array_handle(values)
        out_fid = self._alloc_runtime_field("particle_le")
        stage = Stage(name=self._unique_name("particle_le"))
        stage.map_blocks(
            name="particle_le_mask",
            kernel="particle_le_mask",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(out_fid)],
            output_bytes=[1],
            deps={"kind": "None"},
            params={"scalar": float(scalar)},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        return ParticleMaskHandle(self, out_fid, in_h.chunk_count)

    def particle_gt(
        self, values: ParticleArrayHandle | np.ndarray, scalar: float
    ) -> ParticleMaskHandle:
        in_h = self._coerce_particle_array_handle(values)
        out_fid = self._alloc_runtime_field("particle_gt")
        stage = Stage(name=self._unique_name("particle_gt"))
        stage.map_blocks(
            name="particle_gt_mask",
            kernel="particle_gt_mask",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(out_fid)],
            output_bytes=[1],
            deps={"kind": "None"},
            params={"scalar": float(scalar)},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        return ParticleMaskHandle(self, out_fid, in_h.chunk_count)

    def particle_and(self, *masks: ParticleMaskHandle | np.ndarray) -> ParticleMaskHandle:
        if not masks:
            raise ValueError("particle_and requires at least one mask")
        out_h = self._coerce_particle_mask_handle(masks[0])
        for mask in masks[1:]:
            rhs_h = self._coerce_particle_mask_handle(mask, chunk_count=out_h.chunk_count)
            if rhs_h.chunk_count != out_h.chunk_count:
                raise ValueError("particle_and inputs must have matching chunk_count")
            fid = self._alloc_runtime_field("particle_and")
            stage = Stage(name=self._unique_name("particle_and"))
            stage.map_blocks(
                name="particle_and_mask",
                kernel="particle_and_mask",
                domain=Domain(step=0, level=0, blocks=list(range(out_h.chunk_count))),
                inputs=[FieldRef(out_h.field), FieldRef(rhs_h.field)],
                outputs=[FieldRef(fid)],
                output_bytes=[1],
                deps={"kind": "None"},
                params={},
            )
            self._append_particle_stage(stage, chunk_count=out_h.chunk_count)
            out_h = ParticleMaskHandle(self, fid, out_h.chunk_count)
        return out_h

    def particle_filter(
        self, values: ParticleArrayHandle | np.ndarray, mask: ParticleMaskHandle | np.ndarray
    ) -> ParticleArrayHandle:
        arr_h = self._coerce_particle_array_handle(values)
        mask_h = self._coerce_particle_mask_handle(mask, chunk_count=arr_h.chunk_count)
        if arr_h.chunk_count != mask_h.chunk_count:
            raise ValueError("particle_filter inputs must have matching chunk_count")
        fid = self._alloc_runtime_field("particle_filter")
        stage = Stage(name=self._unique_name("particle_filter"))
        stage.map_blocks(
            name="particle_filter",
            kernel="particle_filter",
            domain=Domain(step=0, level=0, blocks=list(range(arr_h.chunk_count))),
            inputs=[FieldRef(arr_h.field), FieldRef(mask_h.field)],
            outputs=[FieldRef(fid)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={},
        )
        self._append_particle_stage(stage, chunk_count=arr_h.chunk_count)
        return ParticleArrayHandle(self, fid, arr_h.chunk_count, "float64")

    def particle_subtract(
        self, left: ParticleArrayHandle | np.ndarray, right: ParticleArrayHandle | np.ndarray
    ) -> ParticleArrayHandle:
        a_h = self._coerce_particle_array_handle(left)
        b_h = self._coerce_particle_array_handle(right, chunk_count=a_h.chunk_count)
        if a_h.chunk_count != b_h.chunk_count:
            raise ValueError("particle_subtract inputs must have matching chunk_count")
        fid = self._alloc_runtime_field("particle_subtract")
        stage = Stage(name=self._unique_name("particle_subtract"))
        stage.map_blocks(
            name="particle_subtract",
            kernel="particle_subtract",
            domain=Domain(step=0, level=0, blocks=list(range(a_h.chunk_count))),
            inputs=[FieldRef(a_h.field), FieldRef(b_h.field)],
            outputs=[FieldRef(fid)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={},
        )
        self._append_particle_stage(stage, chunk_count=a_h.chunk_count)
        return ParticleArrayHandle(self, fid, a_h.chunk_count, "float64")

    def particle_distance3(
        self,
        ax: ParticleArrayHandle | np.ndarray,
        ay: ParticleArrayHandle | np.ndarray,
        az: ParticleArrayHandle | np.ndarray,
        bx: ParticleArrayHandle | np.ndarray,
        by: ParticleArrayHandle | np.ndarray,
        bz: ParticleArrayHandle | np.ndarray,
    ) -> ParticleArrayHandle:
        ax_h = self._coerce_particle_array_handle(ax)
        ay_h = self._coerce_particle_array_handle(ay, chunk_count=ax_h.chunk_count)
        az_h = self._coerce_particle_array_handle(az, chunk_count=ax_h.chunk_count)
        bx_h = self._coerce_particle_array_handle(bx, chunk_count=ax_h.chunk_count)
        by_h = self._coerce_particle_array_handle(by, chunk_count=ax_h.chunk_count)
        bz_h = self._coerce_particle_array_handle(bz, chunk_count=ax_h.chunk_count)
        all_chunks = {h.chunk_count for h in (ax_h, ay_h, az_h, bx_h, by_h, bz_h)}
        if len(all_chunks) != 1:
            raise ValueError("particle_distance3 inputs must have matching chunk_count")
        chunk_count = ax_h.chunk_count
        fid = self._alloc_runtime_field("particle_distance3")
        stage = Stage(name=self._unique_name("particle_distance3"))
        stage.map_blocks(
            name="particle_distance3",
            kernel="particle_distance3",
            domain=Domain(step=0, level=0, blocks=list(range(chunk_count))),
            inputs=[
                FieldRef(ax_h.field),
                FieldRef(ay_h.field),
                FieldRef(az_h.field),
                FieldRef(bx_h.field),
                FieldRef(by_h.field),
                FieldRef(bz_h.field),
            ],
            outputs=[FieldRef(fid)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={},
        )
        self._append_particle_stage(stage, chunk_count=chunk_count)
        return ParticleArrayHandle(self, fid, chunk_count, "float64")

    def particle_sum(self, values: ParticleArrayHandle | np.ndarray) -> float:
        in_h = self._coerce_particle_array_handle(values)
        fid = self._alloc_runtime_field("particle_sum")
        stage = Stage(name=self._unique_name("particle_sum"))
        stage.map_blocks(
            name="particle_sum",
            kernel="particle_sum",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(fid)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=in_h.chunk_count,
            kernel="uniform_slice_reduce",
            output_bytes=8,
            params={"bytes_per_value": 8},
        )
        return float(self._particle_scalar_from_field(reduced, dtype="float64"))

    def particle_len(self, values: ParticleArrayHandle | np.ndarray) -> int:
        in_h = self._coerce_particle_array_handle(values)
        fid = self._alloc_runtime_field("particle_len")
        stage = Stage(name=self._unique_name("particle_len"))
        stage.map_blocks(
            name="particle_len_f64",
            kernel="particle_len_f64",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(fid)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=in_h.chunk_count,
            kernel="particle_int64_sum_reduce",
            output_bytes=8,
            params={},
        )
        return int(self._particle_scalar_from_field(reduced, dtype="int64"))

    def particle_min(self, values: ParticleArrayHandle | np.ndarray, *, finite_only: bool = True) -> float:
        in_h = self._coerce_particle_array_handle(values)
        fid = self._alloc_runtime_field("particle_min")
        stage = Stage(name=self._unique_name("particle_min"))
        stage.map_blocks(
            name="particle_min",
            kernel="particle_min",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(fid)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={"finite_only": bool(finite_only)},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=in_h.chunk_count,
            kernel="particle_scalar_min_reduce",
            output_bytes=8,
            params={},
        )
        return float(self._particle_scalar_from_field(reduced, dtype="float64"))

    def particle_max(self, values: ParticleArrayHandle | np.ndarray, *, finite_only: bool = True) -> float:
        in_h = self._coerce_particle_array_handle(values)
        fid = self._alloc_runtime_field("particle_max")
        stage = Stage(name=self._unique_name("particle_max"))
        stage.map_blocks(
            name="particle_max",
            kernel="particle_max",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(fid)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={"finite_only": bool(finite_only)},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=in_h.chunk_count,
            kernel="particle_scalar_max_reduce",
            output_bytes=8,
            params={},
        )
        return float(self._particle_scalar_from_field(reduced, dtype="float64"))

    def particle_count(self, mask: ParticleMaskHandle | np.ndarray) -> int:
        in_h = self._coerce_particle_mask_handle(mask)
        fid = self._alloc_runtime_field("particle_count")
        stage = Stage(name=self._unique_name("particle_count"))
        stage.map_blocks(
            name="particle_count",
            kernel="particle_count",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[FieldRef(fid)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={},
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=in_h.chunk_count,
            kernel="particle_int64_sum_reduce",
            output_bytes=8,
            params={},
        )
        return int(self._particle_scalar_from_field(reduced, dtype="int64"))

    def particle_histogram1d(
        self,
        values: ParticleArrayHandle | np.ndarray,
        *,
        bins: int | np.ndarray,
        hist_range: tuple[float, float] | None = None,
        weights: ParticleArrayHandle | np.ndarray | None = None,
        density: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        in_h = self._coerce_particle_array_handle(values)
        if isinstance(bins, int):
            if hist_range is None:
                raise ValueError("hist_range is required when bins is an integer")
            edges = np.linspace(float(hist_range[0]), float(hist_range[1]), int(bins) + 1)
        else:
            edges = np.asarray(bins, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bins must define at least two edges")
        inputs = [FieldRef(in_h.field)]
        chunk_count = in_h.chunk_count
        if weights is not None:
            w_h = self._coerce_particle_array_handle(weights, chunk_count=chunk_count)
            if w_h.chunk_count != chunk_count:
                raise ValueError("weights must match values chunk_count")
            inputs.append(FieldRef(w_h.field))
        fid = self._alloc_runtime_field("particle_hist1d")
        stage = Stage(name=self._unique_name("particle_hist1d"))
        stage.map_blocks(
            name="particle_histogram1d",
            kernel="particle_histogram1d",
            domain=Domain(step=0, level=0, blocks=list(range(chunk_count))),
            inputs=inputs,
            outputs=[FieldRef(fid)],
            output_bytes=[(edges.size - 1) * 8],
            deps={"kind": "None"},
            params={"edges": [float(x) for x in edges], "density": False},
        )
        self._append_particle_stage(stage, chunk_count=chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=chunk_count,
            kernel="uniform_slice_reduce",
            output_bytes=(edges.size - 1) * 8,
            params={"bytes_per_value": 8},
        )
        counts = self._particle_materialize_chunks(reduced, chunk_count=1, dtype="float64")[0]
        if density:
            total = float(np.sum(counts))
            if total > 0.0:
                widths = np.diff(edges)
                valid = widths > 0.0
                counts = counts.astype(np.float64, copy=True)
                counts[valid] /= total * widths[valid]
        return counts, edges

    def plan(self) -> Plan:
        return Plan(stages=list(self._stages))

    def run(self) -> None:
        if self._stages:
            self.runtime.run(self.plan(), runmeta=self.runmeta, dataset=self.dataset)
        self._ensure_particle_executed()


def pipeline(*, runtime: Any, runmeta: Any, dataset: Any) -> Pipeline:
    return Pipeline(runtime=runtime, runmeta=runmeta, dataset=dataset)
