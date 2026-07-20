from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .buffer import (
    BlockShape,
    BufferSpec,
    DType,
    DynamicShape,
    DynamicUpperBound,
    FixedShape,
    InitPolicy,
    LikeInputShape,
)
from .ctx import LoweringContext
from .kernel_params import (
    FieldExprParams,
    FiniteOnlyParams,
    KernelParams,
    NoKernelParams,
    ParticleFieldParams,
    ParticleHistogramParams,
    ScalarParams,
    TopKModesParams,
    ValuesParams,
)
from .ops import (
    CylindricalFluxSurfaceIntegral,
    FluxSurfaceIntegral,
    Histogram1D,
    Histogram2D,
    ParticleCICProjection,
    ToomreQProfile,
    UniformProjection,
    UniformSlice,
    VorticityMag,
    histogram_edges_1d,
    histogram_edges_2d,
)
from .plan import DependencyRule, Domain, FieldRef, OutputRef, Plan, Stage
from .reduction import (
    GraphReductionBuilder,
    ReducedField,
    default_reduce_fan_in,
)
from .runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


def _fixed_output(field: int, dtype: DType, elements: int) -> OutputRef:
    return OutputRef(
        FieldRef(field),
        BufferSpec(dtype, FixedShape((int(elements),)), InitPolicy.ZERO),
    )


def _block_output(field: int, dtype: DType) -> OutputRef:
    return OutputRef(FieldRef(field), BufferSpec(dtype, BlockShape()))


def _like_output(field: int, dtype: DType, input_index: int = 0) -> OutputRef:
    return OutputRef(FieldRef(field), BufferSpec(dtype, LikeInputShape(input_index)))


def _dynamic_output(field: int, dtype: DType, upper_bound: DynamicUpperBound) -> OutputRef:
    return OutputRef(FieldRef(field), BufferSpec(dtype, DynamicShape(upper_bound)))


@dataclass(frozen=True)
class FieldHandle:
    """Symbolic field produced or consumed by the imperative pipeline API."""

    pipeline: "Pipeline"
    field: int
    name: str | None = None

    def iter_chunks(self) -> list[np.ndarray]:
        """Execute producer ancestry and return descriptor-materialized chunks."""

        return self.pipeline._materialize_mesh_field(self.field)

    def compute(self) -> Any:
        """Execute and return one bounded array or a chunked AMR result."""

        from kangaroo.results import ChunkedArray

        chunks = tuple(self.iter_chunks())
        return chunks[0] if len(chunks) == 1 else ChunkedArray(chunks)


@dataclass(frozen=True)
class Histogram1DHandle:
    counts: FieldHandle
    hist_range: tuple[float, float]
    bins: int

    @property
    def edges(self) -> list[float]:
        return histogram_edges_1d(self.hist_range, self.bins)

    def compute(self) -> Any:
        """Execute and return typed histogram counts and edges."""

        from kangaroo.results import HistogramResult

        counts = self.counts.compute()
        return HistogramResult(np.asarray(counts), np.asarray(self.edges))


@dataclass(frozen=True)
class Histogram2DHandle:
    counts: FieldHandle
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    bins: tuple[int, int]

    @property
    def edges(self) -> tuple[list[float], list[float]]:
        return histogram_edges_2d(self.x_range, self.y_range, self.bins)

    def compute(self) -> Any:
        """Execute and return typed 2D histogram counts and edges."""

        from kangaroo.results import Histogram2DResult

        counts = self.counts.compute()
        x_edges, y_edges = self.edges
        return Histogram2DResult(
            np.asarray(counts), np.asarray(x_edges), np.asarray(y_edges)
        )


@dataclass(frozen=True)
class FluxSurfaceIntegralHandle:
    fluxes: FieldHandle
    radii: tuple[float, ...] = ()
    temperature_bins: tuple[float, ...] | None = None
    components: tuple[str, ...] = (
        "mass_flux_sphere_negative",
        "hydro_energy_flux_sphere_negative",
        "mhd_energy_flux_sphere_negative",
        "passive_scalar_flux_sphere_negative",
        "mass_flux_sphere_positive",
        "hydro_energy_flux_sphere_positive",
        "mhd_energy_flux_sphere_positive",
        "passive_scalar_flux_sphere_positive",
    )

    @property
    def field(self) -> int:
        return self.fluxes.field

    @property
    def name(self) -> str | None:
        return self.fluxes.name

    def compute(self) -> Any:
        """Execute and return a typed flux-surface result."""

        from kangaroo.results import FluxSurfaceIntegralResult

        return FluxSurfaceIntegralResult(
            np.asarray(self.fluxes.compute()),
            self.radii,
            self.components,
            self.temperature_bins,
        )


@dataclass(frozen=True)
class CylindricalFluxSurfaceIntegralHandle:
    fluxes: FieldHandle
    radius: float
    heights: tuple[float, ...] = ()
    temperature_bins: tuple[float, ...] | None = None
    geometric_sections: tuple[str, ...] = ("endcaps", "walls")
    components: tuple[str, ...] = (
        "mass_flux_cylinder_negative",
        "hydro_energy_flux_cylinder_negative",
        "mhd_energy_flux_cylinder_negative",
        "passive_scalar_flux_cylinder_negative",
        "mass_flux_cylinder_positive",
        "hydro_energy_flux_cylinder_positive",
        "mhd_energy_flux_cylinder_positive",
        "passive_scalar_flux_cylinder_positive",
    )

    @property
    def field(self) -> int:
        return self.fluxes.field

    @property
    def name(self) -> str | None:
        return self.fluxes.name

    def compute(self) -> Any:
        """Execute and return a typed cylindrical flux result."""

        from kangaroo.results import CylindricalFluxSurfaceIntegralResult

        return CylindricalFluxSurfaceIntegralResult(
            np.asarray(self.fluxes.compute()),
            self.radius,
            self.heights,
            self.geometric_sections,
            self.components,
            self.temperature_bins,
        )


@dataclass(frozen=True)
class ToomreQProfileHandle:
    """Reduced annular moments used to derive gas Toomre-Q profiles."""

    moments: FieldHandle
    radial_edges: tuple[float, ...]
    z_bounds: tuple[float, float]
    center: tuple[float, float, float]
    gamma: float
    components: tuple[str, ...] = (
        "mass",
        "internal_energy",
        "magnetic_b2_volume",
        "radial_momentum",
        "radial_velocity_second_moment",
        "radial_gravity_moment",
        "sampled_volume",
    )

    @property
    def field(self) -> int:
        return self.moments.field

    @property
    def name(self) -> str | None:
        return self.moments.name

    @property
    def radial_range(self) -> tuple[float, float]:
        return (self.radial_edges[0], self.radial_edges[-1])

    @property
    def bins(self) -> int:
        return len(self.radial_edges) - 1

    @property
    def edges(self) -> np.ndarray:
        return np.asarray(self.radial_edges, dtype=np.float64)

    def compute(self) -> Any:
        """Execute and return typed annular moments and coordinates."""

        from kangaroo.results import ToomreQProfileResult

        return ToomreQProfileResult(
            np.asarray(self.moments.compute()),
            self.edges,
            self.components,
            self.z_bounds,
            self.center,
            self.gamma,
        )


@dataclass(frozen=True)
class ParticleArrayHandle:
    pipeline: "Pipeline"
    field: int
    chunk_count: int
    dtype: str = "float64"

    @property
    def values(self) -> np.ndarray:
        raise RuntimeError(
            "Full particle array materialization is disabled. "
            "Use iter_chunks() and reduction operators."
        )

    def iter_chunks(self) -> list[np.ndarray]:
        dtype = "int64" if self.dtype == "int64" else "float64"
        return self.pipeline._particle_materialize_chunks(
            self.field, chunk_count=self.chunk_count, dtype=dtype
        )

    def compute(self, *, gather: bool = False, max_bytes: int | None = None) -> Any:
        """Return chunked values, or explicitly gather them with a byte limit."""

        from kangaroo.results import ChunkedArray

        result = ChunkedArray(tuple(self.iter_chunks()))
        return result.gather(max_bytes=max_bytes) if gather else result


@dataclass(frozen=True)
class ParticleMaskHandle:
    pipeline: "Pipeline"
    field: int
    chunk_count: int

    @property
    def values(self) -> np.ndarray:
        raise RuntimeError(
            "Full particle mask materialization is disabled. "
            "Use iter_chunks() and reduction operators."
        )

    def iter_chunks(self) -> list[np.ndarray]:
        return self.pipeline._particle_materialize_chunks(
            self.field, chunk_count=self.chunk_count, dtype="mask_u8"
        )

    def compute(self, *, gather: bool = False, max_bytes: int | None = None) -> Any:
        """Return chunked mask values, or explicitly gather them with a limit."""

        from kangaroo.results import ChunkedArray

        result = ChunkedArray(tuple(self.iter_chunks()))
        return result.gather(max_bytes=max_bytes) if gather else result


@dataclass(frozen=True)
class ParticleScalarHandle:
    """Lazy scalar produced by a particle reduction."""

    pipeline: "Pipeline"
    field: int
    dtype: str

    def compute(self) -> float | int:
        """Execute the particle graph and return the local Python scalar."""

        return self.pipeline._particle_scalar_from_field(self.field, dtype=self.dtype)

    def __bool__(self) -> bool:
        raise TypeError("a lazy scalar has no truth value; call compute() first")

    def __float__(self) -> float:
        raise TypeError("a lazy scalar cannot be converted implicitly; call compute() first")

    def __int__(self) -> int:
        raise TypeError("a lazy scalar cannot be converted implicitly; call compute() first")


@dataclass(frozen=True)
class ParticleHistogram1DHandle:
    """Lazy particle histogram with statically known bin edges."""

    counts: ParticleArrayHandle
    edges: np.ndarray
    density: bool = False

    def compute(self) -> tuple[np.ndarray, np.ndarray]:
        """Execute and materialize histogram counts and edges."""

        counts = self.counts.iter_chunks()[0]
        if self.density:
            total = float(np.sum(counts))
            if total > 0.0:
                widths = np.diff(self.edges)
                valid = widths > 0.0
                counts = counts.astype(np.float64, copy=True)
                counts[valid] /= total * widths[valid]
        return counts, self.edges.copy()


@dataclass(frozen=True)
class ParticleTopKHandle:
    """Lazy top-k particle modes and their counts."""

    pipeline: "Pipeline"
    field: int
    k: int

    def compute(self) -> tuple[np.ndarray, np.ndarray]:
        """Execute and split the packed values/counts output."""

        self.pipeline._ensure_particle_executed()
        raw = self.pipeline.runtime.get_task_chunk(
            step=0,
            level=0,
            field=self.field,
            version=0,
            block=0,
            dataset=self.pipeline.dataset,
        )
        arr = np.frombuffer(raw, dtype=np.float64)
        if arr.size < 2 * self.k:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        return arr[: self.k].copy(), arr[self.k : 2 * self.k].copy()


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
        self._field_producers: dict[int, list[Stage]] = {}
        self._name_counters: dict[str, int] = {}
        self._particle_stages: list[Stage] = []
        self._particle_producers: dict[int, list[Stage]] = {}
        self._particle_max_chunks: int = 1
        self._particle_executed: bool = False
        self._particle_cache: dict[int, list[np.ndarray]] = {}
        self._derived_builders: dict[str, Callable[["Pipeline"], int | FieldHandle]] = {}
        self._derived_cache: dict[str, FieldHandle] = {}

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

    def _alloc_field_id(self, name: str) -> int:
        alloc = getattr(self.runtime, "alloc_field_id", None)
        if callable(alloc):
            return int(alloc(name))
        core_rt = getattr(self.runtime, "_rt", None)
        if core_rt is not None and hasattr(core_rt, "alloc_field_id"):
            return int(core_rt.alloc_field_id(name))
        raise RuntimeError("runtime does not provide alloc_field_id")

    def _append_particle_stage(self, stage: Stage, *, chunk_count: int) -> None:
        for tmpl in stage.templates:
            for input_ref in tmpl.inputs:
                for parent in self._particle_producers.get(input_ref.field, ()):
                    if parent is not stage and parent not in stage.after:
                        stage.after.append(parent)
        self._particle_stages.append(stage)
        for tmpl in stage.templates:
            for output_ref in tmpl.outputs:
                self._particle_producers.setdefault(output_ref.field.field, []).append(stage)
        self._particle_max_chunks = max(self._particle_max_chunks, int(chunk_count))
        self._particle_executed = False
        self._particle_cache.clear()

    def _append_particle_reduce_tree(
        self,
        *,
        input_field: int,
        chunk_count: int,
        kernel: str,
        output_buffer: BufferSpec,
        params: KernelParams = NoKernelParams(),
    ) -> int:
        if chunk_count <= 1:
            return int(input_field)
        source_producers = self._particle_producers.get(int(input_field), [])
        if not source_producers:
            raise RuntimeError("particle reduction requires a producer stage")

        reductions = GraphReductionBuilder(self._ctx)
        source = FieldRef(int(input_field))
        source_stage = source_producers[-1]
        reductions.add_stage(source_stage, outputs=[source])
        reduced = reductions.reduce_blocks(
            value=ReducedField(source, level=0),
            input_blocks=list(range(chunk_count)),
            step=0,
            fan_in=default_reduce_fan_in(chunk_count),
            kernel=kernel,
            output_buffer=output_buffer,
            stage_name=f"{kernel}_reduce_{{round}}",
            template_name=kernel,
            temporary_name="particle_reduce_{round}",
            after=source_stage,
            params=params,
        )
        for stage in reductions.stages[1:]:
            graph_reduce = stage.templates[0].graph_reduce
            if graph_reduce is None:
                raise RuntimeError("particle reduction is missing graph topology")
            group_count = len(graph_reduce.output_blocks)
            self._append_particle_stage(stage, chunk_count=max(1, group_count))
        return reduced.field.field

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
                array = np.ascontiguousarray(chunk, dtype=np.float64)
                dtype_tag = DType.F64.value
            elif dtype == "int64":
                array = np.ascontiguousarray(chunk, dtype=np.int64)
                dtype_tag = DType.I64.value
            elif dtype == "mask_u8":
                array = np.ascontiguousarray(chunk, dtype=np.uint8)
                dtype_tag = DType.U8.value
            else:
                raise ValueError(f"unsupported particle import dtype '{dtype}'")
            self.dataset._h.set_chunk_ref(
                0, 0, fid, 0, block, array.tobytes(order="C"), dtype_tag, list(array.shape)
            )
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

    def _ensure_particle_executed(self, *, progress_bar: bool = False) -> None:
        if self._particle_executed:
            return
        if not self._particle_stages:
            self._particle_executed = True
            return
        self.runtime.run(
            Plan(stages=list(self._particle_stages)),
            runmeta=self._particle_runmeta(),
            dataset=self.dataset,
            progress_bar=progress_bar,
        )
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

    def _mesh_field_coordinates(self, field: int) -> list[tuple[int, int, int]]:
        producers = self._field_producers.get(int(field), [])
        if not producers:
            coordinates: list[tuple[int, int, int]] = []
            for level, level_meta in enumerate(
                self.runmeta.steps[self.dataset.step].levels
            ):
                coordinates.extend(
                    (self.dataset.step, level, block)
                    for block in range(len(level_meta.boxes))
                )
            return coordinates
        coordinates = []
        for tmpl in producers[-1].templates:
            if not any(output.field.field == int(field) for output in tmpl.outputs):
                continue
            if tmpl.graph_reduce is not None:
                if tmpl.graph_reduce.output_blocks:
                    blocks = tmpl.graph_reduce.output_blocks
                else:
                    group_count = (
                        tmpl.graph_reduce.num_inputs
                        + tmpl.graph_reduce.fan_in
                        - 1
                    ) // tmpl.graph_reduce.fan_in
                    blocks = range(
                        tmpl.graph_reduce.output_base,
                        tmpl.graph_reduce.output_base + group_count,
                    )
            elif tmpl.domain.blocks is not None:
                blocks = tmpl.domain.blocks
            else:
                blocks = range(
                    len(
                        self.runmeta.steps[tmpl.domain.step]
                        .levels[tmpl.domain.level]
                        .boxes
                    )
                )
            coordinates.extend(
                (tmpl.domain.step, tmpl.domain.level, int(block))
                for block in blocks
            )
        return list(dict.fromkeys(coordinates))

    def _materialize_mesh_field(self, field: int) -> list[np.ndarray]:
        self.run_for(mesh_fields=(int(field),))
        return [
            self.runtime.get_task_chunk_array(
                step=step,
                level=level,
                field=int(field),
                block=block,
                dataset=self.dataset,
            ).copy()
            for step, level, block in self._mesh_field_coordinates(field)
        ]

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
                produced.update(ref.field.field for ref in tmpl.outputs)
                consumed.update(ref.field for ref in tmpl.inputs)
        sinks = sorted(produced - consumed)
        if not sinks:
            raise RuntimeError("failed to infer output field from fragment")
        return sinks

    def _append_fragment(self, fragment: list[Stage]) -> None:
        if not fragment:
            return

        fragment_ids = {id(stage) for stage in fragment}
        for stage in fragment:
            for tmpl in stage.templates:
                for input_ref in tmpl.inputs:
                    for parent in self._field_producers.get(input_ref.field, ()):
                        if id(parent) not in fragment_ids and parent not in stage.after:
                            stage.after.append(parent)

        self._stages.extend(fragment)
        for stage in fragment:
            for tmpl in stage.templates:
                for output_ref in tmpl.outputs:
                    self._field_producers.setdefault(output_ref.field.field, []).append(stage)

    def field_expr(
        self,
        expression: str,
        variables: dict[str, int | FieldHandle],
        *,
        out: str | None = None,
        dtype: DType | str = DType.F64,
        domain: Domain | None = None,
    ) -> FieldHandle:
        expr = str(expression).strip()
        if not expr:
            raise ValueError("expression must be non-empty")
        if not variables:
            raise ValueError("variables must be non-empty")

        ordered_vars = list(variables.items())
        var_names = [str(name) for name, _ in ordered_vars]
        if any(not name for name in var_names):
            raise ValueError("variable names must be non-empty strings")
        input_fields = [self._as_field_id(value) for _, value in ordered_vars]
        out_name = out or self._unique_name("field_expr")
        out_fid = self._alloc_field_id(out_name)
        output_dtype = DType(dtype)
        if output_dtype not in (DType.F32, DType.F64):
            raise ValueError("field expressions require dtype='f32' or dtype='f64'")

        ds = self.dataset
        stage = Stage(name=self._unique_name("field_expr"))
        if domain is not None:
            stage.map_blocks(
                name="field_expr_bounded",
                kernel="field_expr",
                domain=domain,
                inputs=[FieldRef(fid) for fid in input_fields],
                outputs=[_like_output(out_fid, output_dtype)],
                deps=DependencyRule(),
                params=FieldExprParams(expr, tuple(var_names)),
            )
            self._append_fragment([stage])
            return FieldHandle(self, out_fid, out_name)
        for level_idx, level_meta in enumerate(self.runmeta.steps[ds.step].levels):
            for block_idx, box in enumerate(level_meta.boxes):
                stage.map_blocks(
                    name=f"field_expr_l{level_idx}_b{block_idx}",
                    kernel="field_expr",
                    domain=Domain(step=ds.step, level=level_idx, blocks=[block_idx]),
                    inputs=[FieldRef(fid) for fid in input_fields],
                    outputs=[_block_output(out_fid, output_dtype)],
                    deps=DependencyRule(),
                    params=FieldExprParams(expr, tuple(var_names)),
                )
        self._append_fragment([stage])
        return FieldHandle(self, out_fid, out_name)

    def mesh_compare(
        self,
        expression: str,
        variables: dict[str, int | FieldHandle],
        *,
        out: str | None = None,
        domain: Domain | None = None,
    ) -> FieldHandle:
        """Lower a real-valued comparison to a normalized U8 mesh mask."""

        expr = str(expression).strip()
        if not expr:
            raise ValueError("expression must be non-empty")
        if not variables:
            raise ValueError("variables must be non-empty")
        ordered_vars = list(variables.items())
        var_names = [str(name) for name, _ in ordered_vars]
        if any(not name for name in var_names):
            raise ValueError("variable names must be non-empty strings")
        input_fields = [self._as_field_id(value) for _, value in ordered_vars]
        out_name = out or self._unique_name("mesh_compare")
        out_fid = self._alloc_field_id(out_name)
        ds = self.dataset
        stage = Stage(name=self._unique_name("mesh_compare"))
        if domain is not None:
            stage.map_blocks(
                name="mesh_compare_bounded",
                kernel="mesh_compare",
                domain=domain,
                inputs=[FieldRef(fid) for fid in input_fields],
                outputs=[_like_output(out_fid, DType.U8)],
                deps=DependencyRule(),
                params=FieldExprParams(expr, tuple(var_names)),
            )
        else:
            for level_idx, level_meta in enumerate(self.runmeta.steps[ds.step].levels):
                for block_idx, _box in enumerate(level_meta.boxes):
                    stage.map_blocks(
                        name=f"mesh_compare_l{level_idx}_b{block_idx}",
                        kernel="mesh_compare",
                        domain=Domain(
                            step=ds.step, level=level_idx, blocks=[block_idx]
                        ),
                        inputs=[FieldRef(fid) for fid in input_fields],
                        outputs=[_block_output(out_fid, DType.U8)],
                        deps=DependencyRule(),
                        params=FieldExprParams(expr, tuple(var_names)),
                    )
        self._append_fragment([stage])
        return FieldHandle(self, out_fid, out_name)

    def mesh_mask_and(
        self,
        left: int | FieldHandle,
        right: int | FieldHandle,
        *,
        out: str | None = None,
        domain: Domain | None = None,
    ) -> FieldHandle:
        """Lower logical conjunction over two normalized U8 mesh masks."""

        input_fields = [self._as_field_id(left), self._as_field_id(right)]
        out_name = out or self._unique_name("mesh_mask_and")
        out_fid = self._alloc_field_id(out_name)
        ds = self.dataset
        stage = Stage(name=self._unique_name("mesh_mask_and"))
        if domain is not None:
            stage.map_blocks(
                name="mesh_mask_and_bounded",
                kernel="mesh_mask_and",
                domain=domain,
                inputs=[FieldRef(fid) for fid in input_fields],
                outputs=[_like_output(out_fid, DType.U8)],
                deps=DependencyRule(),
                params=NoKernelParams(),
            )
        else:
            for level_idx, level_meta in enumerate(self.runmeta.steps[ds.step].levels):
                for block_idx, _box in enumerate(level_meta.boxes):
                    stage.map_blocks(
                        name=f"mesh_mask_and_l{level_idx}_b{block_idx}",
                        kernel="mesh_mask_and",
                        domain=Domain(
                            step=ds.step, level=level_idx, blocks=[block_idx]
                        ),
                        inputs=[FieldRef(fid) for fid in input_fields],
                        outputs=[_block_output(out_fid, DType.U8)],
                        deps=DependencyRule(),
                        params=NoKernelParams(),
                    )
        self._append_fragment([stage])
        return FieldHandle(self, out_fid, out_name)

    def register_derived_field(
        self,
        name: str,
        builder: Callable[["Pipeline"], int | FieldHandle],
        *,
        overwrite: bool = False,
    ) -> None:
        if (not overwrite) and name in self._derived_builders:
            raise ValueError(f"derived field '{name}' is already registered")
        self._derived_builders[name] = builder
        self._derived_cache.pop(name, None)

    def derived_field(self, name: str, *, recache: bool = False) -> FieldHandle:
        if name not in self._derived_builders:
            raise KeyError(f"unknown derived field '{name}'")
        if (not recache) and name in self._derived_cache:
            return self._derived_cache[name]
        built = self._derived_builders[name](self)
        if isinstance(built, FieldHandle):
            if built.pipeline is not self:
                raise ValueError("derived field handle belongs to a different pipeline")
            out = built
        else:
            out = FieldHandle(self, int(built), name)
        self._derived_cache[name] = out
        return out

    def field(self, name_or_id: str | int) -> FieldHandle:
        if isinstance(name_or_id, str):
            if name_or_id in self._derived_builders:
                return self.derived_field(name_or_id)
            fid = int(self.dataset.field_id(name_or_id))
            return FieldHandle(self, fid, name_or_id)
        return FieldHandle(self, int(name_or_id), None)

    def field_add(
        self,
        left: int | FieldHandle,
        right: int | FieldHandle,
        *,
        out: str | None = None,
        dtype: DType | str = DType.F64,
    ) -> FieldHandle:
        return self.field_expr(
            "a + b",
            {"a": left, "b": right},
            out=out or self._unique_name("field_add"),
            dtype=dtype,
        )

    def field_subtract(
        self,
        left: int | FieldHandle,
        right: int | FieldHandle,
        *,
        out: str | None = None,
        dtype: DType | str = DType.F64,
    ) -> FieldHandle:
        return self.field_expr(
            "a - b",
            {"a": left, "b": right},
            out=out or self._unique_name("field_subtract"),
            dtype=dtype,
        )

    def field_multiply(
        self,
        left: int | FieldHandle,
        right: int | FieldHandle,
        *,
        out: str | None = None,
        dtype: DType | str = DType.F64,
    ) -> FieldHandle:
        return self.field_expr(
            "a * b",
            {"a": left, "b": right},
            out=out or self._unique_name("field_multiply"),
            dtype=dtype,
        )

    def field_divide(
        self,
        left: int | FieldHandle,
        right: int | FieldHandle,
        *,
        out: str | None = None,
        dtype: DType | str = DType.F64,
    ) -> FieldHandle:
        return self.field_expr(
            "a / b",
            {"a": left, "b": right},
            out=out or self._unique_name("field_divide"),
            dtype=dtype,
        )

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
        dtype: DType | str = DType.F64,
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
            dtype=dtype,
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
            reduce_fan_in=reduce_fan_in,
            amr_cell_average=amr_cell_average,
        ).lower(self._ctx)
        self._append_fragment(fragment)
        out_field = self._sink_fields(fragment)[-1]
        return FieldHandle(self, out_field, out_name)

    def particle_cic_projection(
        self,
        *,
        particle_type: str,
        axis: str | int,
        axis_bounds: tuple[float, float],
        rect: tuple[float, float, float, float],
        resolution: tuple[int, int],
        mass_max: float | None = None,
        out: str | None = None,
        reduce_fan_in: int | None = None,
    ) -> FieldHandle:
        out_name = out or self._unique_name("particle_cic_projection")
        fragment = ParticleCICProjection(
            particle_type=particle_type,
            axis=axis,
            axis_bounds=axis_bounds,
            rect=rect,
            resolution=resolution,
            mass_max=mass_max,
            out_name=out_name,
            reduce_fan_in=reduce_fan_in,
        ).lower(self._ctx)
        self._append_fragment(fragment)
        out_field = self._sink_fields(fragment)[-1]
        return FieldHandle(self, out_field, out_name)

    def flux_surface_integral(
        self,
        density: int | FieldHandle,
        *,
        momentum: tuple[int | FieldHandle, int | FieldHandle, int | FieldHandle],
        energy: int | FieldHandle,
        passive_scalar: int | FieldHandle,
        magnetic_field: tuple[int | FieldHandle, int | FieldHandle, int | FieldHandle],
        radius: float | Sequence[float],
        temperature: int | FieldHandle | None = None,
        temperature_bins: Sequence[float] | None = None,
        out: str | None = None,
        gamma: float = 5.0 / 3.0,
        reduce_fan_in: int | None = None,
    ) -> FluxSurfaceIntegralHandle:
        if len(momentum) != 3:
            raise ValueError("momentum must contain three fields")
        if len(magnetic_field) != 3:
            raise ValueError("magnetic_field must contain three cell-centered fields")
        out_name = out or self._unique_name("flux_surface_integral")
        op = FluxSurfaceIntegral(
            density=self._as_field_id(density),
            momentum=tuple(self._as_field_id(comp) for comp in momentum),
            energy=self._as_field_id(energy),
            passive_scalar=self._as_field_id(passive_scalar),
            magnetic_field=tuple(self._as_field_id(comp) for comp in magnetic_field),
            radius=radius,
            temperature=(
                None if temperature is None else self._as_field_id(temperature)
            ),
            temperature_bins=temperature_bins,
            out_name=out_name,
            gamma=gamma,
            reduce_fan_in=reduce_fan_in,
        )
        fragment = op.lower(self._ctx)
        self._append_fragment(fragment)
        out_field = self._sink_fields(fragment)[-1]
        return FluxSurfaceIntegralHandle(
            FieldHandle(self, out_field, out_name),
            radii=op.radii,
            temperature_bins=op.temperature_bins,
        )

    def cylindrical_flux_surface_integral(
        self,
        density: int | FieldHandle,
        *,
        momentum: tuple[int | FieldHandle, int | FieldHandle, int | FieldHandle],
        energy: int | FieldHandle,
        passive_scalar: int | FieldHandle,
        magnetic_field: tuple[int | FieldHandle, int | FieldHandle, int | FieldHandle],
        radius: float,
        height: float | Sequence[float],
        temperature: int | FieldHandle | None = None,
        temperature_bins: Sequence[float] | None = None,
        out: str | None = None,
        gamma: float = 5.0 / 3.0,
        reduce_fan_in: int | None = None,
    ) -> CylindricalFluxSurfaceIntegralHandle:
        if len(momentum) != 3:
            raise ValueError("momentum must contain three fields")
        if len(magnetic_field) != 3:
            raise ValueError("magnetic_field must contain three cell-centered fields")
        out_name = out or self._unique_name("cylindrical_flux_surface_integral")
        op = CylindricalFluxSurfaceIntegral(
            density=self._as_field_id(density),
            momentum=tuple(self._as_field_id(comp) for comp in momentum),
            energy=self._as_field_id(energy),
            passive_scalar=self._as_field_id(passive_scalar),
            magnetic_field=tuple(self._as_field_id(comp) for comp in magnetic_field),
            radius=radius,
            height=height,
            temperature=(
                None if temperature is None else self._as_field_id(temperature)
            ),
            temperature_bins=temperature_bins,
            out_name=out_name,
            gamma=gamma,
            reduce_fan_in=reduce_fan_in,
        )
        fragment = op.lower(self._ctx)
        self._append_fragment(fragment)
        out_field = self._sink_fields(fragment)[-1]
        return CylindricalFluxSurfaceIntegralHandle(
            FieldHandle(self, out_field, out_name),
            radius=op.radius,
            heights=op.heights,
            temperature_bins=op.temperature_bins,
        )

    def toomre_q_profile(
        self,
        density: int | FieldHandle,
        *,
        momentum: tuple[int | FieldHandle, int | FieldHandle],
        internal_energy: int | FieldHandle,
        magnetic_field: tuple[
            int | FieldHandle,
            int | FieldHandle,
            int | FieldHandle,
        ],
        potential: int | FieldHandle,
        z_bounds: tuple[float, float],
        radial_range: tuple[float, float] | None = None,
        bins: int | None = None,
        radial_edges: Sequence[float] | None = None,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        out: str | None = None,
        gamma: float = 5.0 / 3.0,
        reduce_fan_in: int | None = None,
    ) -> ToomreQProfileHandle:
        """Accumulate AMR-aware annular moments for gas Toomre-Q profiles."""

        if len(momentum) != 2:
            raise ValueError("momentum must contain x and y fields")
        if len(magnetic_field) != 3:
            raise ValueError("magnetic_field must contain three cell-centered fields")
        if radial_edges is None:
            if radial_range is None or bins is None:
                raise ValueError(
                    "provide radial_edges or both radial_range and bins"
                )
            if len(radial_range) != 2:
                raise ValueError("radial_range must contain two values")
            if int(bins) <= 0:
                raise ValueError("bins must be positive")
            edges = np.linspace(
                float(radial_range[0]), float(radial_range[1]), int(bins) + 1
            )
        else:
            if radial_range is not None or bins is not None:
                raise ValueError(
                    "radial_edges cannot be combined with radial_range or bins"
                )
            edges = np.asarray(radial_edges, dtype=np.float64)
            if edges.ndim != 1:
                raise ValueError("radial_edges must be one-dimensional")
        out_name = out or self._unique_name("toomre_q_profile")
        op = ToomreQProfile(
            density=self._as_field_id(density),
            momentum=tuple(self._as_field_id(comp) for comp in momentum),
            internal_energy=self._as_field_id(internal_energy),
            magnetic_field=tuple(self._as_field_id(comp) for comp in magnetic_field),
            potential=self._as_field_id(potential),
            radial_edges=tuple(float(value) for value in edges),
            z_bounds=z_bounds,
            center=center,
            out_name=out_name,
            gamma=gamma,
            reduce_fan_in=reduce_fan_in,
        )
        fragment = op.lower(self._ctx)
        self._append_fragment(fragment)
        out_field = self._sink_fields(fragment)[-1]
        return ToomreQProfileHandle(
            moments=FieldHandle(self, out_field, out_name),
            radial_edges=op.radial_edges,
            z_bounds=op.z_bounds,
            center=op.center,
            gamma=op.gamma,
        )

    def histogram1d(
        self,
        field: int | FieldHandle,
        *,
        hist_range: tuple[float, float],
        bins: int,
        out: str | None = None,
        weights: int | FieldHandle | None = None,
        domain: Domain | None = None,
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
            domain=domain,
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
        domain: Domain | None = None,
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
            domain=domain,
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
            outputs=[_dynamic_output(out_fid, DType.F64, DynamicUpperBound.backend_chunk())],
            deps=DependencyRule(),
            params=ParticleFieldParams(particle_type, field),
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
            outputs=[_like_output(out_fid, DType.U8)],
            deps=DependencyRule(),
            params=ScalarParams(float(scalar)),
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
            outputs=[_like_output(out_fid, DType.U8)],
            deps=DependencyRule(),
            params=ValuesParams(tuple(float(x) for x in np.asarray(scalars).ravel())),
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
            outputs=[_like_output(out_fid, DType.U8)],
            deps=DependencyRule(),
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
            outputs=[_like_output(out_fid, DType.U8)],
            deps=DependencyRule(),
            params=ScalarParams(float(scalar)),
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
            outputs=[_like_output(out_fid, DType.U8)],
            deps=DependencyRule(),
            params=ScalarParams(float(scalar)),
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
            outputs=[_like_output(out_fid, DType.U8)],
            deps=DependencyRule(),
            params=ScalarParams(float(scalar)),
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        return ParticleMaskHandle(self, out_fid, in_h.chunk_count)

    def particle_compare_scalar(
        self,
        values: ParticleArrayHandle | np.ndarray,
        scalar: float,
        *,
        comparison: str,
    ) -> ParticleMaskHandle:
        """Build a lazy scalar comparison for the supported comparison names."""

        kernels = {
            "lt": "particle_lt_mask",
            "le": "particle_le_mask",
            "gt": "particle_gt_mask",
            "ge": "particle_ge_mask",
            "eq": "particle_eq_mask",
            "ne": "particle_ne_mask",
        }
        try:
            kernel = kernels[comparison]
        except KeyError as exc:
            raise ValueError(f"unsupported particle comparison '{comparison}'") from exc
        in_h = self._coerce_particle_array_handle(values)
        out_fid = self._alloc_runtime_field(f"particle_{comparison}")
        stage = Stage(name=self._unique_name(f"particle_{comparison}"))
        stage.map_blocks(
            name=kernel,
            kernel=kernel,
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[_like_output(out_fid, DType.U8)],
            deps=DependencyRule(),
            params=ScalarParams(float(scalar)),
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
                outputs=[
                    _dynamic_output(fid, DType.U8, DynamicUpperBound.like_input(0))
                ],
                deps=DependencyRule(),
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
            outputs=[_dynamic_output(fid, DType.F64, DynamicUpperBound.like_input(0))],
            deps=DependencyRule(),
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
            outputs=[
                _dynamic_output(fid, DType.F64, DynamicUpperBound.like_input(0))
            ],
            deps=DependencyRule(),
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
            outputs=[
                _dynamic_output(fid, DType.F64, DynamicUpperBound.like_input(0))
            ],
            deps=DependencyRule(),
        )
        self._append_particle_stage(stage, chunk_count=chunk_count)
        return ParticleArrayHandle(self, fid, chunk_count, "float64")

    def particle_binary(
        self,
        left: ParticleArrayHandle | np.ndarray,
        right: ParticleArrayHandle | np.ndarray,
        *,
        operation: str,
    ) -> ParticleArrayHandle:
        """Build a lazy elementwise binary particle operation."""

        kernels = {
            "add": "particle_add",
            "subtract": "particle_subtract",
            "multiply": "particle_multiply",
            "divide": "particle_divide",
            "power": "particle_power",
        }
        try:
            kernel = kernels[operation]
        except KeyError as exc:
            raise ValueError(f"unsupported particle operation '{operation}'") from exc
        left_h = self._coerce_particle_array_handle(left)
        right_h = self._coerce_particle_array_handle(
            right, chunk_count=left_h.chunk_count
        )
        if left_h.chunk_count != right_h.chunk_count:
            raise ValueError("particle operands must have matching chunk_count")
        fid = self._alloc_runtime_field(f"particle_{operation}")
        stage = Stage(name=self._unique_name(f"particle_{operation}"))
        stage.map_blocks(
            name=kernel,
            kernel=kernel,
            domain=Domain(
                step=0, level=0, blocks=list(range(left_h.chunk_count))
            ),
            inputs=[FieldRef(left_h.field), FieldRef(right_h.field)],
            outputs=[
                _dynamic_output(fid, DType.F64, DynamicUpperBound.like_input(0))
            ],
            deps=DependencyRule(),
        )
        self._append_particle_stage(stage, chunk_count=left_h.chunk_count)
        return ParticleArrayHandle(self, fid, left_h.chunk_count, "float64")

    def particle_scalar(
        self,
        values: ParticleArrayHandle | np.ndarray,
        scalar: float,
        *,
        operation: str,
    ) -> ParticleArrayHandle:
        """Build a lazy scalar particle arithmetic operation."""

        kernels = {
            "add": "particle_add_scalar",
            "subtract": "particle_subtract_scalar",
            "rsubtract": "particle_rsubtract_scalar",
            "multiply": "particle_multiply_scalar",
            "divide": "particle_divide_scalar",
            "rdivide": "particle_rdivide_scalar",
            "power": "particle_power_scalar",
            "rpower": "particle_rpower_scalar",
        }
        try:
            kernel = kernels[operation]
        except KeyError as exc:
            raise ValueError(f"unsupported particle scalar operation '{operation}'") from exc
        in_h = self._coerce_particle_array_handle(values)
        fid = self._alloc_runtime_field(f"particle_{operation}")
        stage = Stage(name=self._unique_name(f"particle_{operation}"))
        stage.map_blocks(
            name=kernel,
            kernel=kernel,
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[
                _dynamic_output(fid, DType.F64, DynamicUpperBound.like_input(0))
            ],
            deps=DependencyRule(),
            params=ScalarParams(float(scalar)),
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        return ParticleArrayHandle(self, fid, in_h.chunk_count, "float64")

    def particle_sum_lazy(self, values: ParticleArrayHandle | np.ndarray) -> ParticleScalarHandle:
        """Build a lazy sum reduction without starting execution."""

        in_h = self._coerce_particle_array_handle(values)
        fid = self._alloc_runtime_field("particle_sum")
        stage = Stage(name=self._unique_name("particle_sum"))
        stage.map_blocks(
            name="particle_sum",
            kernel="particle_sum",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[_fixed_output(fid, DType.F64, 1)],
            deps=DependencyRule(),
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=in_h.chunk_count,
            kernel="uniform_slice_reduce",
            output_buffer=BufferSpec(DType.F64, FixedShape((1,)), InitPolicy.ZERO),
        )
        return ParticleScalarHandle(self, reduced, "float64")

    def particle_sum(self, values: ParticleArrayHandle | np.ndarray) -> float:
        """Compatibility eager sum; high-level callers should use lazy values."""

        return float(self.particle_sum_lazy(values).compute())

    def particle_len_lazy(self, values: ParticleArrayHandle | np.ndarray) -> ParticleScalarHandle:
        """Build a lazy particle length reduction without starting execution."""

        in_h = self._coerce_particle_array_handle(values)
        fid = self._alloc_runtime_field("particle_len")
        stage = Stage(name=self._unique_name("particle_len"))
        stage.map_blocks(
            name="particle_len_f64",
            kernel="particle_len_f64",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[_fixed_output(fid, DType.I64, 1)],
            deps=DependencyRule(),
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=in_h.chunk_count,
            kernel="particle_int64_sum_reduce",
            output_buffer=BufferSpec(DType.I64, FixedShape((1,)), InitPolicy.ZERO),
        )
        return ParticleScalarHandle(self, reduced, "int64")

    def particle_len(self, values: ParticleArrayHandle | np.ndarray) -> int:
        """Compatibility eager length; high-level callers should use lazy values."""

        return int(self.particle_len_lazy(values).compute())

    def particle_min_lazy(
        self, values: ParticleArrayHandle | np.ndarray, *, finite_only: bool = True
    ) -> ParticleScalarHandle:
        """Build a lazy minimum reduction without starting execution."""

        in_h = self._coerce_particle_array_handle(values)
        fid = self._alloc_runtime_field("particle_min")
        stage = Stage(name=self._unique_name("particle_min"))
        stage.map_blocks(
            name="particle_min",
            kernel="particle_min",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[_fixed_output(fid, DType.F64, 1)],
            deps=DependencyRule(),
            params=FiniteOnlyParams(bool(finite_only)),
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=in_h.chunk_count,
            kernel="particle_scalar_min_reduce",
            output_buffer=BufferSpec(DType.F64, FixedShape((1,)), InitPolicy.ZERO),
        )
        return ParticleScalarHandle(self, reduced, "float64")

    def particle_min(self, values: ParticleArrayHandle | np.ndarray, *, finite_only: bool = True) -> float:
        """Compatibility eager minimum; high-level callers should use lazy values."""

        return float(self.particle_min_lazy(values, finite_only=finite_only).compute())

    def particle_max_lazy(
        self, values: ParticleArrayHandle | np.ndarray, *, finite_only: bool = True
    ) -> ParticleScalarHandle:
        """Build a lazy maximum reduction without starting execution."""

        in_h = self._coerce_particle_array_handle(values)
        fid = self._alloc_runtime_field("particle_max")
        stage = Stage(name=self._unique_name("particle_max"))
        stage.map_blocks(
            name="particle_max",
            kernel="particle_max",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[_fixed_output(fid, DType.F64, 1)],
            deps=DependencyRule(),
            params=FiniteOnlyParams(bool(finite_only)),
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=in_h.chunk_count,
            kernel="particle_scalar_max_reduce",
            output_buffer=BufferSpec(DType.F64, FixedShape((1,)), InitPolicy.ZERO),
        )
        return ParticleScalarHandle(self, reduced, "float64")

    def particle_max(self, values: ParticleArrayHandle | np.ndarray, *, finite_only: bool = True) -> float:
        """Compatibility eager maximum; high-level callers should use lazy values."""

        return float(self.particle_max_lazy(values, finite_only=finite_only).compute())

    def particle_count_lazy(self, mask: ParticleMaskHandle | np.ndarray) -> ParticleScalarHandle:
        """Build a lazy selected-particle count without starting execution."""

        in_h = self._coerce_particle_mask_handle(mask)
        fid = self._alloc_runtime_field("particle_count")
        stage = Stage(name=self._unique_name("particle_count"))
        stage.map_blocks(
            name="particle_count",
            kernel="particle_count",
            domain=Domain(step=0, level=0, blocks=list(range(in_h.chunk_count))),
            inputs=[FieldRef(in_h.field)],
            outputs=[_fixed_output(fid, DType.I64, 1)],
            deps=DependencyRule(),
        )
        self._append_particle_stage(stage, chunk_count=in_h.chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=in_h.chunk_count,
            kernel="particle_int64_sum_reduce",
            output_buffer=BufferSpec(DType.I64, FixedShape((1,)), InitPolicy.ZERO),
        )
        return ParticleScalarHandle(self, reduced, "int64")

    def particle_count(self, mask: ParticleMaskHandle | np.ndarray) -> int:
        """Compatibility eager count; high-level callers should use lazy values."""

        return int(self.particle_count_lazy(mask).compute())

    def particle_topk_modes_lazy(
        self,
        particle_type: str,
        field: str,
        *,
        k: int,
    ) -> ParticleTopKHandle:
        """Build a lazy top-k mode reduction without starting execution."""

        if k <= 0:
            raise ValueError("k must be positive")
        chunk_count = self._particle_chunk_count(particle_type)
        counts_fid = self._alloc_runtime_field("particle_topk_counts")
        stage = Stage(name=self._unique_name("particle_topk_map"))
        stage.map_blocks(
            name="particle_topk_modes_map",
            kernel="particle_topk_modes_map",
            domain=Domain(step=0, level=0),
            inputs=[],
            outputs=[_dynamic_output(
                counts_fid, DType.OPAQUE, DynamicUpperBound.backend_chunk()
            )],
            deps=DependencyRule(),
            params=ParticleFieldParams(particle_type, field),
        )
        self._append_particle_stage(stage, chunk_count=chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=counts_fid,
            chunk_count=chunk_count,
            kernel="particle_value_counts_reduce",
            output_buffer=BufferSpec(
                DType.OPAQUE,
                DynamicShape(DynamicUpperBound.backend_chunk()),
            ),
        )
        fid = self._alloc_runtime_field("particle_topk")
        finalize = Stage(name=self._unique_name("particle_topk_finalize"))
        finalize.map_blocks(
            name="particle_topk_modes_finalize",
            kernel="particle_topk_modes_finalize",
            domain=Domain(step=0, level=0, blocks=[0]),
            inputs=[FieldRef(reduced)],
            outputs=[_fixed_output(fid, DType.F64, int(k) * 2)],
            deps=DependencyRule(),
            params=TopKModesParams(int(k)),
        )
        self._append_particle_stage(finalize, chunk_count=1)
        return ParticleTopKHandle(self, fid, int(k))

    def particle_topk_modes(
        self,
        particle_type: str,
        field: str,
        *,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility eager top-k; high-level callers should use lazy values."""

        if k <= 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        return self.particle_topk_modes_lazy(particle_type, field, k=k).compute()

    def particle_histogram1d_lazy(
        self,
        values: ParticleArrayHandle | np.ndarray,
        *,
        bins: int | np.ndarray,
        hist_range: tuple[float, float] | None = None,
        weights: ParticleArrayHandle | np.ndarray | None = None,
        density: bool = False,
    ) -> ParticleHistogram1DHandle:
        """Build a lazy particle histogram without starting execution."""

        if (
            weights is not None
            and not isinstance(values, ParticleArrayHandle)
            and not isinstance(weights, ParticleArrayHandle)
            and np.asarray(values).shape != np.asarray(weights).shape
        ):
            raise ValueError("particle_histogram1d values and weights must have matching shapes")
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
        kernel = "particle_histogram1d_weighted" if weights is not None else "particle_histogram1d"
        fid = self._alloc_runtime_field("particle_hist1d")
        stage = Stage(name=self._unique_name("particle_hist1d"))
        stage.map_blocks(
            name=kernel,
            kernel=kernel,
            domain=Domain(step=0, level=0, blocks=list(range(chunk_count))),
            inputs=inputs,
            outputs=[_fixed_output(fid, DType.F64, edges.size - 1)],
            deps=DependencyRule(),
            params=ParticleHistogramParams(tuple(float(x) for x in edges)),
        )
        self._append_particle_stage(stage, chunk_count=chunk_count)
        reduced = self._append_particle_reduce_tree(
            input_field=fid,
            chunk_count=chunk_count,
            kernel="uniform_slice_reduce",
            output_buffer=BufferSpec(
                DType.F64, FixedShape((int(edges.size - 1),)), InitPolicy.ZERO
            ),
        )
        return ParticleHistogram1DHandle(
            ParticleArrayHandle(self, reduced, 1, "float64"),
            edges.copy(),
            density=bool(density),
        )

    def particle_histogram1d(
        self,
        values: ParticleArrayHandle | np.ndarray,
        *,
        bins: int | np.ndarray,
        hist_range: tuple[float, float] | None = None,
        weights: ParticleArrayHandle | np.ndarray | None = None,
        density: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility eager histogram; high-level callers should use lazy values."""

        return self.particle_histogram1d_lazy(
            values,
            bins=bins,
            hist_range=hist_range,
            weights=weights,
            density=density,
        ).compute()

    def plan(self) -> Plan:
        return Plan(stages=list(self._stages))

    @staticmethod
    def _producer_plan(
        fields: Sequence[int], producers: dict[int, list[Stage]]
    ) -> Plan:
        roots: list[Stage] = []
        seen: set[int] = set()
        for field in fields:
            for stage in producers.get(int(field), ()):
                if id(stage) not in seen:
                    roots.append(stage)
                    seen.add(id(stage))
        return Plan(stages=roots)

    def run_for(
        self,
        *,
        mesh_fields: Sequence[int] = (),
        particle_fields: Sequence[int] = (),
        progress_bar: bool = False,
    ) -> None:
        """Execute only the producer ancestry needed by the requested fields."""

        mesh_plan = self._producer_plan(mesh_fields, self._field_producers)
        if mesh_plan.stages:
            self.runtime.run(
                mesh_plan,
                runmeta=self.runmeta,
                dataset=self.dataset,
                progress_bar=progress_bar,
            )
        particle_plan = self._producer_plan(
            particle_fields, self._particle_producers
        )
        if particle_plan.stages:
            self.runtime.run(
                particle_plan,
                runmeta=self._particle_runmeta(),
                dataset=self.dataset,
                progress_bar=progress_bar,
            )
            self._particle_executed = True

    def mark_persisted(
        self, *, mesh_fields: Sequence[int] = (), particle_fields: Sequence[int] = ()
    ) -> None:
        """Treat materialized fields as graph sources for subsequently built work."""

        for field in mesh_fields:
            self._field_producers.pop(int(field), None)
        for field in particle_fields:
            self._particle_producers.pop(int(field), None)

    def run(self, *, progress_bar: bool = False) -> None:
        if self._stages:
            self.runtime.run(
                self.plan(),
                runmeta=self.runmeta,
                dataset=self.dataset,
                progress_bar=progress_bar,
            )
        self._ensure_particle_executed(progress_bar=progress_bar)


def pipeline(*, runtime: Any, runmeta: Any, dataset: Any) -> Pipeline:
    return Pipeline(runtime=runtime, runmeta=runmeta, dataset=dataset)
