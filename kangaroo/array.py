"""Lazy scientific values and the public compute boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from analysis.buffer import DType
from analysis.buffer import BlockShape, DynamicShape, FixedShape
from analysis.pipeline import (
    FieldHandle,
    CylindricalFluxSurfaceIntegralHandle,
    FluxSurfaceIntegralHandle,
    Histogram1DHandle,
    Histogram2DHandle,
    ParticleArrayHandle,
    ParticleHistogram1DHandle,
    ParticleMaskHandle,
    ParticleScalarHandle,
    ParticleTopKHandle,
    ToomreQProfileHandle,
)
from analysis.plan import Domain

from .results import (
    ChunkedArray,
    CylindricalFluxSurfaceIntegralResult,
    FluxSurfaceIntegralResult,
    Histogram2DResult,
    HistogramResult,
    ToomreQProfileResult,
    TopKResult,
)
from . import config


class LazyValue:
    """Common protocol for values that record work without executing it."""

    def __init__(self, dataset: Any, *, name: str | None, dtype: str) -> None:
        self.dataset = dataset
        self.name = name
        self.dtype = dtype
        self._is_materialized = False

    @property
    def is_materialized(self) -> bool:
        """Whether this value has crossed an execution boundary in this process."""

        return self._is_materialized

    @property
    def domain(self) -> str:
        """Concise logical domain description available without execution."""

        return self._domain_description()

    def _domain_description(self) -> str:
        return "unknown"

    def compute(self, **kwargs: Any) -> Any:
        """Execute the minimal compatible request and return a local result."""

        return compute(self, **kwargs)

    def persist(self, **kwargs: Any) -> "LazyValue":
        """Execute this value and retain its result in distributed runtime storage."""

        progress = bool(kwargs.pop("progress", False))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected persist options: {unknown}")
        mesh_fields, particle_fields = self._requested_fields()
        for field in (*mesh_fields, *particle_fields):
            self.dataset.client.runtime.mark_field_persistent(
                int(field), self.name or f"field_{field}"
            )
        _execute((self,), progress=progress)
        self.dataset._pipeline.mark_persisted(
            mesh_fields=mesh_fields, particle_fields=particle_fields
        )
        self._is_materialized = True
        return self

    def explain(self) -> str:
        """Return a textual graph summary without executing scientific kernels."""

        pipeline = self.dataset._pipeline
        mesh_fields, particle_fields = self._requested_fields()
        stages = pipeline._producer_plan(
            mesh_fields, pipeline._field_producers
        ).topo_stages()
        stage_ids = {id(stage) for stage in stages}
        stages.extend(
            stage
            for stage in pipeline._producer_plan(
                particle_fields, pipeline._particle_producers
            ).topo_stages()
            if id(stage) not in stage_ids
        )
        kernels = [tmpl.kernel for stage in stages for tmpl in stage.templates]
        particle_stage_ids = {id(stage) for stage in pipeline._particle_stages}
        task_count = 0
        known_bytes = 0
        has_unknown_storage = False
        reductions = 0
        dtype_bytes = {DType.U8: 1, DType.I64: 8, DType.F32: 4, DType.F64: 8}
        particle_runmeta = pipeline._particle_runmeta()
        for stage in stages:
            runmeta = particle_runmeta if id(stage) in particle_stage_ids else pipeline.runmeta
            for tmpl in stage.templates:
                if tmpl.graph_reduce is not None:
                    blocks = len(tmpl.graph_reduce.output_blocks) or max(
                        1,
                        (tmpl.graph_reduce.num_inputs + tmpl.graph_reduce.fan_in - 1)
                        // tmpl.graph_reduce.fan_in,
                    )
                    reductions += 1
                elif tmpl.domain.blocks is not None:
                    blocks = len(tmpl.domain.blocks)
                else:
                    blocks = len(
                        runmeta.steps[tmpl.domain.step].levels[tmpl.domain.level].boxes
                    )
                task_count += blocks
                for output in tmpl.outputs:
                    output_instances = blocks
                    itemsize = dtype_bytes.get(output.buffer.dtype)
                    shape = output.buffer.shape
                    elements: int | None = None
                    if isinstance(shape, FixedShape):
                        elements = int(np.prod(shape.extents))
                    elif isinstance(shape, BlockShape):
                        level = runmeta.steps[tmpl.domain.step].levels[tmpl.domain.level]
                        selected = (
                            tmpl.domain.blocks
                            if tmpl.domain.blocks is not None
                            else range(len(level.boxes))
                        )
                        elements = sum(
                            (level.boxes[block].hi[0] - level.boxes[block].lo[0] + 1)
                            * (level.boxes[block].hi[1] - level.boxes[block].lo[1] + 1)
                            * (level.boxes[block].hi[2] - level.boxes[block].lo[2] + 1)
                            * shape.components
                            for block in selected
                        )
                        output_instances = 1
                    elif (
                        isinstance(shape, DynamicShape)
                        and shape.upper_bound.value is not None
                    ):
                        elements = shape.upper_bound.value
                    if elements is None or itemsize is None:
                        has_unknown_storage = True
                    else:
                        known_bytes += elements * itemsize * output_instances
        storage = f">={known_bytes} bytes" if has_unknown_storage else f"{known_bytes} bytes"
        localities = self.dataset.client.runtime.num_localities()
        return (
            f"{type(self).__name__}(name={self.name!r}, dtype={self.dtype}, "
            f"domain={self.domain}, stages={len(stages)}, estimated_tasks={task_count}, "
            f"estimated_storage={storage}, reductions={reductions}, "
            f"localities={localities}, kernels={kernels})"
        )

    def visualize(self, filename: str | Path | None = None) -> str:
        """Return Graphviz DOT for this context and optionally write it to a file."""

        pipeline = self.dataset._pipeline
        mesh_fields, particle_fields = self._requested_fields()
        stages = pipeline._producer_plan(
            mesh_fields, pipeline._field_producers
        ).topo_stages()
        stage_ids = {id(stage) for stage in stages}
        stages.extend(
            stage
            for stage in pipeline._producer_plan(
                particle_fields, pipeline._particle_producers
            ).topo_stages()
            if id(stage) not in stage_ids
        )
        ids = {id(stage): index for index, stage in enumerate(stages)}
        lines = ["digraph kangaroo {"]
        for stage in stages:
            sid = ids[id(stage)]
            lines.append(f'  n{sid} [label="{stage.name}"];')
            for parent in stage.after:
                if id(parent) in ids:
                    lines.append(f"  n{ids[id(parent)]} -> n{sid};")
        lines.append("}")
        dot = "\n".join(lines)
        if filename is not None:
            Path(filename).write_text(dot + "\n", encoding="utf-8")
        return dot

    def _materialize(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (), ()


class Array(LazyValue):
    """Lazy AMR field or bounded regular-array expression."""

    def __init__(
        self,
        dataset: Any,
        handle: FieldHandle,
        *,
        name: str | None,
        dtype: str = "float64",
        shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(dataset, name=name, dtype=dtype)
        self._field_handle = handle
        self._shape = shape

    @classmethod
    def _from_handle(
        cls,
        dataset: Any,
        handle: FieldHandle,
        *,
        name: str | None,
        dtype: str = "float64",
        shape: tuple[int, ...] | None = None,
    ) -> "Array":
        return cls(dataset, handle, name=name, dtype=dtype, shape=shape)

    @property
    def chunks(self) -> tuple[tuple[int, int, int], ...]:
        """Per-level ``(level, blocks, cells)`` metadata for AMR values."""

        out: list[tuple[int, int, int]] = []
        runmeta = self.dataset._pipeline.runmeta
        for level, level_meta in enumerate(runmeta.steps[self.dataset.step].levels):
            cells = sum(
                (box.hi[0] - box.lo[0] + 1)
                * (box.hi[1] - box.lo[1] + 1)
                * (box.hi[2] - box.lo[2] + 1)
                for box in level_meta.boxes
            )
            out.append((level, len(level_meta.boxes), cells))
        return tuple(out)

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Bounded regular shape, or ``None`` for a general AMR hierarchy."""

        return self._shape

    def _domain_description(self) -> str:
        if self._shape is not None:
            return f"Regular(shape={self._shape})"
        return f"AMR({len(self.chunks)} levels)"

    def _require_array(self, other: "Array") -> None:
        if not isinstance(other, Array):
            raise TypeError(f"expected Array, got {type(other).__name__}")
        if other.dataset is not self.dataset:
            raise ValueError(
                "cannot combine arrays from different dataset contexts "
                f"({self.dataset!r} and {other.dataset!r})"
            )
        if other._shape != self._shape:
            raise ValueError(
                "cannot combine mesh arrays with different domains or shapes "
                f"({self._shape!r} and {other._shape!r})"
            )

    def _expression_domain(self) -> Domain | None:
        if self._shape is None:
            return None
        return Domain(step=self.dataset.step, level=self.dataset.level, blocks=[0])

    @staticmethod
    def _field_expr_dtype(dtype: str) -> DType:
        return DType.F32 if np.dtype(dtype) == np.dtype("float32") else DType.F64

    def _binary_dtype(self, other: Any) -> str:
        if isinstance(other, Array):
            return np.result_type(np.dtype(self.dtype), np.dtype(other.dtype)).name
        return np.dtype(self.dtype).name

    @staticmethod
    def _literal(value: int | float) -> str:
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"expected a numeric scalar, got {type(value).__name__}")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("non-finite scalar literals are not supported in field expressions")
        return repr(value)

    def _binary(self, other: Any, symbol: str, name: str, *, reverse: bool = False) -> "Array":
        pipe = self.dataset._pipeline
        dtype = self._binary_dtype(other)
        if isinstance(other, Array):
            self._require_array(other)
            left, right = (other, self) if reverse else (self, other)
            handle = pipe.field_expr(
                f"a {symbol} b",
                {"a": left._field_handle, "b": right._field_handle},
                out=name,
                dtype=self._field_expr_dtype(dtype),
                domain=self._expression_domain(),
            )
        else:
            literal = self._literal(other)
            expression = f"{literal} {symbol} a" if reverse else f"a {symbol} {literal}"
            handle = pipe.field_expr(
                expression,
                {"a": self._field_handle},
                out=name,
                dtype=self._field_expr_dtype(dtype),
                domain=self._expression_domain(),
            )
        return Array._from_handle(
            self.dataset, handle, name=name, dtype=dtype, shape=self._shape
        )

    def __add__(self, other: Any) -> "Array":
        return self._binary(other, "+", "add")

    def __radd__(self, other: Any) -> "Array":
        return self._binary(other, "+", "add", reverse=True)

    def __sub__(self, other: Any) -> "Array":
        return self._binary(other, "-", "subtract")

    def __rsub__(self, other: Any) -> "Array":
        return self._binary(other, "-", "subtract", reverse=True)

    def __mul__(self, other: Any) -> "Array":
        return self._binary(other, "*", "multiply")

    def __rmul__(self, other: Any) -> "Array":
        return self._binary(other, "*", "multiply", reverse=True)

    def __truediv__(self, other: Any) -> "Array":
        return self._binary(other, "/", "divide")

    def __rtruediv__(self, other: Any) -> "Array":
        return self._binary(other, "/", "divide", reverse=True)

    def __pow__(self, other: Any) -> "Array":
        dtype = self._binary_dtype(other)
        if isinstance(other, Array):
            self._require_array(other)
            handle = self.dataset._pipeline.field_expr(
                "pow(a, b)",
                {"a": self._field_handle, "b": other._field_handle},
                out="power",
                dtype=self._field_expr_dtype(dtype),
                domain=self._expression_domain(),
            )
        else:
            handle = self.dataset._pipeline.field_expr(
                f"pow(a, {self._literal(other)})",
                {"a": self._field_handle},
                out="power",
                dtype=self._field_expr_dtype(dtype),
                domain=self._expression_domain(),
            )
        return Array._from_handle(
            self.dataset,
            handle,
            name="power",
            dtype=dtype,
            shape=self._shape,
        )

    def __neg__(self) -> "Array":
        return self * -1.0

    def _compare(self, other: Any, symbol: str) -> "Array":
        return self._binary(other, symbol, "mask")

    def __lt__(self, other: Any) -> "Array":
        return self._compare(other, "<")

    def __le__(self, other: Any) -> "Array":
        return self._compare(other, "<=")

    def __gt__(self, other: Any) -> "Array":
        return self._compare(other, ">")

    def __ge__(self, other: Any) -> "Array":
        return self._compare(other, ">=")

    def __eq__(self, other: object) -> "Array":  # type: ignore[override]
        if not isinstance(other, (Array, int, float, np.integer, np.floating)):
            return NotImplemented
        return self._compare(other, "==")

    def __ne__(self, other: object) -> "Array":  # type: ignore[override]
        if not isinstance(other, (Array, int, float, np.integer, np.floating)):
            return NotImplemented
        return self._compare(other, "!=")

    def __and__(self, other: Any) -> "Array":
        return self._binary(other, "*", "mask_and")

    def rename(self, name: str) -> "Array":
        """Return an equivalent lazy value with a new display name."""

        if not name:
            raise ValueError("name must be non-empty")
        return Array._from_handle(
            self.dataset,
            self._field_handle,
            name=name,
            dtype=self.dtype,
            shape=self._shape,
        )

    def astype(self, dtype: Any) -> "Array":
        """Lazily convert this array to float32 or float64."""

        target = np.dtype(dtype)
        if target not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("mesh astype currently supports float32 and float64")
        tag = DType.F32 if target == np.dtype("float32") else DType.F64
        handle = self.dataset._pipeline.field_expr(
            "a",
            {"a": self._field_handle},
            out=f"astype_{target.name}",
            dtype=tag,
            domain=self._expression_domain(),
        )
        return Array._from_handle(
            self.dataset, handle, name=self.name, dtype=target.name, shape=self._shape
        )

    def slice(
        self,
        *,
        axis: str | int,
        coord: float | None = None,
        resolution: tuple[int, int] | str | None = None,
        zoom: float = 1.0,
        rect: tuple[float, float, float, float] | None = None,
    ) -> "Array":
        """Lazily sample this AMR value onto a bounded uniform plane."""

        geometry = self.dataset.geometry.plane(
            axis=axis, coord=coord, resolution=resolution, zoom=zoom
        )
        output_rect = geometry.rect if rect is None else rect
        handle = self.dataset._pipeline.uniform_slice(
            self._field_handle,
            axis=axis,
            coord=geometry.coord,
            rect=output_rect,
            resolution=geometry.resolution,
            out=f"{self.name or 'field'}_slice",
            dtype=self._field_expr_dtype(self.dtype),
            reduce_fan_in=config.get("reduction.fan_in"),
        )
        return Array._from_handle(
            self.dataset,
            handle,
            name=handle.name,
            dtype=self.dtype,
            shape=geometry.resolution[::-1],
        )

    def project(
        self,
        *,
        axis: str | int,
        bounds: tuple[float, float] | None = None,
        resolution: tuple[int, int] | str | None = None,
        zoom: float = 1.0,
        rect: tuple[float, float, float, float] | None = None,
        amr_cell_average: bool = True,
    ) -> "Array":
        """Lazily project this AMR value through a physical axis interval."""

        geometry = self.dataset.geometry.plane(
            axis=axis, resolution=resolution, zoom=zoom
        )
        handle = self.dataset._pipeline.uniform_projection(
            self._field_handle,
            axis=axis,
            axis_bounds=geometry.axis_bounds if bounds is None else bounds,
            rect=geometry.rect if rect is None else rect,
            resolution=geometry.resolution,
            out=f"{self.name or 'field'}_projection",
            amr_cell_average=amr_cell_average,
            reduce_fan_in=config.get("reduction.fan_in"),
        )
        return Array._from_handle(
            self.dataset,
            handle,
            name=handle.name,
            dtype="float64",
            shape=geometry.resolution[::-1],
        )

    def histogram(
        self,
        *,
        bins: int,
        range: tuple[float, float],
        weights: "Array | None" = None,
    ) -> "Histogram":
        """Build a lazy one-dimensional histogram reduction."""

        if weights is not None:
            self._require_array(weights)
        handle = self.dataset._pipeline.histogram1d(
            self._field_handle,
            hist_range=range,
            bins=bins,
            weights=None if weights is None else weights._field_handle,
            domain=self._expression_domain(),
            reduce_fan_in=config.get("reduction.fan_in"),
        )
        return Histogram(self.dataset, handle)

    def histogram2d(
        self,
        other: "Array",
        *,
        bins: tuple[int, int],
        range: tuple[tuple[float, float], tuple[float, float]],
        weights: "Array | None" = None,
    ) -> "Histogram2D":
        """Build a lazy two-dimensional histogram reduction."""

        self._require_array(other)
        if weights is not None:
            self._require_array(weights)
        handle = self.dataset._pipeline.histogram2d(
            self._field_handle,
            other._field_handle,
            x_range=range[0],
            y_range=range[1],
            bins=bins,
            weights=None if weights is None else weights._field_handle,
            domain=self._expression_domain(),
            reduce_fan_in=config.get("reduction.fan_in"),
        )
        return Histogram2D(self.dataset, handle)

    def vorticity(self, y: "Array", z: "Array") -> "Array":
        """Build the lazy vorticity magnitude of three velocity components."""

        self._require_array(y)
        self._require_array(z)
        handle = self.dataset._pipeline.vorticity_mag(
            (self._field_handle, y._field_handle, z._field_handle)
        )
        return Array._from_handle(self.dataset, handle, name=handle.name)

    def flux_surface_integral(
        self,
        *,
        momentum: tuple["Array", "Array", "Array"],
        energy: "Array",
        passive_scalar: "Array",
        magnetic_field: tuple["Array", "Array", "Array"],
        radius: float | Sequence[float],
        temperature: "Array | None" = None,
        temperature_bins: Sequence[float] | None = None,
        gamma: float = 5.0 / 3.0,
    ) -> "FluxSurfaceIntegral":
        """Build a lazy spherical flux-surface integral using this density."""

        operands = [*momentum, energy, passive_scalar, *magnetic_field]
        if temperature is not None:
            operands.append(temperature)
        for operand in operands:
            self._require_array(operand)
        handle = self.dataset._pipeline.flux_surface_integral(
            self._field_handle,
            momentum=tuple(item._field_handle for item in momentum),
            energy=energy._field_handle,
            passive_scalar=passive_scalar._field_handle,
            magnetic_field=tuple(item._field_handle for item in magnetic_field),
            radius=radius,
            temperature=None if temperature is None else temperature._field_handle,
            temperature_bins=temperature_bins,
            gamma=gamma,
            reduce_fan_in=config.get("reduction.fan_in"),
        )
        return FluxSurfaceIntegral(self.dataset, handle)

    def cylindrical_flux_surface_integral(
        self,
        *,
        momentum: tuple["Array", "Array", "Array"],
        energy: "Array",
        passive_scalar: "Array",
        magnetic_field: tuple["Array", "Array", "Array"],
        radius: float,
        height: float | Sequence[float],
        temperature: "Array | None" = None,
        temperature_bins: Sequence[float] | None = None,
        gamma: float = 5.0 / 3.0,
    ) -> "CylindricalFluxSurfaceIntegral":
        """Build a lazy cylindrical flux-surface integral using this density."""

        operands = [*momentum, energy, passive_scalar, *magnetic_field]
        if temperature is not None:
            operands.append(temperature)
        for operand in operands:
            self._require_array(operand)
        handle = self.dataset._pipeline.cylindrical_flux_surface_integral(
            self._field_handle,
            momentum=tuple(item._field_handle for item in momentum),
            energy=energy._field_handle,
            passive_scalar=passive_scalar._field_handle,
            magnetic_field=tuple(item._field_handle for item in magnetic_field),
            radius=radius,
            height=height,
            temperature=None if temperature is None else temperature._field_handle,
            temperature_bins=temperature_bins,
            gamma=gamma,
            reduce_fan_in=config.get("reduction.fan_in"),
        )
        return CylindricalFluxSurfaceIntegral(self.dataset, handle)

    def toomre_q_profile(
        self,
        *,
        momentum: tuple["Array", "Array"],
        internal_energy: "Array",
        magnetic_field: tuple["Array", "Array", "Array"],
        potential: "Array",
        z_bounds: tuple[float, float],
        radial_range: tuple[float, float] | None = None,
        bins: int | None = None,
        radial_edges: Sequence[float] | None = None,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gamma: float = 5.0 / 3.0,
    ) -> "ToomreQProfile":
        """Build a lazy AMR-aware gas Toomre-Q annular profile."""

        operands = [*momentum, internal_energy, *magnetic_field, potential]
        for operand in operands:
            self._require_array(operand)
        handle = self.dataset._pipeline.toomre_q_profile(
            self._field_handle,
            momentum=tuple(item._field_handle for item in momentum),
            internal_energy=internal_energy._field_handle,
            magnetic_field=tuple(item._field_handle for item in magnetic_field),
            potential=potential._field_handle,
            z_bounds=z_bounds,
            radial_range=radial_range,
            bins=bins,
            radial_edges=radial_edges,
            center=center,
            gamma=gamma,
            reduce_fan_in=config.get("reduction.fan_in"),
        )
        return ToomreQProfile(self.dataset, handle)

    def iter_chunks(self) -> Iterable[np.ndarray]:
        """Execute and iterate local chunks without dense AMR gathering."""

        result = self.compute()
        if isinstance(result, ChunkedArray):
            return iter(result.chunks)
        return iter((result,))

    def _materialize(self, **kwargs: Any) -> np.ndarray | ChunkedArray:
        gather = bool(kwargs.pop("gather", False))
        max_bytes = kwargs.pop("max_bytes", None)
        if kwargs:
            raise TypeError(f"unexpected compute options: {', '.join(sorted(kwargs))}")
        runtime = self.dataset.client.runtime
        if self._shape is not None:
            return runtime.get_task_chunk_array(
                step=self.dataset.step,
                level=self.dataset.level,
                field=self._field_handle.field,
                block=0,
                shape=self._shape,
                dataset=self.dataset._backend,
            ).copy()
        chunks: list[np.ndarray] = []
        runmeta = self.dataset._pipeline.runmeta
        for level, level_meta in enumerate(runmeta.steps[self.dataset.step].levels):
            for block in range(len(level_meta.boxes)):
                chunks.append(
                    runtime.get_task_chunk_array(
                        step=self.dataset.step,
                        level=level,
                        field=self._field_handle.field,
                        block=block,
                        dataset=self.dataset._backend,
                    ).copy()
                )
        result = ChunkedArray(tuple(chunks))
        return result.gather(max_bytes=max_bytes) if gather else result

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self._field_handle.field,), ()

    def __repr__(self) -> str:
        return (
            f"kangaroo.Array<name={self.name!r}, dtype={self.dtype}, "
            f"domain={self.domain}, lazy={not self.is_materialized}>"
        )


class ParticleArray(LazyValue):
    """Lazy chunked particle field or filtered particle expression."""

    def __init__(
        self,
        dataset: Any,
        handle: ParticleArrayHandle,
        *,
        name: str,
        dtype: str,
        species: str | None = None,
        backend_field: tuple[str, str] | None = None,
        position_lineage: object | None = None,
    ) -> None:
        super().__init__(dataset, name=name, dtype=dtype)
        self._particle_handle = handle
        self._particle_species = species
        self._backend_field = backend_field
        self._position_lineage = (
            position_lineage
            if position_lineage is not None
            else (("particle_species", species) if species is not None else object())
        )

    @classmethod
    def _from_handle(
        cls,
        dataset: Any,
        handle: ParticleArrayHandle,
        *,
        name: str,
        dtype: str,
        species: str | None = None,
        backend_field: tuple[str, str] | None = None,
        position_lineage: object | None = None,
    ) -> "ParticleArray":
        return cls(
            dataset,
            handle,
            name=name,
            dtype=dtype,
            species=species,
            backend_field=backend_field,
            position_lineage=position_lineage,
        )

    def _domain_description(self) -> str:
        return f"Particles(chunks={self._particle_handle.chunk_count})"

    @property
    def chunks(self) -> int:
        """Number of distributed particle chunks."""

        return self._particle_handle.chunk_count

    def _require_same(self, other: Any) -> None:
        if not isinstance(other, (ParticleArray, ParticleMask)):
            raise TypeError(f"expected a particle value, got {type(other).__name__}")
        if other.dataset is not self.dataset:
            raise ValueError("cannot combine particle values from different dataset contexts")
        if other._particle_species != self._particle_species:
            raise ValueError(
                "cannot combine values from different particle species "
                f"({self._particle_species!r} and {other._particle_species!r})"
            )
        if other._position_lineage != self._position_lineage:
            raise ValueError(
                "cannot combine particle values from different filtered particle domains"
            )

    def isfinite(self) -> "ParticleMask":
        """Return a lazy mask selecting finite values."""

        handle = self.dataset._pipeline.particle_isfinite(self._particle_handle)
        return ParticleMask(
            self.dataset,
            handle,
            name=f"isfinite({self.name})",
            species=self._particle_species,
            position_lineage=self._position_lineage,
        )

    def __gt__(self, scalar: float) -> "ParticleMask":
        handle = self.dataset._pipeline.particle_compare_scalar(
            self._particle_handle, scalar, comparison="gt"
        )
        return ParticleMask(
            self.dataset,
            handle,
            name=f"{self.name}>scalar",
            species=self._particle_species,
            position_lineage=self._position_lineage,
        )

    def __le__(self, scalar: float) -> "ParticleMask":
        handle = self.dataset._pipeline.particle_compare_scalar(
            self._particle_handle, scalar, comparison="le"
        )
        return ParticleMask(
            self.dataset,
            handle,
            name=f"{self.name}<=scalar",
            species=self._particle_species,
            position_lineage=self._position_lineage,
        )

    def __lt__(self, scalar: float) -> "ParticleMask":
        handle = self.dataset._pipeline.particle_compare_scalar(
            self._particle_handle, scalar, comparison="lt"
        )
        return ParticleMask(
            self.dataset,
            handle,
            name=f"{self.name}<scalar",
            species=self._particle_species,
            position_lineage=self._position_lineage,
        )

    def __ge__(self, scalar: float) -> "ParticleMask":
        handle = self.dataset._pipeline.particle_compare_scalar(
            self._particle_handle, scalar, comparison="ge"
        )
        return ParticleMask(
            self.dataset,
            handle,
            name=f"{self.name}>=scalar",
            species=self._particle_species,
            position_lineage=self._position_lineage,
        )

    def __eq__(self, scalar: object) -> "ParticleMask":  # type: ignore[override]
        if not isinstance(scalar, (int, float, np.integer, np.floating)):
            return NotImplemented
        handle = self.dataset._pipeline.particle_compare_scalar(
            self._particle_handle, float(scalar), comparison="eq"
        )
        return ParticleMask(
            self.dataset,
            handle,
            name=f"{self.name}==scalar",
            species=self._particle_species,
            position_lineage=self._position_lineage,
        )

    def __ne__(self, scalar: object) -> "ParticleMask":  # type: ignore[override]
        if not isinstance(scalar, (int, float, np.integer, np.floating)):
            return NotImplemented
        handle = self.dataset._pipeline.particle_compare_scalar(
            self._particle_handle, float(scalar), comparison="ne"
        )
        return ParticleMask(
            self.dataset,
            handle,
            name=f"{self.name}!=scalar",
            species=self._particle_species,
            position_lineage=self._position_lineage,
        )

    def _binary_particle(
        self, other: Any, operation: str, *, reverse: bool = False
    ) -> "ParticleArray":
        if isinstance(other, ParticleArray):
            self._require_same(other)
            left, right = (other, self) if reverse else (self, other)
            handle = self.dataset._pipeline.particle_binary(
                left._particle_handle, right._particle_handle, operation=operation
            )
        elif isinstance(other, (int, float, np.integer, np.floating)):
            scalar_operation = f"r{operation}" if reverse and operation in {
                "subtract", "divide", "power"
            } else operation
            handle = self.dataset._pipeline.particle_scalar(
                self._particle_handle, float(other), operation=scalar_operation
            )
        else:
            return NotImplemented
        return ParticleArray(
            self.dataset,
            handle,
            name=f"particle_{operation}",
            dtype="float64",
            species=self._particle_species,
            position_lineage=self._position_lineage,
        )

    def __add__(self, other: Any) -> "ParticleArray":
        return self._binary_particle(other, "add")

    def __radd__(self, other: Any) -> "ParticleArray":
        return self._binary_particle(other, "add", reverse=True)

    def __sub__(self, other: Any) -> "ParticleArray":
        return self._binary_particle(other, "subtract")

    def __rsub__(self, other: Any) -> "ParticleArray":
        return self._binary_particle(other, "subtract", reverse=True)

    def __mul__(self, other: Any) -> "ParticleArray":
        return self._binary_particle(other, "multiply")

    def __rmul__(self, other: Any) -> "ParticleArray":
        return self._binary_particle(other, "multiply", reverse=True)

    def __truediv__(self, other: Any) -> "ParticleArray":
        return self._binary_particle(other, "divide")

    def __rtruediv__(self, other: Any) -> "ParticleArray":
        return self._binary_particle(other, "divide", reverse=True)

    def __pow__(self, other: Any) -> "ParticleArray":
        return self._binary_particle(other, "power")

    def __rpow__(self, other: Any) -> "ParticleArray":
        return self._binary_particle(other, "power", reverse=True)

    def __neg__(self) -> "ParticleArray":
        return self * -1.0

    def rename(self, name: str) -> "ParticleArray":
        """Return an equivalent particle expression with a new display name."""

        if not name:
            raise ValueError("name must be non-empty")
        return ParticleArray(
            self.dataset,
            self._particle_handle,
            name=name,
            dtype=self.dtype,
            species=self._particle_species,
            backend_field=self._backend_field,
            position_lineage=self._position_lineage,
        )

    def astype(self, dtype: Any) -> "ParticleArray":
        """Lazily select the local floating dtype used when materializing chunks."""

        target = np.dtype(dtype)
        if target not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("particle astype currently supports float32 and float64")
        return ParticleArray(
            self.dataset,
            self._particle_handle,
            name=self.name or "particle",
            dtype=target.name,
            species=self._particle_species,
            position_lineage=self._position_lineage,
        )

    def __getitem__(self, mask: "ParticleMask") -> "ParticleArray":
        """Lazily retain values selected by a particle mask."""

        self._require_same(mask)
        handle = self.dataset._pipeline.particle_filter(
            self._particle_handle, mask._mask_handle
        )
        return ParticleArray(
            self.dataset,
            handle,
            name=self.name or "filtered",
            dtype=self.dtype,
            species=self._particle_species,
            position_lineage=mask._selection_lineage,
        )

    def sum(self) -> "Scalar":
        """Return a lazy sum reduction."""

        return Scalar(self.dataset, self.dataset._pipeline.particle_sum_lazy(self._particle_handle))

    def min(self, *, finite_only: bool = True) -> "Scalar":
        """Return a lazy minimum reduction."""

        return Scalar(
            self.dataset,
            self.dataset._pipeline.particle_min_lazy(
                self._particle_handle, finite_only=finite_only
            ),
        )

    def max(self, *, finite_only: bool = True) -> "Scalar":
        """Return a lazy maximum reduction."""

        return Scalar(
            self.dataset,
            self.dataset._pipeline.particle_max_lazy(
                self._particle_handle, finite_only=finite_only
            ),
        )

    def mean(self) -> "Scalar":
        """Return a lazy mean whose sum and length share one execution graph."""

        total = self.dataset._pipeline.particle_sum_lazy(self._particle_handle)
        count = self.dataset._pipeline.particle_len_lazy(self._particle_handle)
        return Scalar(
            self.dataset,
            total,
            evaluator=lambda: float(total.compute()) / int(count.compute()),
            dependencies=(total, count),
            name=f"mean({self.name})",
        )

    def histogram(
        self,
        *,
        bins: int | np.ndarray,
        range: tuple[float, float] | None = None,
        weights: "ParticleArray | None" = None,
        density: bool = False,
    ) -> "Histogram":
        """Return a lazy chunk-preserving particle histogram reduction."""

        if weights is not None:
            self._require_same(weights)
        handle = self.dataset._pipeline.particle_histogram1d_lazy(
            self._particle_handle,
            bins=bins,
            hist_range=range,
            weights=None if weights is None else weights._particle_handle,
            density=density,
        )
        return Histogram(self.dataset, handle)

    def topk(self, k: int) -> "TopK":
        """Return lazy top-k modes for a backend particle field."""

        if self._backend_field is None:
            raise ValueError("topk requires a backend particle field")
        particle_type, field = self._backend_field
        handle = self.dataset._pipeline.particle_topk_modes_lazy(
            particle_type, field, k=k
        )
        return TopK(self.dataset, handle)

    def iter_chunks(self) -> Iterable[np.ndarray]:
        """Execute and iterate chunks without concatenating them."""

        return iter(self.compute().chunks)

    def _materialize(self, **kwargs: Any) -> np.ndarray | ChunkedArray:
        gather = bool(kwargs.pop("gather", False))
        max_bytes = kwargs.pop("max_bytes", None)
        if kwargs:
            raise TypeError(f"unexpected compute options: {', '.join(sorted(kwargs))}")
        chunks = tuple(
            chunk.astype(self.dtype, copy=False)
            for chunk in self._particle_handle.iter_chunks()
        )
        result = ChunkedArray(chunks)
        return result.gather(max_bytes=max_bytes) if gather else result

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (), (self._particle_handle.field,)

    def __repr__(self) -> str:
        return (
            f"kangaroo.ParticleArray<name={self.name!r}, dtype={self.dtype}, "
            f"domain={self.domain}, lazy={not self.is_materialized}>"
        )


class ParticleMask(LazyValue):
    """Lazy chunked boolean mask over a particle species."""

    def __init__(
        self,
        dataset: Any,
        handle: ParticleMaskHandle,
        *,
        name: str,
        species: str | None = None,
        position_lineage: object | None = None,
        selection_lineage: object | None = None,
    ) -> None:
        super().__init__(dataset, name=name, dtype="bool")
        self._mask_handle = handle
        self._particle_species = species
        self._position_lineage = (
            position_lineage
            if position_lineage is not None
            else (("particle_species", species) if species is not None else object())
        )
        self._selection_lineage = (
            selection_lineage if selection_lineage is not None else object()
        )

    def _domain_description(self) -> str:
        return f"ParticleMask(chunks={self._mask_handle.chunk_count})"

    def __and__(self, other: "ParticleMask") -> "ParticleMask":
        if not isinstance(other, ParticleMask) or other.dataset is not self.dataset:
            raise ValueError("particle mask operands must share a dataset context")
        if other._particle_species != self._particle_species:
            raise ValueError(
                "cannot combine masks from different particle species "
                f"({self._particle_species!r} and {other._particle_species!r})"
            )
        if other._position_lineage != self._position_lineage:
            raise ValueError(
                "cannot combine masks from different filtered particle domains"
            )
        handle = self.dataset._pipeline.particle_and(
            self._mask_handle, other._mask_handle
        )
        return ParticleMask(
            self.dataset,
            handle,
            name="mask_and",
            species=self._particle_species,
            position_lineage=self._position_lineage,
        )

    def count(self) -> "Scalar":
        """Return a lazy count of selected particles."""

        return Scalar(self.dataset, self.dataset._pipeline.particle_count_lazy(self._mask_handle))

    def _materialize(self, **kwargs: Any) -> np.ndarray | ChunkedArray:
        gather = bool(kwargs.pop("gather", False))
        max_bytes = kwargs.pop("max_bytes", None)
        if kwargs:
            raise TypeError(f"unexpected compute options: {', '.join(sorted(kwargs))}")
        result = ChunkedArray(tuple(self._mask_handle.iter_chunks()))
        return result.gather(max_bytes=max_bytes) if gather else result

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (), (self._mask_handle.field,)


class Scalar(LazyValue):
    """Lazy scalar reduction that refuses implicit Python conversion."""

    def __init__(
        self,
        dataset: Any,
        handle: ParticleScalarHandle,
        *,
        evaluator: Callable[[], float | int] | None = None,
        dependencies: tuple[ParticleScalarHandle, ...] = (),
        name: str | None = None,
    ) -> None:
        super().__init__(dataset, name=name or "scalar", dtype=handle.dtype)
        self._scalar_handle = handle
        self._evaluator = evaluator
        self._dependencies = dependencies or (handle,)

    def _domain_description(self) -> str:
        return "Scalar"

    def _materialize(self, **kwargs: Any) -> float | int:
        if kwargs:
            raise TypeError(f"unexpected compute options: {', '.join(sorted(kwargs))}")
        return self._scalar_handle.compute() if self._evaluator is None else self._evaluator()

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (), tuple(handle.field for handle in self._dependencies)

    def __bool__(self) -> bool:
        raise TypeError("a lazy scalar has no truth value; call compute() first")

    def __float__(self) -> float:
        raise TypeError("a lazy scalar cannot be converted implicitly; call compute() first")

    def __int__(self) -> int:
        raise TypeError("a lazy scalar cannot be converted implicitly; call compute() first")

    def __repr__(self) -> str:
        return f"kangaroo.Scalar<dtype={self.dtype}, lazy={not self.is_materialized}>"


class Histogram(LazyValue):
    """Lazy one-dimensional histogram expression."""

    def __init__(self, dataset: Any, handle: Histogram1DHandle | ParticleHistogram1DHandle) -> None:
        super().__init__(dataset, name="histogram", dtype="float64")
        self._histogram_handle = handle

    @property
    def edges(self) -> np.ndarray:
        """Bin edges known before execution."""

        return np.asarray(self._histogram_handle.edges, dtype=np.float64)

    def _domain_description(self) -> str:
        return f"Histogram(bins={self.edges.size - 1})"

    def _materialize(self, **kwargs: Any) -> HistogramResult:
        if kwargs:
            raise TypeError(f"unexpected compute options: {', '.join(sorted(kwargs))}")
        if isinstance(self._histogram_handle, ParticleHistogram1DHandle):
            counts, edges = self._histogram_handle.compute()
        else:
            counts = self.dataset.client.runtime.get_task_chunk_array(
                step=self.dataset.step,
                level=self.dataset.level,
                field=self._histogram_handle.counts.field,
                block=0,
                shape=(self._histogram_handle.bins,),
                dataset=self.dataset._backend,
            ).copy()
            edges = self.edges
        return HistogramResult(np.asarray(counts), np.asarray(edges))

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if isinstance(self._histogram_handle, ParticleHistogram1DHandle):
            return (), (self._histogram_handle.counts.field,)
        return (self._histogram_handle.counts.field,), ()


class Histogram2D(LazyValue):
    """Lazy two-dimensional histogram expression."""

    def __init__(self, dataset: Any, handle: Histogram2DHandle) -> None:
        super().__init__(dataset, name="histogram2d", dtype="float64")
        self._handle = handle

    def _domain_description(self) -> str:
        return f"Histogram2D(bins={self._handle.bins})"

    def _materialize(self, **kwargs: Any) -> Histogram2DResult:
        if kwargs:
            raise TypeError(f"unexpected compute options: {', '.join(sorted(kwargs))}")
        counts = self.dataset.client.runtime.get_task_chunk_array(
            step=self.dataset.step,
            level=self.dataset.level,
            field=self._handle.counts.field,
            block=0,
            shape=self._handle.bins,
            dataset=self.dataset._backend,
        ).copy()
        x_edges, y_edges = self._handle.edges
        return Histogram2DResult(counts, np.asarray(x_edges), np.asarray(y_edges))

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self._handle.counts.field,), ()


class TopK(LazyValue):
    """Lazy top-k particle mode expression."""

    def __init__(self, dataset: Any, handle: ParticleTopKHandle) -> None:
        super().__init__(dataset, name="topk", dtype="float64")
        self._handle = handle

    def _domain_description(self) -> str:
        return f"TopK(k={self._handle.k})"

    def _materialize(self, **kwargs: Any) -> TopKResult:
        if kwargs:
            raise TypeError(f"unexpected compute options: {', '.join(sorted(kwargs))}")
        values, counts = self._handle.compute()
        return TopKResult(values, counts)

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (), (self._handle.field,)


class FluxSurfaceIntegral(LazyValue):
    """Lazy spherical flux integral expression."""

    def __init__(self, dataset: Any, handle: FluxSurfaceIntegralHandle) -> None:
        super().__init__(dataset, name=handle.name, dtype="float64")
        self._handle = handle

    def _materialize(self, **kwargs: Any) -> FluxSurfaceIntegralResult:
        if kwargs:
            raise TypeError(f"unexpected compute options: {', '.join(sorted(kwargs))}")
        values = self.dataset.client.runtime.get_task_chunk_array(
            step=self.dataset.step,
            level=0,
            field=self._handle.field,
            block=0,
            dataset=self.dataset._backend,
        ).copy()
        return FluxSurfaceIntegralResult(
            values, self._handle.radii, self._handle.components, self._handle.temperature_bins
        )

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self._handle.field,), ()


class CylindricalFluxSurfaceIntegral(LazyValue):
    """Lazy cylindrical flux integral expression."""

    def __init__(self, dataset: Any, handle: CylindricalFluxSurfaceIntegralHandle) -> None:
        super().__init__(dataset, name=handle.name, dtype="float64")
        self._handle = handle

    def _materialize(self, **kwargs: Any) -> CylindricalFluxSurfaceIntegralResult:
        if kwargs:
            raise TypeError(f"unexpected compute options: {', '.join(sorted(kwargs))}")
        values = self.dataset.client.runtime.get_task_chunk_array(
            step=self.dataset.step,
            level=0,
            field=self._handle.field,
            block=0,
            dataset=self.dataset._backend,
        ).copy()
        return CylindricalFluxSurfaceIntegralResult(
            values,
            self._handle.radius,
            self._handle.heights,
            self._handle.geometric_sections,
            self._handle.components,
            self._handle.temperature_bins,
        )

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self._handle.field,), ()


class ToomreQProfile(LazyValue):
    """Lazy gas Toomre-Q annular-moment expression."""

    def __init__(self, dataset: Any, handle: ToomreQProfileHandle) -> None:
        super().__init__(dataset, name=handle.name, dtype="float64")
        self._handle = handle

    def _materialize(self, **kwargs: Any) -> ToomreQProfileResult:
        if kwargs:
            raise TypeError(f"unexpected compute options: {', '.join(sorted(kwargs))}")
        moments = self.dataset.client.runtime.get_task_chunk_array(
            step=self.dataset.step,
            level=0,
            field=self._handle.field,
            block=0,
            dataset=self.dataset._backend,
        ).copy()
        return ToomreQProfileResult(
            moments,
            self._handle.edges,
            self._handle.components,
            self._handle.z_bounds,
            self._handle.center,
            self._handle.gamma,
        )

    def _requested_fields(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self._handle.field,), ()


def _execute(values: Sequence[LazyValue], *, progress: bool) -> None:
    datasets: list[Any] = []
    for value in values:
        if not isinstance(value, LazyValue):
            raise TypeError(f"compute expects lazy values, got {type(value).__name__}")
        if value.dataset not in datasets:
            datasets.append(value.dataset)
    if len(datasets) != 1:
        owners = ", ".join(repr(dataset) for dataset in datasets)
        raise ValueError(f"compute values must share one dataset context; got {owners}")
    mesh_fields: list[int] = []
    particle_fields: list[int] = []
    for value in values:
        mesh, particle = value._requested_fields()
        mesh_fields.extend(mesh)
        particle_fields.extend(particle)
    pipeline = datasets[0]._pipeline
    pipeline.run_for(
        mesh_fields=tuple(dict.fromkeys(mesh_fields)),
        particle_fields=tuple(dict.fromkeys(particle_fields)),
        progress_bar=progress or datasets[0].client.progress,
    )


def compute(*values: LazyValue, progress: bool = False, **kwargs: Any) -> Any:
    """Compute one or more compatible lazy values, sharing one graph execution."""

    if not values:
        return ()
    _execute(values, progress=progress)
    array_types = (Array, ParticleArray, ParticleMask)
    has_array_value = any(isinstance(value, array_types) for value in values)
    results = []
    for value in values:
        options = kwargs
        if has_array_value and not isinstance(value, array_types):
            options = {
                key: option
                for key, option in kwargs.items()
                if key not in {"gather", "max_bytes"}
            }
        result = value._materialize(**options)
        value._is_materialized = True
        results.append(result)
    return results[0] if len(results) == 1 else tuple(results)
