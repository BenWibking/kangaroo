"""Browsable high-level dataset facade and typed geometry metadata."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from analysis.pipeline import Pipeline

from .array import Array, ParticleArray

T = TypeVar("T")


@dataclass(frozen=True)
class PlaneGeometry:
    """Immutable geometry needed to lower a physical plane operation."""

    coord: float
    rect: tuple[float, float, float, float]
    resolution: tuple[int, int]
    labels: tuple[str, str]
    plane: str
    axis_index: int
    axis_bounds: tuple[float, float]


class _NamedCollection(Mapping[str, T], Generic[T]):
    def __init__(self, names: list[str], factory: Any, kind: str) -> None:
        self._names = tuple(dict.fromkeys(names))
        self._factory = factory
        self._kind = kind

    def __getitem__(self, name: str) -> T:
        if name not in self._names:
            suggestions = get_close_matches(name, self._names, n=3)
            suffix = f"; close matches: {', '.join(suggestions)}" if suggestions else ""
            available = ", ".join(self._names[:8])
            raise KeyError(f"unknown {self._kind} '{name}'; available: {available}{suffix}")
        return self._factory(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


class Geometry:
    """Typed geometry operations for a selected dataset view."""

    def __init__(self, dataset: "Dataset") -> None:
        self._dataset = dataset

    def plane(
        self,
        *,
        axis: str | int,
        coord: float | None = None,
        zoom: float = 1.0,
        resolution: str | tuple[int, int] | None = None,
    ) -> PlaneGeometry:
        """Return immutable plane geometry with dataset-derived defaults."""

        raw = self._dataset._backend.plane_geometry(
            axis=axis,
            level=self._dataset.level,
            coord=coord,
            zoom=zoom,
            resolution=resolution,
        )
        return PlaneGeometry(
            coord=float(raw["coord"]),
            rect=tuple(raw["rect"]),
            resolution=tuple(raw["resolution"]),
            labels=tuple(raw["labels"]),
            plane=str(raw["plane"]),
            axis_index=int(raw["axis_index"]),
            axis_bounds=tuple(raw["axis_bounds"]),
        )


class ParticleSpecies:
    """Mapping-like particle species whose fields are lazy arrays."""

    def __init__(self, dataset: "Dataset", name: str) -> None:
        self.dataset = dataset
        self.name = name
        self.fields = _NamedCollection(
            dataset._backend.list_particle_fields(name), self._field, "particle field"
        )

    def _field(self, name: str) -> ParticleArray:
        handle = self.dataset._pipeline.particle_field(self.name, name)
        return ParticleArray._from_handle(
            self.dataset,
            handle,
            name=f"{self.name}/{name}",
            dtype=handle.dtype,
            species=self.name,
            backend_field=(self.name, name),
        )

    def __getitem__(self, name: str) -> ParticleArray:
        """Return a lazy particle field by name."""

        return self.fields[name]

    def project(
        self,
        *,
        axis: str | int,
        bounds: tuple[float, float] | None = None,
        resolution: tuple[int, int] | str | None = None,
        zoom: float = 1.0,
        mass_max: float | None = None,
    ) -> Array:
        """Lazily cloud-in-cell project this particle species onto a plane."""

        from . import config

        geometry = self.dataset.geometry.plane(
            axis=axis, resolution=resolution, zoom=zoom
        )
        handle = self.dataset._pipeline.particle_cic_projection(
            particle_type=self.name,
            axis=axis,
            axis_bounds=geometry.axis_bounds if bounds is None else bounds,
            rect=geometry.rect,
            resolution=geometry.resolution,
            mass_max=mass_max,
            reduce_fan_in=config.get("reduction.fan_in"),
        )
        return Array._from_handle(
            self.dataset,
            handle,
            name=handle.name,
            dtype="float64",
            shape=geometry.resolution[::-1],
        )

    def __repr__(self) -> str:
        return f"kangaroo.ParticleSpecies<name={self.name!r}, fields={len(self.fields)}>"


class Dataset:
    """A selected dataset view with named lazy fields and execution context."""

    def __init__(self, backend: Any, client: Any) -> None:
        self._backend = backend
        self.client = client
        self._pipeline = Pipeline(
            runtime=client.runtime,
            runmeta=backend.get_runmeta(),
            dataset=backend,
        )
        names = list(backend.metadata.get("var_names", []))
        for info in backend.metadata.get("variable_info", {}).values():
            names.extend(info.get("component_names", []))
        self.fields = _NamedCollection(names, self._field, "field")
        self.particles = _NamedCollection(
            backend.list_particle_types(), self._particle_species, "particle species"
        )
        self.meshes = _NamedCollection(
            backend.list_meshes(), lambda name: name, "mesh"
        )
        self.geometry = Geometry(self)

    @property
    def step(self) -> int:
        """Active dataset step."""

        return int(self._backend.step)

    @property
    def level(self) -> int:
        """Active AMR level used by geometry defaults."""

        return int(self._backend.level)

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Read-only common and backend-specific metadata."""

        return MappingProxyType(self._backend.metadata)

    def _field(self, name: str) -> Array:
        return Array._from_handle(self, self._pipeline.field(name), name=name)

    def _particle_species(self, name: str) -> ParticleSpecies:
        return ParticleSpecies(self, name)

    def __getitem__(self, name: str) -> Array:
        """Return a named lazy mesh field."""

        return self.fields[name]

    def select(
        self,
        *,
        step: int | None = None,
        level: int | None = None,
        mesh: str | None = None,
    ) -> "Dataset":
        """Return an independent selected view without mutating this dataset."""

        selected = self.client.open_dataset(
            self._backend.uri,
            runmeta=self._backend.runmeta,
            step=self.step if step is None else step,
            level=self.level if level is None else level,
        )
        if mesh is not None:
            selected._backend.select_mesh(mesh)
            selected = Dataset(selected._backend, self.client)
        return selected

    def __repr__(self) -> str:
        return (
            f"kangaroo.Dataset<uri={self._backend.uri!r}, kind={self._backend.kind!r}, "
            f"step={self.step}, level={self.level}, fields={len(self.fields)}>"
        )
