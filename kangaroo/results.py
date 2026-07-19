"""Materialized result types returned by the high-level Kangaroo API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from analysis.runmeta import BlockBox, LevelGeom


@dataclass(frozen=True)
class AMRChunk:
    """One AMR block together with the geometry needed to locate it."""

    values: np.ndarray
    step: int
    level: int
    block: int
    box: BlockBox
    geometry: LevelGeom


@dataclass(frozen=True)
class AMRChunkedArray:
    """A hierarchy of geometrically located AMR blocks.

    AMR blocks are not a linear sequence: blocks may be disjoint and finer
    levels may overlap coarser levels.  Consequently this type deliberately
    refuses dense concatenation.
    """

    chunks: tuple[AMRChunk, ...]

    def __iter__(self) -> Iterator[AMRChunk]:
        return iter(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def nbytes(self) -> int:
        """Return the total bytes represented by all AMR blocks."""

        return sum(chunk.values.nbytes for chunk in self.chunks)

    def gather(self, *, max_bytes: int | None = None) -> np.ndarray:
        """Reject concatenation because AMR blocks have no dense ordering."""

        raise ValueError(
            "gather is not defined for an AMR hierarchy; use iter_chunks(), "
            "slice(), or project() instead"
        )


@dataclass(frozen=True)
class ChunkedArray:
    """A linearly ordered chunked array, such as a particle field."""

    chunks: tuple[np.ndarray, ...]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def nbytes(self) -> int:
        """Return the total bytes represented by all local chunks."""

        return sum(chunk.nbytes for chunk in self.chunks)

    def gather(self, *, max_bytes: int | None = None) -> np.ndarray:
        """Concatenate one-dimensional chunks after enforcing an optional limit."""

        if max_bytes is not None and self.nbytes > max_bytes:
            raise MemoryError(
                f"gather would materialize {self.nbytes} bytes, exceeding max_bytes={max_bytes}"
            )
        if not self.chunks:
            return np.empty(0)
        return np.concatenate(self.chunks)


@dataclass(frozen=True)
class HistogramResult:
    """Materialized one-dimensional histogram counts and bin edges."""

    counts: np.ndarray
    edges: np.ndarray


@dataclass(frozen=True)
class Histogram2DResult:
    """Materialized two-dimensional histogram counts and axis edges."""

    counts: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray


@dataclass(frozen=True)
class FluxSurfaceIntegralResult:
    """Materialized spherical flux components and their scientific coordinates."""

    values: np.ndarray
    radii: tuple[float, ...]
    components: tuple[str, ...]
    temperature_bins: tuple[float, ...] | None = None


@dataclass(frozen=True)
class CylindricalFluxSurfaceIntegralResult:
    """Materialized cylindrical flux components and geometric coordinates."""

    values: np.ndarray
    radius: float
    heights: tuple[float, ...]
    geometric_sections: tuple[str, ...]
    components: tuple[str, ...]
    temperature_bins: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ToomreQProfileResult:
    """Materialized annular moments used to derive a gas Toomre-Q profile."""

    moments: np.ndarray
    radial_edges: np.ndarray
    components: tuple[str, ...]
    z_bounds: tuple[float, float]
    center: tuple[float, float, float]
    gamma: float


@dataclass(frozen=True)
class TopKResult:
    """Materialized particle modes and their counts."""

    values: np.ndarray
    counts: np.ndarray
