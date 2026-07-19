"""Kangaroo's high-level lazy scientific analysis API."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import config

from .array import (
    Array,
    CylindricalFluxSurfaceIntegral,
    FluxSurfaceIntegral,
    Histogram,
    Histogram2D,
    LazyValue,
    ParticleArray,
    ParticleMask,
    Scalar,
    ToomreQProfile,
    TopK,
    compute,
)
from .client import Client
from .dataset import Dataset, Geometry, ParticleSpecies, PlaneGeometry
from .results import (
    AMRChunk,
    AMRChunkedArray,
    ChunkedArray,
    CylindricalFluxSurfaceIntegralResult,
    FluxSurfaceIntegralResult,
    Histogram2DResult,
    HistogramResult,
    ToomreQProfileResult,
    TopKResult,
)

__all__ = [
    "AMRChunk",
    "AMRChunkedArray",
    "Array",
    "ChunkedArray",
    "Client",
    "CylindricalFluxSurfaceIntegralResult",
    "Dataset",
    "CylindricalFluxSurfaceIntegral",
    "FluxSurfaceIntegral",
    "FluxSurfaceIntegralResult",
    "Geometry",
    "Histogram",
    "Histogram2D",
    "Histogram2DResult",
    "HistogramResult",
    "LazyValue",
    "Mask",
    "ParticleArray",
    "ParticleMask",
    "ParticleSpecies",
    "PlaneGeometry",
    "Scalar",
    "ToomreQProfile",
    "ToomreQProfileResult",
    "TopKResult",
    "TopK",
    "compute",
    "config",
    "get_default_client",
    "open_dataset",
    "set_default_client",
    "bool_",
    "float32",
    "float64",
    "int64",
]

_default_client: Client | None = None
bool_ = np.dtype("bool")
float32 = np.dtype("float32")
float64 = np.dtype("float64")
int64 = np.dtype("int64")
Mask = ParticleMask


def get_default_client() -> Client:
    """Return the process-local default client, creating it lazily."""

    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


def set_default_client(client: Client | None) -> None:
    """Replace the process-local default client, primarily for applications and tests."""

    global _default_client
    if client is not None and not isinstance(client, Client):
        raise TypeError("default client must be a Client or None")
    _default_client = client


def open_dataset(
    uri: str,
    *,
    step: int = 0,
    level: int | None = None,
    runmeta: Any | None = None,
    client: Client | None = None,
) -> Dataset:
    """Open a dataset through an explicit client or the lazy default client."""

    return (client or get_default_client()).open_dataset(
        uri, runmeta=runmeta, step=step, level=level
    )
