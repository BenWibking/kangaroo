"""Kangaroo orchestration for compact potential samples (not model mathematics)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from potential_replay.fit import solve_samples

from .pipeline import Pipeline


def extract_samples(
    runtime: Any,
    dataset: Any,
    runmeta: Any,
    field: int,
    points: ArrayLike,
    *,
    half_width: float,
    batch_size: int = 128,
    condition_limit: float = 1e10,
    progress: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read/reduce on distributed workers; return only values and fit diagnostics.

    Batch output size is bounded independently of the mesh and number of modes.
    The axis is deduplicated, as are any other coincident quadrature points.
    """
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("points must be a finite (N,3) array")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    values, chunks = [], {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        pipe = Pipeline(runtime=runtime, runmeta=runmeta, dataset=dataset)
        handle = pipe.potential_samples(field, points=batch, half_width=half_width)
        pipe.run(progress_bar=progress)
        stats = runtime.get_task_chunk_array(
            step=dataset.step,
            level=dataset.level,
            field=handle.field,
            version=0,
            block=0,
            dataset=dataset,
        )
        value, diagnostics = solve_samples(stats, condition_limit=condition_limit)
        values.append(value)
        for key, data in diagnostics.items():
            chunks.setdefault(key, []).append(data)
    return np.concatenate(values)[inverse], {
        key: np.concatenate(data)[inverse] for key, data in chunks.items()
    }
