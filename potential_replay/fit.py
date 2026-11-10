"""Pure numerical construction; neither orchestration nor native dependencies."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .model import SCHEMA, VERSION, PotentialModel, validate_frame, validate_knots


def quadrature_points(
    radii: ArrayLike,
    heights: ArrayLike,
    nphi: int,
    *,
    center: ArrayLike = (0, 0, 0),
    rotation: ArrayLike = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
) -> np.ndarray:
    radii = validate_knots(radii, "radii", 5)
    heights = validate_knots(heights, "heights")
    if radii[0] != 0:
        raise ValueError("radii must begin at the axis (0)")
    if not isinstance(nphi, (int, np.integer)) or nphi < 4:
        raise ValueError("nphi must be an integer >= 4")
    center, rotation = validate_frame(center, rotation)
    r, z, phi = np.meshgrid(
        radii, heights, 2 * np.pi * np.arange(nphi) / nphi, indexing="ij"
    )
    local = np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=-1)
    return (local.reshape(-1, 3) @ rotation.T + center).reshape(local.shape)


def solve_samples(
    statistics: ArrayLike,
    *,
    condition_limit: float = 1e10,
    coverage_tolerance: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Solve compact native MLS statistics; fail closed on every invalid sample.

    Returns scalar samples and diagnostic arrays (all in flattened sample order).
    Normal matrices use support-scaled coordinates. Diagonal equilibration and
    an eigenvalue ratio guard detect loss of rank before the solve.
    """
    stats = np.asarray(statistics, dtype=float)
    if stats.ndim != 2 or stats.shape[1] != 114 or not np.isfinite(stats).all():
        raise ValueError("statistics must be a finite (N,114) array")
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and > 1")
    if not 0 < coverage_tolerance < 0.01:
        raise ValueError("coverage_tolerance must be in (0,0.01)")
    bad = (
        (np.abs(stats[:, 110] - 1) > coverage_tolerance)
        | (stats[:, 111] != 0)
        | (stats[:, 112] < 10)
    )
    if np.any(bad):
        raise ValueError(
            f"incomplete/nonfinite/insufficient samples at indices {np.flatnonzero(bad)[:12].tolist()}"
        )
    gram = stats[:, :100].reshape(-1, 10, 10)
    diag = np.diagonal(gram, axis1=1, axis2=2)
    if np.any(diag <= 0):
        raise ValueError("rank-deficient sampling")
    scale = np.sqrt(diag)
    matrix = gram / scale[:, :, None] / scale[:, None, :]
    eigenvalues = np.linalg.eigvalsh(matrix)
    bad = eigenvalues[:, 0] <= eigenvalues[:, -1] / condition_limit
    if np.any(bad):
        raise ValueError(
            f"ill-conditioned sampling at indices {np.flatnonzero(bad)[:12].tolist()}"
        )
    # Center the RHS using the weighted mean to reduce cancellation from gpot's gauge.
    mean = stats[:, 100] / gram[:, 0, 0]
    rhs = stats[:, 100:110] - mean[:, None] * gram[:, :, 0]
    solution = np.linalg.solve(matrix, (rhs / scale)[..., None])[..., 0] / scale
    samples = solution[:, 0] + mean
    diagnostics = {
        "coverage": stats[:, 110],
        "cell_count": stats[:, 112],
        "condition": eigenvalues[:, -1] / eigenvalues[:, 0],
        "cell_width_over_support": stats[:, 113],
    }
    return samples, diagnostics


def _cubic_pieces(knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Not-a-knot cubic coefficients, ascending powers of interval-local t in [0,1]."""
    n = len(knots)
    h = np.diff(knots)
    matrix = np.zeros((n, n))
    rhs = np.zeros_like(values, dtype=float)
    matrix[0, :3] = (-h[1], h[0] + h[1], -h[0])
    matrix[-1, -3:] = (-h[-1], h[-2] + h[-1], -h[-2])
    expand = (slice(None),) + (None,) * (values.ndim - 1)
    slopes = np.diff(values, axis=0) / h[expand]
    for i in range(1, n - 1):
        matrix[i, i - 1 : i + 2] = (h[i - 1], 2 * (h[i - 1] + h[i]), h[i])
        rhs[i] = 6 * (slopes[i] - slopes[i - 1])
    second = np.linalg.solve(matrix, rhs.reshape(n, -1)).reshape(values.shape)
    hh = h[expand] ** 2
    return np.stack(
        (
            values[:-1],
            np.diff(values, axis=0) - hh * (2 * second[:-1] + second[1:]) / 6,
            hh * second[:-1] / 2,
            hh * (second[1:] - second[:-1]) / 6,
        ),
        axis=1,
    )


def fit_samples(
    radii: ArrayLike,
    heights: ArrayLike,
    samples: ArrayLike,
    *,
    mmax: int = 4,
    center: ArrayLike = (0, 0, 0),
    rotation: ArrayLike = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    time: float,
    units: dict[str, float],
    provenance: dict[str, Any] | None = None,
) -> PotentialModel:
    """Project all azimuthal modes together and construct regular tensor splines."""
    radii = validate_knots(radii, "radii", 5)
    heights = validate_knots(heights, "heights")
    center, rotation = validate_frame(center, rotation)
    if radii[0] != 0 or not isinstance(mmax, int) or mmax < 0:
        raise ValueError("radii must start at 0 and mmax must be a nonnegative integer")
    samples = np.asarray(samples, dtype=float)
    if (
        samples.ndim != 3
        or samples.shape[:2] != (len(radii), len(heights))
        or not np.isfinite(samples).all()
    ):
        raise ValueError("finite samples must have shape (nr,nz,nphi)")
    nphi = samples.shape[2]
    if nphi < 4 * mmax + 4:
        raise ValueError("need nphi >= 4*mmax+4 to oversample retained harmonics")
    if not np.allclose(
        samples[0],
        samples[0, :, :1],
        rtol=0,
        atol=1e-12 * max(1, float(np.max(np.abs(samples)))),
    ):
        raise ValueError("axis samples must be independent of azimuth")
    offset = float(samples[0, 0].mean())
    centered = samples - offset
    phi = 2 * np.pi * np.arange(nphi) / nphi
    basis = [np.ones(nphi) / nphi]
    for m in range(1, mmax + 1):
        basis.extend((2 * np.cos(m * phi) / nphi, 2 * np.sin(m * phi) / nphi))
    amplitudes = np.einsum("rzp,mp->rzm", centered, np.asarray(basis))
    s = (radii / radii[-1]) ** 2
    # Polynomial extrapolation of normalized amplitudes at the regular axis.
    weights = np.array(
        [
            np.prod([-s[j] / (s[i] - s[j]) for j in range(1, 5) if j != i])
            for i in range(1, 5)
        ]
    )
    for m in range(1, mmax + 1):
        for idx in (2 * m - 1, 2 * m):
            amplitudes[1:, :, idx] /= (radii[1:] / radii[-1])[:, None] ** m
            amplitudes[0, :, idx] = weights @ amplitudes[1:5, :, idx]
    radial = _cubic_pieces(s, amplitudes)  # (nr-1,4,nz,nmode)
    pieces = _cubic_pieces(heights, radial.transpose(2, 0, 1, 3))
    coefficients = pieces.transpose(4, 2, 0, 3, 1)
    metadata = {
        "schema": SCHEMA,
        "version": VERSION,
        "mmax": mmax,
        "nphi": nphi,
        "basis": "regular-fourier-squared-radius-cubic",
        "outside": "error",
        "center": center.tolist(),
        "world_from_model": rotation.tolist(),
        "frame": "fixed",
        "rmax": float(radii[-1]),
        "time": float(time),
        "units": units,
        "potential_offset": offset,
        "gauge": "subtract A0(0,zmin)",
        "normalization": "A0=mean; Am,Bm=2*mean(Phi*cos,sin)",
        "spline": "tensor not-a-knot cubic in s=(R/Rmax)^2 and signed z",
        "axis": "Rm regularity; normalized modes extrapolated from first 4 positive radii",
        "provenance": provenance or {},
    }
    return PotentialModel(s, heights, coefficients, metadata)
