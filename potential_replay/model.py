"""Version 1 cylindrical Fourier spline reader/evaluator (NumPy only).

Copy this package beside a saved .npz to evaluate it independently of Kangaroo.
All coordinates and returned values use the native units recorded in metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

SCHEMA = "kangaroo.total-potential"
VERSION = 1


def validate_frame(
    center: ArrayLike, rotation: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center must have three finite coordinates")
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("orientation must be a right-handed orthonormal matrix")
    return center, rotation


def validate_knots(values: ArrayLike, name: str, minimum: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or len(values) < minimum
        or not np.isfinite(values).all()
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError(f"{name} must have at least {minimum} finite increasing knots")
    return values


class PotentialModel:
    def __init__(
        self,
        s_knots: ArrayLike,
        z_knots: ArrayLike,
        coefficients: ArrayLike,
        metadata: dict[str, Any],
    ) -> None:
        self.s_knots = validate_knots(s_knots, "s")
        self.z_knots = validate_knots(z_knots, "z")
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        m = self.metadata
        if m.get("schema") != SCHEMA or m.get("version") != VERSION:
            raise ValueError("unsupported potential schema/version")
        order = m["mmax"]
        if not isinstance(order, int) or order < 0:
            raise ValueError("mmax must be a nonnegative integer")
        expected = (2 * order + 1, len(self.s_knots) - 1, len(self.z_knots) - 1, 4, 4)
        if (
            self.coefficients.shape != expected
            or not np.isfinite(self.coefficients).all()
        ):
            raise ValueError(f"invalid coefficient array; expected {expected}")
        if self.s_knots[0] != 0 or self.s_knots[-1] != 1:
            raise ValueError("s knots must span [0,1]")
        self.center, self.rotation = validate_frame(m["center"], m["world_from_model"])
        self.rmax = float(m["rmax"])
        if not np.isfinite(self.rmax) or self.rmax <= 0:
            raise ValueError("rmax must be positive")
        for key in ("time", "potential_offset"):
            if not np.isfinite(m[key]):
                raise ValueError(f"invalid {key}")
        for key in ("length_to_cm", "time_to_s", "potential_to_cm2_s2"):
            if not np.isfinite(m["units"][key]) or m["units"][key] <= 0:
                raise ValueError(f"invalid unit scale {key}")
        if (
            m.get("basis") != "regular-fourier-squared-radius-cubic"
            or m.get("outside") != "error"
            or m.get("frame") != "fixed"
        ):
            raise ValueError("unsupported basis or boundary policy")

    def save(self, path: str | Path) -> None:
        """Write a portable NPZ exclusively, without pickle or overwriting files."""
        with Path(path).open("xb") as stream:
            np.savez_compressed(
                stream,
                s_knots=self.s_knots,
                z_knots=self.z_knots,
                coefficients=self.coefficients,
                metadata=np.array(json.dumps(self.metadata, allow_nan=False)),
            )

    @classmethod
    def load(cls, path: str | Path) -> PotentialModel:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"s_knots", "z_knots", "coefficients", "metadata"}:
                raise ValueError("invalid model archive members")
            return cls(
                data["s_knots"],
                data["z_knots"],
                data["coefficients"],
                json.loads(str(data["metadata"].item())),
            )

    def evaluate(
        self, positions: ArrayLike, *, restore_offset: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (potential, Cartesian acceleration) at world-frame (...,3) points.

        Acceleration is gravitational only. No translating/rotating-frame fictitious
        forces or time interpolation are applied. Any exterior/nonfinite point fails.
        """
        pos = np.asarray(positions, dtype=float)
        if pos.shape[-1:] != (3,) or not np.isfinite(pos).all():
            raise ValueError("positions must be finite (...,3) coordinates")
        shape = pos.shape[:-1]
        local = (pos.reshape(-1, 3) - self.center) @ self.rotation
        x, y, z = local.T
        w = (x + 1j * y) / self.rmax
        s = w.real**2 + w.imag**2
        if np.any((s > 1) | (z < self.z_knots[0]) | (z > self.z_knots[-1])):
            raise ValueError("position outside fitted cylinder")
        i = np.clip(
            np.searchsorted(self.s_knots, s, side="right") - 1, 0, len(self.s_knots) - 2
        )
        j = np.clip(
            np.searchsorted(self.z_knots, z, side="right") - 1, 0, len(self.z_knots) - 2
        )
        ds, dz = np.diff(self.s_knots)[i], np.diff(self.z_knots)[j]
        u, v = (s - self.s_knots[i]) / ds, (z - self.z_knots[j]) / dz
        up = np.array([np.ones_like(u), u, u * u, u * u * u]).T
        vp = np.array([np.ones_like(v), v, v * v, v * v * v]).T
        du = (
            np.array([np.zeros_like(u), np.ones_like(u), 2 * u, 3 * u * u]).T
            / ds[:, None]
        )
        dv = (
            np.array([np.zeros_like(v), np.ones_like(v), 2 * v, 3 * v * v]).T
            / dz[:, None]
        )
        c = self.coefficients[:, i, j]
        values = np.einsum("mnab,na,nb->mn", c, up, vp)
        deriv_s = np.einsum("mnab,na,nb->mn", c, du, vp)
        deriv_z = np.einsum("mnab,na,nb->mn", c, up, dv)
        phi = values[0].copy()
        gx, gy = deriv_s[0] * 2 * x / self.rmax**2, deriv_s[0] * 2 * y / self.rmax**2
        gz = deriv_z[0].copy()
        for m in range(1, self.metadata["mmax"] + 1):
            power = w**m
            d = m * w ** (m - 1) / self.rmax
            for idx, factor, fx, fy in (
                (2 * m - 1, power.real, d.real, -d.imag),
                (2 * m, power.imag, d.imag, d.real),
            ):
                phi += factor * values[idx]
                gx += fx * values[idx] + factor * deriv_s[idx] * 2 * x / self.rmax**2
                gy += fy * values[idx] + factor * deriv_s[idx] * 2 * y / self.rmax**2
                gz += factor * deriv_z[idx]
        if restore_offset:
            phi += self.metadata["potential_offset"]
        acceleration = -np.column_stack((gx, gy, gz)) @ self.rotation.T
        return phi.reshape(shape), acceleration.reshape(shape + (3,))

    def potential(
        self, positions: ArrayLike, *, restore_offset: bool = False
    ) -> np.ndarray:
        return self.evaluate(positions, restore_offset=restore_offset)[0]

    def acceleration(self, positions: ArrayLike) -> np.ndarray:
        return self.evaluate(positions)[1]
