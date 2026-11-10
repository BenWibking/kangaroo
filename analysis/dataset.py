from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from . import _core  # type: ignore
from .buffer import DType, dtype_from_numpy, materialize_payload, parse_dtype_tag


@dataclass(frozen=True)
class DatasetMetadata:
    dataset: Dict[str, Any]
    runmeta: Any


@dataclass
class Dataset:
    uri: str
    runmeta: Optional[Any] = None
    step: int = 0
    level: int = 0
    runtime: Any = None
    _h: Any = field(init=False)
    _fields: Dict[str, int] = field(default_factory=dict)
    _kind: str = field(init=False, default="unknown")

    def __post_init__(self) -> None:
        self._h = _core.DatasetHandle(self.uri, self.step, self.level)
        self._kind = str(self._h.kind())
        self._auto_register()

    @property
    def kind(self) -> str:
        return self._kind

    def _auto_register(self) -> None:
        meta = self.metadata
        names = list(meta.get("var_names", []))
        for info in meta.get("variable_info", {}).values():
            names.extend(info.get("component_names", []))
        for name in dict.fromkeys(names):
            fid = self.field_id(name)
            self._h.register_field(fid, name)

    def list_meshes(self) -> list[str]:
        return list(self._h.list_meshes())

    def select_mesh(self, name: str) -> None:
        self._h.select_mesh(name)
        self._fields.clear()
        self._auto_register()

    def list_particle_types(self) -> list[str]:
        if not hasattr(self._h, "list_particle_types"):
            return []
        return list(self._h.list_particle_types())

    def list_particle_fields(self, particle_type: str) -> list[str]:
        if not hasattr(self._h, "list_particle_fields"):
            return []
        return list(self._h.list_particle_fields(particle_type))

    def get_particle_array(self, particle_type: str, field: str) -> np.ndarray:
        raise RuntimeError(
            "Full particle array materialization is disabled. "
            "Use get_particle_chunk_count() and get_particle_chunk_array()."
        )

    def get_particle_chunk_count(self, particle_type: str) -> int:
        if not hasattr(self._h, "particle_chunk_count"):
            raise RuntimeError("Particle chunk metadata is not available in this runtime")
        return int(self._h.particle_chunk_count(particle_type))

    def get_particle_chunk_array(self, particle_type: str, field: str, chunk_index: int) -> np.ndarray:
        if not hasattr(self._h, "read_particle_field_chunk"):
            raise RuntimeError("Particle chunk field access is not available in this runtime")
        payload = self._h.read_particle_field_chunk(particle_type, field, int(chunk_index))
        return materialize_payload(payload)

    def register_field(self, name: str, fid: int) -> None:
        self._fields[name] = fid
        self._h.register_field(fid, name)

    def field_id(self, name: str) -> int:
        if name in self._fields:
            return self._fields[name]
        fid = self.runtime.alloc_field_id(name)
        self._fields[name] = fid
        return fid

    @property
    def metadata(self) -> Dict[str, Any]:
        try:
            return self._h.metadata()
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load dataset metadata for '{self.uri}': {exc}"
            ) from exc

    def metadata_bundle(self, periodic: Optional[tuple[bool, bool, bool]] = None) -> DatasetMetadata:
        return DatasetMetadata(dataset=self.metadata, runmeta=self.get_runmeta(periodic=periodic))

    def get_runmeta(self, periodic: Optional[tuple[bool, bool, bool]] = None) -> Any:
        from .runmeta import RunMeta, StepMeta, LevelMeta, LevelGeom, BlockBox
        if self.runmeta is not None:
            return self.runmeta
        m = self.metadata
        if not m:
            raise RuntimeError("Backend does not provide metadata for RunMeta construction")

        meta_periodic = m.get("is_periodic")
        is_periodic = tuple(bool(v) for v in meta_periodic) if meta_periodic is not None else (False, False, False)
        if periodic is not None:
            is_periodic = periodic
        
        # AMReX-specific RunMeta construction
        steps = []
        # PlotfileBackend currently represents a single step.
        # We'll use the step passed during open_dataset or 0.
        levels = []
        for lev in range(m["finest_level"] + 1):
            dx = m["cell_size"][lev][0]
            geom = LevelGeom(
                dx=(dx, dx, dx),
                x0=tuple(m["prob_lo"]),
                index_origin=tuple(m["prob_domain"][lev][0]),
                is_periodic=is_periodic,
                ref_ratio=m["ref_ratio"][lev] if lev < len(m["ref_ratio"]) else 1
            )
            boxes = [BlockBox(tuple(lo), tuple(hi)) for lo, hi in m["level_boxes"][lev]]
            levels.append(LevelMeta(geom=geom, boxes=boxes))
        
        steps.append(StepMeta(step=self.step, levels=levels))
        particle_species: Dict[str, int] = {}
        for ptype in self.list_particle_types():
            try:
                particle_species[ptype] = self.get_particle_chunk_count(ptype)
            except Exception:
                continue
        return RunMeta(steps=steps, particle_species=particle_species)

    def resolve_field(self, var: str | None) -> tuple[str, int, Dict[str, Any]]:
        meta = self.metadata
        var_names = list(meta.get("var_names", []))

        if self.kind != "openpmd":
            if not var:
                if not var_names:
                    raise RuntimeError("Dataset metadata does not list any fields")
                var = var_names[0]
            return var, self.field_id(var), meta

        mesh_names = list(meta.get("mesh_names", []))
        if not mesh_names and not var_names:
            raise RuntimeError("openPMD metadata does not list any meshes or fields")

        if not var:
            if var_names:
                var = var_names[0]
            else:
                raise RuntimeError("openPMD metadata does not list any fields")
            return var, self.field_id(var), meta

        if "/" in var:
            mesh, comp = var.split("/", 1)
            if mesh:
                if mesh_names and mesh not in mesh_names:
                    raise RuntimeError(f"openPMD mesh '{mesh}' not found")
                self.select_mesh(mesh)
                meta = self.metadata
                var_names = list(meta.get("var_names", []))
                if comp:
                    candidate = f"{mesh}/{comp}"
                    if candidate in var_names:
                        return candidate, self.field_id(candidate), meta
                    if comp in var_names:
                        return comp, self.field_id(comp), meta
                    raise RuntimeError(f"openPMD mesh '{mesh}' does not contain component '{comp}'")
            if var in var_names:
                return var, self.field_id(var), meta
            raise RuntimeError(f"openPMD field '{var}' not found")

        if var in mesh_names:
            self.select_mesh(var)
            meta = self.metadata
            var_names = list(meta.get("var_names", []))
            if var_names:
                first = var_names[0]
                return first, self.field_id(first), meta
            raise RuntimeError(f"openPMD mesh '{var}' has no fields")

        if var in var_names:
            return var, self.field_id(var), meta

        if len(mesh_names) == 1:
            candidate = f"{mesh_names[0]}/{var}"
            if candidate in var_names:
                return candidate, self.field_id(candidate), meta

        raise RuntimeError(f"openPMD field '{var}' not found")

    @staticmethod
    def _axis_index(axis: str | int) -> int:
        if isinstance(axis, int):
            if axis in (0, 1, 2):
                return axis
            raise ValueError("axis index must be 0, 1, or 2")
        try:
            return {"x": 0, "y": 1, "z": 2}[axis]
        except KeyError as exc:
            raise ValueError("axis must be one of x, y, z") from exc

    @staticmethod
    def _parse_resolution_override(
        resolution: str | tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if resolution is None:
            return None
        if isinstance(resolution, tuple):
            nx, ny = resolution
        else:
            nx, ny = (int(v.strip()) for v in resolution.split(","))
        if nx <= 0 or ny <= 0:
            raise ValueError("resolution must be positive")
        return int(nx), int(ny)

    def plane_geometry(
        self,
        *,
        axis: str | int,
        level: int = 0,
        coord: float | None = None,
        zoom: float = 1.0,
        resolution: str | tuple[int, int] | None = None,
    ) -> Dict[str, Any]:
        if zoom <= 0:
            raise ValueError("--zoom must be a positive number")

        meta = self.metadata
        prob_lo = meta["prob_lo"]
        prob_hi = meta["prob_hi"]
        domain_lo, domain_hi = meta["prob_domain"][level]
        nx, ny, nz = (int(domain_hi[i]) - int(domain_lo[i]) + 1 for i in range(3))
        cell_size = meta["cell_size"][level]

        def bounds(i: int, n: int, dlo: int) -> tuple[float, float]:
            dx = float(cell_size[i])
            if not np.isfinite(dx) or dx == 0.0:
                dx = (float(prob_hi[i]) - float(prob_lo[i])) / max(1, n)
            lo = float(prob_lo[i]) + float(dlo) * dx
            hi = lo + dx * float(n)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
                return lo, lo + (float(n) if n else 1.0)
            return lo, hi

        x_lo, x_hi = bounds(0, nx, int(domain_lo[0]))
        y_lo, y_hi = bounds(1, ny, int(domain_lo[1]))
        z_lo, z_hi = bounds(2, nz, int(domain_lo[2]))

        axis_idx = self._axis_index(axis)
        axis_bounds = ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))[axis_idx]
        if coord is None:
            dx_axis = float(cell_size[axis_idx])
            if np.isfinite(dx_axis) and dx_axis != 0.0:
                coord = float(prob_lo[axis_idx]) + (
                    ((int(domain_lo[axis_idx]) + int(domain_hi[axis_idx])) // 2) + 0.5
                ) * dx_axis
            else:
                coord = 0.5 * (axis_bounds[0] + axis_bounds[1])

        if axis_idx == 2:
            rect, out_res, labels, plane = (x_lo, y_lo, x_hi, y_hi), (nx, ny), ("x", "y"), "xy"
        elif axis_idx == 1:
            rect, out_res, labels, plane = (x_lo, z_lo, x_hi, z_hi), (nx, nz), ("x", "z"), "xz"
        else:
            rect, out_res, labels, plane = (y_lo, z_lo, y_hi, z_hi), (ny, nz), ("y", "z"), "yz"

        parsed_res = self._parse_resolution_override(resolution)
        if parsed_res is not None:
            out_res = parsed_res

        if zoom != 1.0:
            xm, ym = 0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3])
            hw, hh = 0.5 * (rect[2] - rect[0]) / zoom, 0.5 * (rect[3] - rect[1]) / zoom
            rect = (xm - hw, ym - hh, xm + hw, ym + hh)

        return {
            "coord": float(coord),
            "rect": rect,
            "resolution": out_res,
            "labels": labels,
            "plane": plane,
            "axis_index": axis_idx,
            "axis_bounds": axis_bounds,
        }

    def set_chunk(
        self,
        *,
        field: int,
        version: int = 0,
        block: int,
        data: bytes | bytearray | memoryview | np.ndarray,
        dtype: DType | str | None = None,
        shape: tuple[int, ...] | None = None,
        opaque: bool = False,
    ) -> None:
        if isinstance(data, np.ndarray):
            array = np.ascontiguousarray(data)
            inferred_dtype = dtype_from_numpy(array.dtype)
            if dtype is not None and parse_dtype_tag(dtype) is not inferred_dtype:
                raise ValueError("explicit dtype does not match NumPy array dtype")
            if shape is not None and tuple(shape) != tuple(array.shape):
                raise ValueError("explicit shape does not match NumPy array shape")
            self._h.set_chunk(
                field,
                version,
                block,
                array.tobytes(order="C"),
                inferred_dtype.value,
                list(array.shape),
            )
            return

        raw = bytes(data)
        if opaque:
            if dtype is not None or shape is not None:
                raise ValueError("opaque writes cannot also specify numeric dtype or shape")
            self._h.set_chunk(field, version, block, raw, DType.OPAQUE.value, [len(raw)])
            return
        if dtype is None or shape is None:
            raise ValueError("raw-byte chunk writes require dtype and shape, or opaque=True")
        self._h.set_chunk(field, version, block, raw, parse_dtype_tag(dtype).value, list(shape))


def _resolve_dataset_uri(path_or_uri: str) -> str:
    if path_or_uri.startswith(("amrex://", "openpmd://", "parthenon://", "file://", "memory://")):
        return path_or_uri
    if "://" in path_or_uri:
        return path_or_uri
    if not os.path.exists(path_or_uri):
        raise FileNotFoundError(path_or_uri)
    if os.path.isdir(path_or_uri) and os.path.exists(os.path.join(path_or_uri, "Header")):
        return f"amrex://{path_or_uri}"
    if os.path.isfile(path_or_uri) and path_or_uri.endswith((".phdf", ".h5", ".hdf5")):
        return f"parthenon://{path_or_uri}"
    return f"openpmd://{path_or_uri}"


def open_dataset(
    uri: str,
    *,
    runmeta: Optional[Any] = None,
    step: int = 0,
    level: int | None = None,
    runtime: Any = None,
) -> Dataset:
    uri = _resolve_dataset_uri(uri)
    return Dataset(uri=uri, runmeta=runmeta, step=step, level=(0 if level is None else level), runtime=runtime)
