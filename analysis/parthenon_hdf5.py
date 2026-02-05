from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _VariableInfo:
    name: str
    num_components: int
    component_names: list[str]


class ParthenonHDF5Reader:
    """Reader for Parthenon HDF5 outputs.

    This reader follows metadata conventions written by Parthenon's
    `parthenon_hdf5.cpp` output path.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        try:
            import h5py  # type: ignore
        except Exception as exc:  # pragma: no cover - import-time env issue
            raise ImportError(
                "h5py is required for ParthenonHDF5Reader. Install h5py in your environment."
            ) from exc
        self._h5py = h5py

        with self._h5py.File(self._path, "r") as f:
            self._levels = np.asarray(f["Levels"], dtype=np.int64)
            self._logical_locations = np.asarray(f["LogicalLocations"], dtype=np.int64)
            self._info_attrs = dict(f["Info"].attrs)
            self._var_infos = self._read_var_infos(self._info_attrs)

        if self._levels.ndim != 1:
            raise RuntimeError("Levels dataset must be rank-1")
        if self._logical_locations.shape != (self._levels.shape[0], 3):
            raise RuntimeError("LogicalLocations must have shape (num_blocks, 3)")

        self._blocks_by_level = self._build_blocks_by_level(self._levels)

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        out: list[str] = []
        for item in arr.tolist():
            if isinstance(item, bytes):
                out.append(item.decode("utf-8"))
            else:
                out.append(str(item))
        return out

    @staticmethod
    def _as_int_list(value: Any) -> list[int]:
        if value is None:
            return []
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return [int(x) for x in arr.tolist()]

    def _read_var_infos(self, attrs: dict[str, Any]) -> dict[str, _VariableInfo]:
        names = self._as_str_list(attrs.get("OutputDatasetNames"))
        num_components = self._as_int_list(attrs.get("NumComponents"))
        component_names = self._as_str_list(attrs.get("ComponentNames"))

        infos: dict[str, _VariableInfo] = {}
        if not names:
            return infos

        comp_counts = num_components if len(num_components) == len(names) else [1] * len(names)
        offset = 0
        for name, ncomp in zip(names, comp_counts):
            if ncomp < 1:
                ncomp = 1
            labels = component_names[offset : offset + ncomp]
            offset += ncomp
            infos[name] = _VariableInfo(name=name, num_components=ncomp, component_names=labels)
        return infos

    @staticmethod
    def _build_blocks_by_level(levels: np.ndarray) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for block_idx, lvl in enumerate(levels.tolist()):
            key = int(lvl)
            out.setdefault(key, []).append(block_idx)
        return out

    @property
    def path(self) -> str:
        return self._path

    def num_levels(self) -> int:
        if self._levels.size == 0:
            return 0
        return int(np.max(self._levels)) + 1

    def num_fabs(self, level: int) -> int:
        return len(self._blocks_by_level.get(level, []))

    def variable_names(self) -> list[str]:
        return list(self._var_infos.keys())

    def variable_info(self, name: str) -> dict[str, Any]:
        info = self._var_infos.get(name)
        if info is None:
            raise KeyError(f"Unknown variable: {name}")
        return {
            "name": info.name,
            "num_components": info.num_components,
            "component_names": list(info.component_names),
        }

    def _meshblock_size_xyz(self) -> tuple[int, int, int]:
        size = self._as_int_list(self._info_attrs.get("MeshBlockSize"))
        if len(size) >= 3:
            return int(size[0]), int(size[1]), int(size[2])
        return (1, 1, 1)

    def _prob_lo_hi(self) -> tuple[list[float], list[float]]:
        rgd = np.asarray(self._info_attrs.get("RootGridDomain", []), dtype=np.float64)
        if rgd.size >= 9:
            prob_lo = [float(rgd[0]), float(rgd[3]), float(rgd[6])]
            prob_hi = [float(rgd[1]), float(rgd[4]), float(rgd[7])]
            return prob_lo, prob_hi
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    def _cell_size(self) -> list[list[float]]:
        nlevels = self.num_levels()
        prob_lo, prob_hi = self._prob_lo_hi()
        root_size = self._as_int_list(self._info_attrs.get("RootGridSize"))
        if len(root_size) < 3:
            root_size = [1, 1, 1]
        base = [
            (prob_hi[d] - prob_lo[d]) / max(1, int(root_size[d]))
            for d in range(3)
        ]
        out: list[list[float]] = []
        for lev in range(nlevels):
            scale = float(1 << lev)
            out.append([base[d] / scale for d in range(3)])
        return out

    def _level_boxes(self) -> list[list[tuple[tuple[int, int, int], tuple[int, int, int]]]]:
        nlevels = self.num_levels()
        nx, ny, nz = self._meshblock_size_xyz()
        out: list[list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = []
        for lev in range(nlevels):
            boxes: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
            for block_idx in self._blocks_by_level.get(lev, []):
                lx, ly, lz = self._logical_locations[block_idx].tolist()
                lo = (int(lx * nx), int(ly * ny), int(lz * nz))
                hi = (lo[0] + nx - 1, lo[1] + ny - 1, lo[2] + nz - 1)
                boxes.append((lo, hi))
            out.append(boxes)
        return out

    def metadata(self) -> dict[str, Any]:
        nlevels = self.num_levels()
        prob_lo, prob_hi = self._prob_lo_hi()
        level_boxes = self._level_boxes()

        prob_domain: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for boxes in level_boxes:
            if not boxes:
                prob_domain.append(((0, 0, 0), (-1, -1, -1)))
                continue
            lo = [min(b[0][d] for b in boxes) for d in range(3)]
            hi = [max(b[1][d] for b in boxes) for d in range(3)]
            prob_domain.append((tuple(lo), tuple(hi)))

        return {
            "file_version": "parthenon_hdf5",
            "ncomp": len(self._var_infos),
            "var_names": self.variable_names(),
            "spacedim": 3,
            "time": float(np.asarray(self._info_attrs.get("Time", 0.0)).reshape(-1)[0]),
            "finest_level": max(-1, nlevels - 1),
            "prob_lo": prob_lo,
            "prob_hi": prob_hi,
            "ref_ratio": [1] + [2] * max(0, nlevels - 1),
            "level_boxes": level_boxes,
            "prob_domain": prob_domain,
            "level_steps": [0] * nlevels,
            "cell_size": self._cell_size(),
            "coord_sys": 0,
            "bwidth": 0,
            "mf_name": ["" for _ in range(nlevels)],
            "variable_info": {
                name: {
                    "num_components": info.num_components,
                    "component_names": list(info.component_names),
                }
                for name, info in self._var_infos.items()
            },
        }

    def header(self) -> dict[str, Any]:
        m = self.metadata()
        return {
            "file_version": m["file_version"],
            "ncomp": m["ncomp"],
            "var_names": m["var_names"],
            "spacedim": m["spacedim"],
            "time": m["time"],
            "finest_level": m["finest_level"],
            "ref_ratio": m["ref_ratio"],
            "mf_name": m["mf_name"],
        }

    def _global_block_index(self, level: int, fab: int) -> int:
        blocks = self._blocks_by_level.get(level, [])
        if fab < 0 or fab >= len(blocks):
            raise IndexError(f"fab index out of range for level {level}: {fab}")
        return blocks[fab]

    def read_block(
        self,
        var_name: str,
        level: int,
        fab: int,
        comp_start: int = 0,
        comp_count: int = 1,
        *,
        return_ndarray: bool = False,
    ) -> dict[str, Any]:
        if comp_count <= 0:
            raise ValueError("comp_count must be > 0")
        gblock = self._global_block_index(level, fab)

        with self._h5py.File(self._path, "r") as f:
            if var_name not in f:
                raise KeyError(f"variable dataset not found: {var_name}")
            dset = f[var_name]
            if dset.ndim < 4:
                raise RuntimeError(
                    f"unsupported dataset rank for {var_name}: {dset.ndim}; expected >= 4"
                )
            if dset.shape[0] <= gblock:
                raise RuntimeError(
                    f"dataset {var_name} has {dset.shape[0]} blocks, requested {gblock}"
                )

            block_data = np.asarray(dset[gblock])

        spatial = block_data.shape[-3:]
        nz, ny, nx = int(spatial[0]), int(spatial[1]), int(spatial[2])

        comp_shape = block_data.shape[:-3]
        total_components = int(np.prod(comp_shape)) if comp_shape else 1
        if comp_start < 0 or comp_start + comp_count > total_components:
            raise ValueError(
                f"invalid component range [{comp_start}:{comp_start + comp_count}) for "
                f"{total_components} total components"
            )

        flat = block_data.reshape((total_components, nz, ny, nx))
        sel = np.ascontiguousarray(flat[comp_start : comp_start + comp_count])

        dtype_name = str(sel.dtype)
        payload: dict[str, Any] = {
            "shape": (comp_count, nz, ny, nx),
            "dtype": dtype_name,
            "data": sel if return_ndarray else sel.tobytes(),
        }
        return payload

    def read_fab(
        self,
        level: int,
        fab: int,
        comp_start: int,
        comp_count: int,
        *,
        var_name: str,
        return_ndarray: bool = False,
    ) -> dict[str, Any]:
        """PlotfileReader-compatible alias that requires explicit var_name."""

        return self.read_block(
            var_name,
            level,
            fab,
            comp_start,
            comp_count,
            return_ndarray=return_ndarray,
        )
