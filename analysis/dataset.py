from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import _core  # type: ignore


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
    _path: str = field(init=False, default="")
    _metadata_override: Optional[Dict[str, Any]] = field(init=False, default=None)
    _parthenon_reader: Any = field(init=False, default=None)
    _parthenon_field_specs: Dict[int, tuple[str, int, int]] = field(init=False, default_factory=dict)
    _parthenon_loaded_fields: set[int] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        kind, path = self._parse_uri(self.uri)
        self._kind = kind
        self._path = path

        if kind == "parthenon":
            from .parthenon_hdf5 import ParthenonHDF5Reader

            self._parthenon_reader = ParthenonHDF5Reader(path)
            self._metadata_override = self._parthenon_reader.metadata()
            # Route Parthenon data through the in-memory backend to reuse existing runtime path.
            self._h = _core.DatasetHandle("memory://local", self.step, self.level)
        else:
            self._h = _core.DatasetHandle(self.uri, self.step, self.level)
        self._auto_register()

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        if uri.startswith("amrex://"):
            return "amrex", uri[8:]
        if uri.startswith("openpmd://"):
            return "openpmd", uri
        if uri.startswith("parthenon://"):
            return "parthenon", uri[12:]
        if uri.startswith("file://"):
            path = uri[7:]
            import os

            if os.path.isfile(path) and path.endswith((".phdf", ".h5", ".hdf5")):
                return "parthenon", path
            return "amrex", path
        if uri == "memory://local":
            return "memory", uri
        return "unknown", uri

    def _register_parthenon_field(self, name: str, fid: int) -> None:
        meta = self.metadata
        vinfo = meta.get("variable_info", {})
        if name in vinfo:
            ncomp = int(vinfo[name].get("num_components", 1))
            self._parthenon_field_specs[fid] = (name, 0, ncomp)
            return
        for var_name, info in vinfo.items():
            labels = info.get("component_names", [])
            for idx, label in enumerate(labels):
                if label == name:
                    self._parthenon_field_specs[fid] = (var_name, idx, 1)
                    return
        raise KeyError(f"Parthenon field '{name}' not found in metadata")

    def _ensure_parthenon_field_loaded(self, fid: int) -> None:
        if self._kind != "parthenon" or self._parthenon_reader is None:
            return
        if fid in self._parthenon_loaded_fields:
            return
        if fid not in self._parthenon_field_specs:
            return

        var_name, comp_start, comp_count = self._parthenon_field_specs[fid]
        num_blocks = self._parthenon_reader.num_fabs(self.level)
        for block in range(num_blocks):
            payload = self._parthenon_reader.read_block(
                var_name=var_name,
                level=self.level,
                fab=block,
                comp_start=comp_start,
                comp_count=comp_count,
                return_ndarray=False,
            )
            self._h.set_chunk(fid, 0, block, payload["data"])
        self._parthenon_loaded_fields.add(fid)

    def _auto_register(self) -> None:
        if self._kind == "amrex":
            path = self._path
            import os
            header_path = os.path.join(path, "Header")
            if os.path.exists(header_path):
                with open(header_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 2:
                        ncomp = int(lines[1].strip())
                        for i in range(ncomp):
                            name = lines[2 + i].strip()
                            fid = self.field_id(name)
                            self._h.register_field(fid, i)
        elif self._kind == "openpmd":
            meta = self.metadata
            for name in meta.get("var_names", []):
                fid = self.field_id(name)
                self._h.register_field(fid, name)
        elif self._kind == "parthenon":
            meta = self.metadata
            for name in meta.get("var_names", []):
                self.field_id(name)
            for info in meta.get("variable_info", {}).values():
                for label in info.get("component_names", []):
                    self.field_id(label)

    def list_meshes(self) -> list[str]:
        return list(self._h.list_meshes())

    def select_mesh(self, name: str) -> None:
        self._h.select_mesh(name)
        self._fields.clear()
        self._auto_register()

    def register_field(self, name: str, fid: int) -> None:
        self._fields[name] = fid
        if self._kind == "parthenon":
            self._register_parthenon_field(name, fid)
            self._ensure_parthenon_field_loaded(fid)

    def field_id(self, name: str) -> int:
        if name in self._fields:
            fid = self._fields[name]
            if self._kind == "parthenon":
                self._ensure_parthenon_field_loaded(fid)
            return fid
        fid = self.runtime.alloc_field_id(name)
        self._fields[name] = fid
        if self._kind == "parthenon":
            self._register_parthenon_field(name, fid)
            self._ensure_parthenon_field_loaded(fid)
        return fid

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata_override is not None:
            return self._metadata_override
        try:
            return self._h.metadata()
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load dataset metadata for '{self.uri}': {exc}. "
                "Only cell-centered openPMD mesh records are supported."
            ) from exc

    def get_runmeta(self) -> Any:
        from .runmeta import RunMeta, StepMeta, LevelMeta, LevelGeom, BlockBox
        m = self.metadata
        if not m:
            raise RuntimeError("Backend does not provide metadata for RunMeta construction")
        
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
                ref_ratio=m["ref_ratio"][lev] if lev < len(m["ref_ratio"]) else 1
            )
            boxes = [BlockBox(tuple(lo), tuple(hi)) for lo, hi in m["level_boxes"][lev]]
            levels.append(LevelMeta(geom=geom, boxes=boxes))
        
        steps.append(StepMeta(step=self.step, levels=levels))
        return RunMeta(steps=steps)

    def set_chunk(self, *, field: int, version: int = 0, block: int, data: bytes) -> None:
        self._h.set_chunk(field, version, block, data)


def open_dataset(uri: str, *, runmeta: Optional[Any] = None, step: int = 0, level: int = 0, runtime: Any = None) -> Dataset:
    return Dataset(uri=uri, runmeta=runmeta, step=step, level=level, runtime=runtime)
