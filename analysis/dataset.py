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

    def __post_init__(self) -> None:
        self._h = _core.DatasetHandle(self.uri, self.step, self.level)
        self._auto_register()

    def _auto_register(self) -> None:
        if self.uri.startswith("amrex://") or self.uri.startswith("file://"):
            path = self.uri
            if self.uri.startswith("amrex://"):
                path = self.uri[8:]
            elif self.uri.startswith("file://"):
                path = self.uri[7:]
            
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
        elif self.uri.startswith("openpmd://"):
            meta = self.metadata
            for name in meta.get("var_names", []):
                fid = self.field_id(name)
                self._h.register_field(fid, name)

    def list_meshes(self) -> list[str]:
        return list(self._h.list_meshes())

    def select_mesh(self, name: str) -> None:
        self._h.select_mesh(name)
        self._fields.clear()
        self._auto_register()

    def register_field(self, name: str, fid: int) -> None:
        self._fields[name] = fid

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
