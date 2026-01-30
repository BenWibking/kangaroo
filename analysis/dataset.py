from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import _core  # type: ignore


@dataclass
class Dataset:
    uri: str
    runmeta: Any
    step: int
    level: int
    runtime: Any
    _h: Any = field(init=False)
    _fields: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._h = _core.DatasetHandle(self.uri, self.step, self.level)

    def register_field(self, name: str, fid: int) -> None:
        self._fields[name] = fid

    def field_id(self, name: str) -> int:
        if name in self._fields:
            return self._fields[name]
        fid = self.runtime.alloc_field_id(name)
        self._fields[name] = fid
        return fid


def open_dataset(uri: str, *, runmeta: Any, step: int, level: int, runtime: Any) -> Dataset:
    return Dataset(uri=uri, runmeta=runmeta, step=step, level=level, runtime=runtime)
