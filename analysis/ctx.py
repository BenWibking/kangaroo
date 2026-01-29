from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .plan import Domain, FieldRef, Stage


@dataclass
class LoweringContext:
    runtime: Any  # bound C++ Runtime handle
    runmeta: Any  # bound C++ RunMeta handle
    dataset: Any  # Dataset wrapper

    def stage(self, name: str, *, plane: str = "chunk", after: Optional[Iterable[Stage]] = None) -> Stage:
        return Stage(name=name, plane=plane, after=list(after or []))

    def domain(self, *, step: int, level: int, blocks=None) -> Domain:
        return Domain(step=step, level=level, blocks=blocks)

    def temp_field(self, name: str) -> FieldRef:
        fid = self.runtime.alloc_field_id(name)
        return FieldRef(fid, version=0)

    def output_field(self, name: str) -> FieldRef:
        fid = self.runtime.alloc_field_id(name)
        self.runtime.mark_field_persistent(fid, name)
        return FieldRef(fid, version=0)

    def fragment(self, stages):
        return stages
