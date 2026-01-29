from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

Plane = str  # "chunk" | "graph" | "mixed" (v0 supports only "chunk")


@dataclass(frozen=True)
class FieldRef:
    field: int
    version: int = 0


@dataclass(frozen=True)
class Domain:
    step: int
    level: int
    blocks: Optional[Sequence[int]] = None  # None means "all blocks on level"


@dataclass
class TaskTemplate:
    name: str
    plane: Plane
    kernel: str
    domain: Domain
    inputs: List[FieldRef]
    outputs: List[FieldRef]
    deps: Dict[str, Any] = field(default_factory=lambda: {"kind": "None"})
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage:
    name: str
    plane: Plane = "chunk"
    after: List["Stage"] = field(default_factory=list)
    templates: List[TaskTemplate] = field(default_factory=list)

    def map_blocks(
        self,
        *,
        name: str,
        kernel: str,
        domain: Domain,
        inputs: List[FieldRef],
        outputs: List[FieldRef],
        deps: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        self.templates.append(
            TaskTemplate(
                name=name,
                plane=self.plane,
                kernel=kernel,
                domain=domain,
                inputs=inputs,
                outputs=outputs,
                deps=deps,
                params=params,
            )
        )


@dataclass
class Plan:
    stages: List[Stage]

    def topo_stages(self) -> List[Stage]:
        out: List[Stage] = []
        seen: set[int] = set()

        def visit(stage: Stage) -> None:
            sid = id(stage)
            if sid in seen:
                return
            for parent in stage.after:
                visit(parent)
            seen.add(sid)
            out.append(stage)

        for stage in self.stages:
            visit(stage)
        return out
