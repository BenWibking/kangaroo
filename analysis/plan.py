from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

from .buffer import BufferSpec
from .kernel_params import KernelParams, NoKernelParams, validate_kernel_params

Plane = Literal["chunk", "graph", "mixed"]


@dataclass(frozen=True)
class FieldRef:
    field: int
    version: int = 0
    domain: Domain | None = None


@dataclass(frozen=True)
class Domain:
    step: int
    level: int
    blocks: Optional[Sequence[int]] = None  # None means "all blocks on level"


@dataclass(frozen=True)
class OutputRef:
    field: FieldRef
    buffer: BufferSpec


@dataclass(frozen=True)
class DependencyRule:
    kind: Literal["None", "FaceNeighbors"] = "None"
    width: int = 0
    faces: tuple[bool, bool, bool, bool, bool, bool] = (
        True,
        True,
        True,
        True,
        True,
        True,
    )
    halo_inputs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.width < 0:
            raise ValueError("dependency width must be non-negative")
        if len(self.faces) != 6:
            raise ValueError("dependency faces must contain 6 values")
        if any(index < 0 for index in self.halo_inputs):
            raise ValueError("halo input indices must be non-negative")


@dataclass(frozen=True)
class GraphReduceSpec:
    fan_in: int = 1
    num_inputs: int = 0
    input_base: int = 0
    output_base: int = 0
    input_blocks: tuple[int, ...] = ()
    output_blocks: tuple[int, ...] = ()
    group_offsets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.fan_in < 1:
            raise ValueError("graph reduction fan_in must be positive")
        if self.num_inputs < 1:
            raise ValueError("graph reduction num_inputs must be positive")
        if self.input_blocks and len(self.input_blocks) != self.num_inputs:
            raise ValueError("graph reduction input_blocks must match num_inputs")
        if self.group_offsets:
            if self.group_offsets[0] != 0 or self.group_offsets[-1] != self.num_inputs:
                raise ValueError("graph reduction group_offsets must span num_inputs")
            if any(
                right <= left
                for left, right in zip(self.group_offsets, self.group_offsets[1:])
            ):
                raise ValueError("graph reduction group_offsets must be strictly increasing")
            groups = len(self.group_offsets) - 1
        else:
            groups = (self.num_inputs + self.fan_in - 1) // self.fan_in
        if self.output_blocks and len(self.output_blocks) != groups:
            raise ValueError("graph reduction output_blocks must match group count")


@dataclass
class TaskTemplate:
    name: str
    plane: Plane
    kernel: str
    domain: Domain
    inputs: List[FieldRef]
    outputs: List[OutputRef]
    deps: DependencyRule = field(default_factory=DependencyRule)
    params: KernelParams = field(default_factory=NoKernelParams)
    graph_reduce: GraphReduceSpec | None = None

    def __post_init__(self) -> None:
        validate_kernel_params(self.kernel, self.params)
        if self.plane == "graph" and self.graph_reduce is None:
            raise ValueError("graph task templates require a GraphReduceSpec")
        if self.plane != "graph" and self.graph_reduce is not None:
            raise ValueError("only graph task templates accept a GraphReduceSpec")


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
        outputs: List[OutputRef],
        deps: DependencyRule = DependencyRule(),
        params: KernelParams = NoKernelParams(),
        graph_reduce: GraphReduceSpec | None = None,
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
                graph_reduce=graph_reduce,
            )
        )


@dataclass
class Plan:
    stages: List[Stage]

    def topo_stages(self) -> List[Stage]:
        out: List[Stage] = []
        seen: set[int] = set()
        visiting: set[int] = set()

        def visit(stage: Stage) -> None:
            sid = id(stage)
            if sid in seen:
                return
            if sid in visiting:
                raise ValueError(f"stage dependency cycle at {stage.name!r}")
            visiting.add(sid)
            for parent in stage.after:
                visit(parent)
            visiting.remove(sid)
            seen.add(sid)
            out.append(stage)

        for stage in self.stages:
            visit(stage)
        return out
