"""Advanced typed plan IR and buffer-contract interfaces."""

from analysis.buffer import (
    BlockShape,
    BufferSpec,
    DType,
    DynamicShape,
    DynamicUpperBound,
    FixedShape,
    InitPolicy,
    LikeInputShape,
)
from analysis.ctx import LoweringContext
from analysis.plan import (
    DependencyRule,
    Domain,
    FieldRef,
    GraphReduceSpec,
    OutputRef,
    Plan,
    Stage,
    TaskTemplate,
)

__all__ = [
    "BlockShape",
    "BufferSpec",
    "DType",
    "DependencyRule",
    "Domain",
    "DynamicShape",
    "DynamicUpperBound",
    "FieldRef",
    "FixedShape",
    "GraphReduceSpec",
    "InitPolicy",
    "LikeInputShape",
    "LoweringContext",
    "OutputRef",
    "Plan",
    "Stage",
    "TaskTemplate",
]

