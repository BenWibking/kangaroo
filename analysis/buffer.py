from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, TypeAlias

import numpy as np


class DType(str, Enum):
    OPAQUE = "opaque"
    U8 = "u8"
    I64 = "i64"
    F32 = "f32"
    F64 = "f64"


class InitPolicy(str, Enum):
    UNINITIALIZED = "uninitialized"
    ZERO = "zero"


class DynamicUpperBoundKind(str, Enum):
    LITERAL = "literal"
    LIKE_INPUT = "like_input"
    BACKEND_CHUNK = "backend_chunk"
    AMR_SUBBOX_PACK = "amr_subbox_pack"


@dataclass(frozen=True)
class DynamicUpperBound:
    kind: DynamicUpperBoundKind
    value: int | None = None
    input_index: int | None = None

    @classmethod
    def literal(cls, elements: int) -> "DynamicUpperBound":
        return cls(DynamicUpperBoundKind.LITERAL, value=int(elements))

    @classmethod
    def like_input(cls, input_index: int) -> "DynamicUpperBound":
        return cls(DynamicUpperBoundKind.LIKE_INPUT, input_index=int(input_index))

    @classmethod
    def backend_chunk(cls, input_index: int = 0) -> "DynamicUpperBound":
        return cls(DynamicUpperBoundKind.BACKEND_CHUNK, input_index=int(input_index))

    @classmethod
    def amr_subbox_pack(cls) -> "DynamicUpperBound":
        return cls(DynamicUpperBoundKind.AMR_SUBBOX_PACK)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind.value}
        if self.value is not None:
            if self.value < 0:
                raise ValueError("dynamic upper bound must be non-negative")
            out["value"] = self.value
        if self.input_index is not None:
            if self.input_index < 0:
                raise ValueError("dynamic upper-bound input index must be non-negative")
            out["input_index"] = self.input_index
        return out


@dataclass(frozen=True)
class BlockShape:
    components: int = 1


@dataclass(frozen=True)
class FixedShape:
    extents: tuple[int, ...]

    def __init__(self, extents: tuple[int, ...] | list[int]):
        object.__setattr__(self, "extents", tuple(int(value) for value in extents))


@dataclass(frozen=True)
class LikeInputShape:
    input_index: int


@dataclass(frozen=True)
class DynamicShape:
    upper_bound: DynamicUpperBound


BufferShape: TypeAlias = BlockShape | FixedShape | LikeInputShape | DynamicShape


@dataclass(frozen=True)
class BufferSpec:
    dtype: DType
    shape: BufferShape
    init: InitPolicy = InitPolicy.UNINITIALIZED

    def __post_init__(self) -> None:
        if isinstance(self.shape, BlockShape):
            if self.shape.components < 1:
                raise ValueError("block components must be positive")
            if self.dtype is DType.OPAQUE:
                raise ValueError("opaque buffers cannot use block shape")
        elif isinstance(self.shape, FixedShape):
            if not 1 <= len(self.shape.extents) <= 4:
                raise ValueError("fixed buffer rank must be between 1 and 4")
            if any(value <= 0 for value in self.shape.extents):
                raise ValueError("fixed buffer extents must be positive")
            if self.dtype is DType.OPAQUE and len(self.shape.extents) != 1:
                raise ValueError("opaque buffers must have rank 1")
        elif isinstance(self.shape, LikeInputShape):
            if self.shape.input_index < 0:
                raise ValueError("like-input index must be non-negative")
            if self.dtype is DType.OPAQUE:
                raise ValueError("opaque buffers cannot use like-input shape")
        elif not isinstance(self.shape, DynamicShape):
            raise TypeError(f"unsupported buffer shape: {type(self.shape).__name__}")

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.shape, BlockShape):
            shape = {"kind": "block", "components": self.shape.components}
        elif isinstance(self.shape, FixedShape):
            shape = {"kind": "fixed", "extents": list(self.shape.extents)}
        elif isinstance(self.shape, LikeInputShape):
            shape = {"kind": "like_input", "input_index": self.shape.input_index}
        else:
            shape = {"kind": "dynamic", "upper_bound": self.shape.upper_bound.to_dict()}
        return {"dtype": self.dtype.value, "shape": shape, "init": self.init.value}


NUMPY_DTYPES: Mapping[DType, np.dtype[Any]] = {
    DType.U8: np.dtype(np.uint8),
    DType.I64: np.dtype(np.int64),
    DType.F32: np.dtype(np.float32),
    DType.F64: np.dtype(np.float64),
}

DTYPE_TAGS: Mapping[str, DType] = {
    "opaque": DType.OPAQUE,
    "u8": DType.U8,
    "uint8": DType.U8,
    "i64": DType.I64,
    "int64": DType.I64,
    "f32": DType.F32,
    "float32": DType.F32,
    "f64": DType.F64,
    "float64": DType.F64,
}


def parse_dtype_tag(dtype: DType | str) -> DType:
    if isinstance(dtype, DType):
        return dtype
    try:
        return DTYPE_TAGS[str(dtype)]
    except KeyError as exc:
        raise TypeError(f"unsupported dtype tag: {dtype}") from exc


def numpy_dtype(dtype: DType | str) -> np.dtype[Any]:
    dtype = parse_dtype_tag(dtype)
    if dtype is DType.OPAQUE:
        raise TypeError("opaque payloads do not have a NumPy numeric dtype")
    return NUMPY_DTYPES[dtype]


def dtype_from_numpy(dtype: np.dtype[Any] | str | type[Any]) -> DType:
    normalized = np.dtype(dtype)
    for tag, candidate in NUMPY_DTYPES.items():
        if normalized == candidate:
            return tag
    raise TypeError(f"unsupported NumPy dtype: {normalized}")
