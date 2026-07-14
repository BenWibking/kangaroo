from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class DynamicUpperBoundKind(object):
  Literal = cast(int, ...)
  LikeInput = cast(int, ...)
  BackendChunk = cast(int, ...)
  AmrSubboxPack = cast(int, ...)

