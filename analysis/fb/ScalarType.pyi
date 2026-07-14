from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class ScalarType(object):
  Opaque = cast(int, ...)
  U8 = cast(int, ...)
  I64 = cast(int, ...)
  F32 = cast(int, ...)
  F64 = cast(int, ...)

