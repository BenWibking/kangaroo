from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class ShapeRuleKind(object):
  Block = cast(int, ...)
  Fixed = cast(int, ...)
  LikeInput = cast(int, ...)
  Dynamic = cast(int, ...)

