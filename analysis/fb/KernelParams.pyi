from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from analysis.fb.AmrSubboxPackParams import AmrSubboxPackParams
from analysis.fb.CylindricalFluxParams import CylindricalFluxParams
from analysis.fb.FieldExprParams import FieldExprParams
from analysis.fb.FiniteOnlyParams import FiniteOnlyParams
from analysis.fb.FluxSurfaceParams import FluxSurfaceParams
from analysis.fb.GradStencilParams import GradStencilParams
from analysis.fb.Histogram1DParams import Histogram1DParams
from analysis.fb.Histogram2DParams import Histogram2DParams
from analysis.fb.ParticleCicGridParams import ParticleCicGridParams
from analysis.fb.ParticleCicProjectionParams import ParticleCicProjectionParams
from analysis.fb.ParticleFieldParams import ParticleFieldParams
from analysis.fb.ParticleHistogramParams import ParticleHistogramParams
from analysis.fb.PlotfileLoadParams import PlotfileLoadParams
from analysis.fb.ScalarParams import ScalarParams
from analysis.fb.SliceFinalizeParams import SliceFinalizeParams
from analysis.fb.TopKModesParams import TopKModesParams
from analysis.fb.UniformProjectionParams import UniformProjectionParams
from analysis.fb.UniformSliceCellParams import UniformSliceCellParams
from analysis.fb.UniformSliceParams import UniformSliceParams
from analysis.fb.ValuesParams import ValuesParams
from flatbuffers import table
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class KernelParams(object):
  NONE = cast(int, ...)
  AmrSubboxPackParams = cast(int, ...)
  GradStencilParams = cast(int, ...)
  PlotfileLoadParams = cast(int, ...)
  FluxSurfaceParams = cast(int, ...)
  CylindricalFluxParams = cast(int, ...)
  UniformSliceCellParams = cast(int, ...)
  UniformProjectionParams = cast(int, ...)
  FieldExprParams = cast(int, ...)
  UniformSliceParams = cast(int, ...)
  Histogram1DParams = cast(int, ...)
  Histogram2DParams = cast(int, ...)
  ParticleFieldParams = cast(int, ...)
  ParticleCicGridParams = cast(int, ...)
  ParticleCicProjectionParams = cast(int, ...)
  ScalarParams = cast(int, ...)
  ValuesParams = cast(int, ...)
  FiniteOnlyParams = cast(int, ...)
  ParticleHistogramParams = cast(int, ...)
  TopKModesParams = cast(int, ...)
  SliceFinalizeParams = cast(int, ...)
def KernelParamsCreator(union_type: typing.Literal[KernelParams.NONE, KernelParams.AmrSubboxPackParams, KernelParams.GradStencilParams, KernelParams.PlotfileLoadParams, KernelParams.FluxSurfaceParams, KernelParams.CylindricalFluxParams, KernelParams.UniformSliceCellParams, KernelParams.UniformProjectionParams, KernelParams.FieldExprParams, KernelParams.UniformSliceParams, KernelParams.Histogram1DParams, KernelParams.Histogram2DParams, KernelParams.ParticleFieldParams, KernelParams.ParticleCicGridParams, KernelParams.ParticleCicProjectionParams, KernelParams.ScalarParams, KernelParams.ValuesParams, KernelParams.FiniteOnlyParams, KernelParams.ParticleHistogramParams, KernelParams.TopKModesParams, KernelParams.SliceFinalizeParams], table: table.Table) -> typing.Union[None, AmrSubboxPackParams, GradStencilParams, PlotfileLoadParams, FluxSurfaceParams, CylindricalFluxParams, UniformSliceCellParams, UniformProjectionParams, FieldExprParams, UniformSliceParams, Histogram1DParams, Histogram2DParams, ParticleFieldParams, ParticleCicGridParams, ParticleCicProjectionParams, ScalarParams, ValuesParams, FiniteOnlyParams, ParticleHistogramParams, TopKModesParams, SliceFinalizeParams]: ...

