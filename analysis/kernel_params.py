from __future__ import annotations

from dataclasses import dataclass
from math import nan
from typing import TypeAlias

Index3: TypeAlias = tuple[int, int, int]
CoveredBox: TypeAlias = tuple[Index3, Index3]
CoveredBoxes: TypeAlias = tuple[CoveredBox, ...]


@dataclass(frozen=True)
class NoKernelParams:
    pass


@dataclass(frozen=True)
class AmrSubboxPackParams:
    input_field: int = -1
    input_version: int = 0
    input_step: int = 0
    input_level: int = 0
    halo_cells: int = 1


@dataclass(frozen=True)
class GradStencilParams:
    input_field: int = -1
    input_version: int = 0
    input_step: int = 0
    input_level: int = 0
    stencil_radius: int = 1


@dataclass(frozen=True)
class PlotfileLoadParams:
    plotfile: str = ""
    level: int = 0
    comp: int = 0


@dataclass(frozen=True)
class FluxSurfaceParams:
    radii: tuple[float, ...] = ()
    radius_indices: tuple[int, ...] = ()
    temperature_bins: tuple[float, ...] = ()
    num_radii: int = 0
    gamma: float = 5.0 / 3.0
    covered_boxes: CoveredBoxes = ()


@dataclass(frozen=True)
class CylindricalFluxParams:
    radius: float = 0.0
    heights: tuple[float, ...] = ()
    height_indices: tuple[int, ...] = ()
    temperature_bins: tuple[float, ...] = ()
    num_heights: int = 0
    gamma: float = 5.0 / 3.0
    covered_boxes: CoveredBoxes = ()


@dataclass(frozen=True)
class UniformSliceCellParams:
    axis: int = 2
    coord: float = 0.0
    plane_index: int | None = None
    rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    resolution: tuple[int, int] = (1, 1)
    covered_boxes: CoveredBoxes = ()


@dataclass(frozen=True)
class UniformProjectionParams:
    axis: int = 2
    axis_bounds: tuple[float, float] = (0.0, 1.0)
    rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    resolution: tuple[int, int] = (1, 1)
    covered_boxes: CoveredBoxes = ()


@dataclass(frozen=True)
class FieldExprParams:
    expression: str = ""
    variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class UniformSliceParams:
    axis: int = 2
    coord: float = 0.0
    rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    resolution: tuple[int, int] = (1, 1)


@dataclass(frozen=True)
class Histogram1DParams:
    range: tuple[float, float] = (0.0, 1.0)
    bins: int = 1
    covered_boxes: CoveredBoxes = ()


@dataclass(frozen=True)
class Histogram2DParams:
    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (0.0, 1.0)
    bins: tuple[int, int] = (1, 1)
    weight_mode: str = "input"
    covered_boxes: CoveredBoxes = ()


@dataclass(frozen=True)
class ParticleFieldParams:
    particle_type: str = ""
    field_name: str = ""


@dataclass(frozen=True)
class ParticleCicGridParams:
    particle_type: str = ""
    level_index: int = -1
    axis: int = 2
    axis_bounds: tuple[float, float] = (0.0, 0.0)
    mass_max: float = nan
    covered_boxes: CoveredBoxes = ()


@dataclass(frozen=True)
class ParticleCicProjectionParams:
    particle_type: str = ""
    level_index: int = -1
    axis: int = 2
    axis_bounds: tuple[float, float] = (0.0, 0.0)
    rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    resolution: tuple[int, int] = (1, 1)
    mass_max: float = nan
    covered_boxes: CoveredBoxes = ()


@dataclass(frozen=True)
class ScalarParams:
    scalar: float = 0.0


@dataclass(frozen=True)
class ValuesParams:
    values: tuple[float, ...] = ()


@dataclass(frozen=True)
class FiniteOnlyParams:
    finite_only: bool = True


@dataclass(frozen=True)
class ParticleHistogramParams:
    edges: tuple[float, ...] = ()
    density: bool = False


@dataclass(frozen=True)
class TopKModesParams:
    k: int = 0


@dataclass(frozen=True)
class SliceFinalizeParams:
    pixel_area: float = 1.0


KernelParams: TypeAlias = (
    NoKernelParams
    | AmrSubboxPackParams
    | GradStencilParams
    | PlotfileLoadParams
    | FluxSurfaceParams
    | CylindricalFluxParams
    | UniformSliceCellParams
    | UniformProjectionParams
    | FieldExprParams
    | UniformSliceParams
    | Histogram1DParams
    | Histogram2DParams
    | ParticleFieldParams
    | ParticleCicGridParams
    | ParticleCicProjectionParams
    | ScalarParams
    | ValuesParams
    | FiniteOnlyParams
    | ParticleHistogramParams
    | TopKModesParams
    | SliceFinalizeParams
)


_EXPECTED_PARAMS: dict[str, type[KernelParams]] = {
    "amr_subbox_fetch_pack": AmrSubboxPackParams,
    "gradU_stencil": GradStencilParams,
    "plotfile_load": PlotfileLoadParams,
    "flux_surface_integral_accumulate": FluxSurfaceParams,
    "cylindrical_flux_surface_integral_accumulate": CylindricalFluxParams,
    "uniform_slice_cellavg_accumulate": UniformSliceCellParams,
    "uniform_projection_accumulate": UniformProjectionParams,
    "field_expr": FieldExprParams,
    "uniform_slice": UniformSliceParams,
    "histogram1d_accumulate": Histogram1DParams,
    "histogram2d_accumulate": Histogram2DParams,
    "particle_load_field_chunk_f64": ParticleFieldParams,
    "particle_cic_grid_accumulate": ParticleCicGridParams,
    "particle_cic_projection_accumulate": ParticleCicProjectionParams,
    "particle_eq_mask": ScalarParams,
    "particle_abs_lt_mask": ScalarParams,
    "particle_le_mask": ScalarParams,
    "particle_gt_mask": ScalarParams,
    "particle_isin_mask": ValuesParams,
    "particle_min": FiniteOnlyParams,
    "particle_max": FiniteOnlyParams,
    "particle_histogram1d": ParticleHistogramParams,
    "particle_histogram1d_weighted": ParticleHistogramParams,
    "particle_topk_modes_map": ParticleFieldParams,
    "particle_topk_modes_finalize": TopKModesParams,
    "uniform_slice_finalize": SliceFinalizeParams,
}


def validate_kernel_params(kernel: str, params: KernelParams) -> None:
    expected = _EXPECTED_PARAMS.get(kernel, NoKernelParams)
    if not isinstance(params, expected):
        raise TypeError(
            f"kernel {kernel!r} requires {expected.__name__}, "
            f"got {type(params).__name__}"
        )


def covered_boxes(params: KernelParams) -> CoveredBoxes:
    return getattr(params, "covered_boxes", ())
