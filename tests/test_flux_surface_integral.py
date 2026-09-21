from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from analysis import Runtime
from analysis.buffer import FixedShape, numpy_dtype
from analysis.dataset import open_dataset
from analysis.kernel_params import NoKernelParams
from analysis.pipeline import Pipeline
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


def _output_bytes(template) -> int:
    spec = template.outputs[0].buffer
    assert isinstance(spec.shape, FixedShape)
    return int(np.prod(spec.shape.extents)) * numpy_dtype(spec.dtype).itemsize


class _FakeCoreRuntime:
    def __init__(self) -> None:
        self._next_field = 1000
        self.persistent: dict[int, str] = {}

    def alloc_field_id(self, name: str) -> int:
        fid = self._next_field
        self._next_field += 1
        return fid

    def mark_field_persistent(self, fid: int, name: str) -> None:
        self.persistent[fid] = name

    def num_localities(self) -> int:
        return 1


class _FakeRuntime:
    def __init__(self) -> None:
        self._rt = _FakeCoreRuntime()

    def alloc_field_id(self, name: str) -> int:
        return self._rt.alloc_field_id(name)


class _FakeDataset:
    def __init__(self, runtime: _FakeRuntime, *, step: int = 0, level: int = 0) -> None:
        self.runtime = runtime
        self.step = step
        self.level = level


@dataclass(frozen=True)
class _Box:
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]


@dataclass(frozen=True)
class _Geom:
    dx: tuple[float, float, float]
    x0: tuple[float, float, float]
    index_origin: tuple[int, int, int]
    ref_ratio: int = 1


@dataclass(frozen=True)
class _Level:
    geom: _Geom
    boxes: list[_Box]


@dataclass(frozen=True)
class _Step:
    step: int
    levels: list[_Level]


@dataclass(frozen=True)
class _RunMeta:
    steps: list[_Step]


def _set_block_double(
    ds,
    *,
    step: int,
    level: int,
    field: int,
    block: int,
    values: np.ndarray,
) -> None:
    arr = np.ascontiguousarray(values)
    dtype = "f32" if arr.dtype == np.float32 else "f64"
    ds._h.set_chunk_ref(
        step, level, field, 0, block, arr.tobytes(order="C"), dtype, list(arr.shape)
    )


def _runmeta_with_step_index(step: int, levels: list[LevelMeta]) -> RunMeta:
    return RunMeta(steps=[StepMeta(step=i, levels=levels) for i in range(step + 1)])


def _one_cell_state(*, rho: float, momx: float, energy: float, scalar: float, bz: float) -> dict[int, np.ndarray]:
    return {
        1: np.array([[[rho]]], dtype=np.float64),
        2: np.array([[[momx]]], dtype=np.float64),
        3: np.array([[[0.0]]], dtype=np.float64),
        4: np.array([[[0.0]]], dtype=np.float64),
        5: np.array([[[energy]]], dtype=np.float64),
        6: np.array([[[scalar]]], dtype=np.float64),
        7: np.array([[[0.0]]], dtype=np.float64),
        8: np.array([[[0.0]]], dtype=np.float64),
        9: np.array([[[bz]]], dtype=np.float64),
    }


def test_flux_surface_integral_lowering_wires_accumulate_reduce_and_covered_boxes() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, 0.0, 0.0),
                            index_origin=(0, 0, 0),
                            ref_ratio=2,
                        ),
                        boxes=[_Box((0, 0, 0), (1, 1, 1)), _Box((2, 0, 0), (3, 1, 1))],
                    ),
                    _Level(
                        geom=_Geom(
                            dx=(0.5, 0.5, 0.5),
                            x0=(0.0, 0.0, 0.0),
                            index_origin=(0, 0, 0),
                            ref_ratio=1,
                        ),
                        boxes=[_Box((0, 0, 0), (1, 3, 3))],
                    ),
                ],
            )
        ]
    )
    ds = _FakeDataset(rt)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=2.5,
        out="flux",
        reduce_fan_in=2,
    )
    plan = pipe.plan()

    assert flux.components == (
        "mass_flux_sphere_negative",
        "hydro_energy_flux_sphere_negative",
        "mhd_energy_flux_sphere_negative",
        "passive_scalar_flux_sphere_negative",
        "mass_flux_sphere_positive",
        "hydro_energy_flux_sphere_positive",
        "mhd_energy_flux_sphere_positive",
        "passive_scalar_flux_sphere_positive",
    )
    templates = [tmpl for stage in plan.stages for tmpl in stage.templates]
    accum_stages = [
        stage
        for stage in plan.stages
        if any(tmpl.kernel == "flux_surface_integral_accumulate" for tmpl in stage.templates)
    ]
    assert len(accum_stages) == 1
    assert accum_stages[0].after == []

    accum = [tmpl for tmpl in templates if tmpl.kernel == "flux_surface_integral_accumulate"]
    assert len(accum) == 3
    assert accum_stages[0].templates == accum
    assert all(len(tmpl.inputs) == 9 for tmpl in accum)
    assert all(_output_bytes(tmpl) == 64 for tmpl in accum)

    coarse = [tmpl for tmpl in accum if tmpl.domain.level == 0]
    fine = [tmpl for tmpl in accum if tmpl.domain.level == 1]
    assert coarse
    assert fine
    assert ((0, 0, 0), (0, 1, 1)) in coarse[0].params.covered_boxes
    assert fine[0].params.covered_boxes == ()

    reducers = [tmpl for tmpl in templates if tmpl.kernel == "uniform_slice_reduce"]
    assert reducers
    assert all(isinstance(tmpl.params, NoKernelParams) for tmpl in reducers)
    assert all(_output_bytes(tmpl) == 64 for tmpl in reducers)
    first_reduce_stages = [
        stage
        for stage in plan.stages
        if stage.plane == "graph"
        and accum_stages[0] in stage.after
        and any(tmpl.kernel == "uniform_slice_reduce" for tmpl in stage.templates)
    ]
    assert first_reduce_stages


def test_flux_surface_integral_lowering_accepts_radius_array() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, -0.5, -0.5),
                            index_origin=(0, 0, 0),
                        ),
                        boxes=[_Box((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=_FakeDataset(rt))

    flux = pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=np.array([0.25, 0.5, 0.75]),
        out="flux",
    )
    plan = pipe.plan()

    assert flux.radii == (0.25, 0.5, 0.75)
    templates = [tmpl for stage in plan.stages for tmpl in stage.templates]
    accum = [tmpl for tmpl in templates if tmpl.kernel == "flux_surface_integral_accumulate"]
    assert len(accum) == 1
    assert accum[0].params.radii == (0.25, 0.5, 0.75)
    assert _output_bytes(accum[0]) == 192
    reducers = [tmpl for tmpl in templates if tmpl.kernel == "uniform_slice_reduce"]
    assert reducers
    assert all(_output_bytes(tmpl) == 192 for tmpl in reducers)


def test_flux_surface_integral_lowering_accepts_temperature_bins() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, -0.5, -0.5),
                            index_origin=(0, 0, 0),
                        ),
                        boxes=[_Box((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=_FakeDataset(rt))

    flux = pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=np.array([0.5]),
        temperature=10,
        temperature_bins=np.array([1.0, 10.0, 100.0]),
        out="flux",
    )
    plan = pipe.plan()

    assert flux.temperature_bins == (1.0, 10.0, 100.0)
    templates = [tmpl for stage in plan.stages for tmpl in stage.templates]
    accum = [tmpl for tmpl in templates if tmpl.kernel == "flux_surface_integral_accumulate"]
    assert len(accum) == 1
    assert len(accum[0].inputs) == 10
    assert accum[0].params.temperature_bins == (1.0, 10.0, 100.0)
    assert _output_bytes(accum[0]) == 128
    reducers = [tmpl for tmpl in templates if tmpl.kernel == "uniform_slice_reduce"]
    assert reducers
    assert all(_output_bytes(tmpl) == 128 for tmpl in reducers)


def test_cylindrical_flux_surface_integral_lowering_accepts_height_array() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, -0.5, -1.5),
                            index_origin=(0, 0, 0),
                        ),
                        boxes=[_Box((0, 0, 0), (0, 0, 2))],
                    )
                ],
            )
        ]
    )
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=_FakeDataset(rt))

    flux = pipe.cylindrical_flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=0.5,
        height=np.array([0.5, 1.5]),
        out="flux",
    )
    plan = pipe.plan()

    assert flux.radius == 0.5
    assert flux.heights == (0.5, 1.5)
    assert flux.geometric_sections == ("endcaps", "walls")
    assert flux.components == (
        "mass_flux_cylinder_negative",
        "hydro_energy_flux_cylinder_negative",
        "mhd_energy_flux_cylinder_negative",
        "passive_scalar_flux_cylinder_negative",
        "advective_radial_angular_momentum_flux_cylinder_negative",
        "maxwell_radial_angular_momentum_flux_cylinder_negative",
        "advective_vertical_angular_momentum_flux_cylinder_negative",
        "maxwell_vertical_angular_momentum_flux_cylinder_negative",
        "mass_flux_cylinder_positive",
        "hydro_energy_flux_cylinder_positive",
        "mhd_energy_flux_cylinder_positive",
        "passive_scalar_flux_cylinder_positive",
        "advective_radial_angular_momentum_flux_cylinder_positive",
        "maxwell_radial_angular_momentum_flux_cylinder_positive",
        "advective_vertical_angular_momentum_flux_cylinder_positive",
        "maxwell_vertical_angular_momentum_flux_cylinder_positive",
    )
    templates = [tmpl for stage in plan.stages for tmpl in stage.templates]
    accum = [
        tmpl
        for tmpl in templates
        if tmpl.kernel == "cylindrical_flux_surface_integral_accumulate"
    ]
    assert len(accum) == 1
    assert accum[0].params.radius == 0.5
    assert accum[0].params.heights == (0.5, 1.5)
    assert _output_bytes(accum[0]) == 512
    reducers = [tmpl for tmpl in templates if tmpl.kernel == "uniform_slice_reduce"]
    assert reducers
    assert all(_output_bytes(tmpl) == 512 for tmpl in reducers)


def test_flux_surface_integral_rejects_temperature_bins_without_temperature_field() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, -0.5, -0.5),
                            index_origin=(0, 0, 0),
                        ),
                        boxes=[_Box((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=_FakeDataset(rt))

    with pytest.raises(ValueError, match="temperature must be provided"):
        pipe.flux_surface_integral(
            1,
            momentum=(2, 3, 4),
            energy=5,
            passive_scalar=6,
            magnetic_field=(7, 8, 9),
            radius=0.5,
            temperature_bins=[1.0, 10.0],
        )


def test_flux_surface_integral_lowering_uses_per_block_radius_subsets() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, -0.5, -0.5),
                            index_origin=(0, 0, 0),
                        ),
                        boxes=[
                            _Box((0, 0, 0), (0, 0, 0)),
                            _Box((3, 0, 0), (3, 0, 0)),
                        ],
                    )
                ],
            )
        ]
    )
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=_FakeDataset(rt))

    pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=[0.5, 3.5],
        out="flux",
    )
    plan = pipe.plan()

    templates = [tmpl for stage in plan.stages for tmpl in stage.templates]
    accum = [tmpl for tmpl in templates if tmpl.kernel == "flux_surface_integral_accumulate"]
    assert len(accum) == 2
    by_block = {tmpl.domain.blocks[0]: tmpl for tmpl in accum}
    assert by_block[0].params.radii == (0.5,)
    assert by_block[0].params.radius_indices == (0,)
    assert by_block[1].params.radii == (3.5,)
    assert by_block[1].params.radius_indices == (1,)
    assert all(tmpl.params.num_radii == 2 for tmpl in accum)
    assert all(_output_bytes(tmpl) == 128 for tmpl in accum)


def test_flux_surface_integral_lowering_normalizes_single_nonzero_block() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, -0.5, -0.5),
                            index_origin=(0, 0, 0),
                        ),
                        boxes=[
                            _Box((2, 0, 0), (2, 0, 0)),
                            _Box((0, 0, 0), (0, 0, 0)),
                        ],
                    )
                ],
            )
        ]
    )
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=_FakeDataset(rt))

    pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=0.5,
        out="flux",
    )
    plan = pipe.plan()

    templates = [tmpl for stage in plan.stages for tmpl in stage.templates]
    accum = [tmpl for tmpl in templates if tmpl.kernel == "flux_surface_integral_accumulate"]
    assert len(accum) == 1
    assert accum[0].domain.blocks == [1]

    reductions = [
        tmpl
        for tmpl in templates
        if tmpl.kernel == "uniform_slice_reduce"
        and tmpl.name == "flux_surface_integral_reduce_single"
    ]
    assert len(reductions) == 1
    assert reductions[0].graph_reduce.input_blocks == (1,)
    assert reductions[0].graph_reduce.output_blocks == (0,)
    assert reductions[0].graph_reduce.group_offsets == (0, 1)


def test_flux_surface_integral_rejects_invalid_radius() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, 0.0, 0.0),
                            index_origin=(0, 0, 0),
                        ),
                        boxes=[_Box((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=_FakeDataset(rt))

    with pytest.raises(ValueError, match="radius"):
        pipe.flux_surface_integral(
            1,
            momentum=(2, 3, 4),
            energy=5,
            passive_scalar=6,
            magnetic_field=(7, 8, 9),
            radius=0.0,
        )


def test_flux_surface_integral_rejects_invalid_radius_array_value() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, 0.0, 0.0),
                            index_origin=(0, 0, 0),
                        ),
                        boxes=[_Box((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=_FakeDataset(rt))

    with pytest.raises(ValueError, match="radius"):
        pipe.flux_surface_integral(
            1,
            momentum=(2, 3, 4),
            energy=5,
            passive_scalar=6,
            magnetic_field=(7, 8, 9),
            radius=[0.5, 0.0],
        )


def test_flux_surface_integral_rejects_radius_with_no_intersecting_blocks() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, 0.0, 0.0),
                            index_origin=(0, 0, 0),
                        ),
                        boxes=[_Box((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=_FakeDataset(rt))

    with pytest.raises(ValueError, match="does not intersect any mesh block"):
        pipe.flux_surface_integral(
            1,
            momentum=(2, 3, 4),
            energy=5,
            passive_scalar=6,
            magnetic_field=(7, 8, 9),
            radius=10.0,
        )


def test_flux_surface_integral_rejects_radius_array_with_missing_intersection() -> None:
    rt = _FakeRuntime()
    runmeta = _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, -0.5, -0.5),
                            index_origin=(0, 0, 0),
                        ),
                        boxes=[_Box((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=_FakeDataset(rt))

    with pytest.raises(ValueError, match="do not intersect any mesh block"):
        pipe.flux_surface_integral(
            1,
            momentum=(2, 3, 4),
            energy=5,
            passive_scalar=6,
            magnetic_field=(7, 8, 9),
            radius=[0.5, 10.0],
        )


def test_flux_surface_integral_runtime_one_cell_mhd_energy_term() -> None:
    rt = Runtime()
    step = 4
    levels = [
        LevelMeta(
            geom=LevelGeom(
                dx=(1.0, 1.0, 1.0),
                x0=(0.0, -0.5, -0.5),
                ref_ratio=1,
            ),
            boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
        )
    ]
    runmeta = _runmeta_with_step_index(step, levels)
    ds = open_dataset("memory://flux-one-cell", runmeta=runmeta, step=step, level=0, runtime=rt)
    for name, fid in {
        "rho": 1,
        "momx": 2,
        "momy": 3,
        "momz": 4,
        "energy": 5,
        "scalar": 6,
        "bx": 7,
        "by": 8,
        "bz": 9,
    }.items():
        ds.register_field(name, fid)

    # Cell edges are x=[0,1], y=[-0.5,0.5], z=[-0.5,0.5].
    # At R=0.5 the Quokka tangent-plane section area is exactly 1.
    for fid, values in _one_cell_state(rho=2.0, momx=6.0, energy=21.5, scalar=5.0, bz=1.0).items():
        _set_block_double(ds, step=step, level=0, field=fid, block=0, values=values)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.flux_surface_integral(
        pipe.field(1),
        momentum=(pipe.field(2), pipe.field(3), pipe.field(4)),
        energy=pipe.field(5),
        passive_scalar=pipe.field(6),
        magnetic_field=(pipe.field(7), pipe.field(8), pipe.field(9)),
        radius=0.5,
        out="flux",
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=flux.field,
        shape=(2, 4),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    assert np.allclose(raw, np.array([[0.0, 0.0, 0.0, 0.0], [6.0, 87.0, 90.0, 15.0]]))


def test_cylindrical_flux_surface_integral_runtime_outputs_endcaps_and_walls() -> None:
    rt = Runtime()
    step = 19
    levels = [
        LevelMeta(
            geom=LevelGeom(
                dx=(1.0, 1.0, 1.0),
                x0=(0.0, -0.5, -0.5),
                ref_ratio=1,
            ),
            boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
        )
    ]
    runmeta = _runmeta_with_step_index(step, levels)
    ds = open_dataset(
        "memory://cylflux-endcaps-walls",
        runmeta=runmeta,
        step=step,
        level=0,
        runtime=rt,
    )
    for fid in range(1, 10):
        ds.register_field(f"f{fid}", fid)

    state = _one_cell_state(rho=2.0, momx=0.0, energy=21.5, scalar=5.0, bz=0.0)
    state[4] = np.array([[[4.0]]], dtype=np.float64)
    for fid, values in state.items():
        _set_block_double(ds, step=step, level=0, field=fid, block=0, values=values)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.cylindrical_flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=0.5,
        height=0.25,
        out="flux",
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=flux.field,
        shape=(2, 2, 8),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    expected = np.zeros((2, 2, 8), dtype=np.float64)
    expected[0, 0, :4] = np.array([-4.0, -(199.0 / 3.0), -(199.0 / 3.0), -10.0])
    expected[1, 0, :4] = np.array([4.0, 199.0 / 3.0, 199.0 / 3.0, 10.0])
    expected *= np.pi / 8.0  # cap footprint is a half disk of radius 0.5
    assert np.allclose(raw, expected)


def test_flux_surface_integral_runtime_radius_array() -> None:
    rt = Runtime()
    step = 8
    levels = [
        LevelMeta(
            geom=LevelGeom(
                dx=(1.0, 1.0, 1.0),
                x0=(0.0, -0.5, -0.5),
                ref_ratio=1,
            ),
            boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
        )
    ]
    runmeta = _runmeta_with_step_index(step, levels)
    ds = open_dataset("memory://flux-radius-array", runmeta=runmeta, step=step, level=0, runtime=rt)
    for fid in range(1, 10):
        ds.register_field(f"f{fid}", fid)

    for fid, values in _one_cell_state(rho=2.0, momx=6.0, energy=21.5, scalar=5.0, bz=1.0).items():
        _set_block_double(ds, step=step, level=0, field=fid, block=0, values=values)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=[0.25, 0.5, 0.75],
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=flux.field,
        shape=(3, 2, 4),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    expected = np.tile(
        np.array([[0.0, 0.0, 0.0, 0.0], [6.0, 87.0, 90.0, 15.0]]),
        (3, 1, 1),
    )
    assert np.allclose(raw, expected)


def test_cylindrical_flux_surface_integral_runtime_one_cell_mhd_energy_term() -> None:
    rt = Runtime()
    step = 18
    levels = [
        LevelMeta(
            geom=LevelGeom(
                dx=(1.0, 1.0, 1.0),
                x0=(0.0, -0.5, -0.5),
                ref_ratio=1,
            ),
            boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
        )
    ]
    runmeta = _runmeta_with_step_index(step, levels)
    ds = open_dataset("memory://cylflux-one-cell", runmeta=runmeta, step=step, level=0, runtime=rt)
    for fid in range(1, 10):
        ds.register_field(f"f{fid}", fid)

    for fid, values in _one_cell_state(rho=2.0, momx=6.0, energy=21.5, scalar=5.0, bz=1.0).items():
        _set_block_double(ds, step=step, level=0, field=fid, block=0, values=values)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.cylindrical_flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=0.5,
        height=0.5,
        out="flux",
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=flux.field,
        shape=(2, 2, 8),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    expected = np.zeros((2, 2, 8), dtype=np.float64)
    expected[1, 1, :4] = np.array([6.0, 87.0, 90.0, 15.0])
    assert np.allclose(raw, expected)


def test_flux_surface_integral_runtime_sparse_radius_slots() -> None:
    def run_flux(radii: list[float], *, uri_suffix: str) -> np.ndarray:
        rt = Runtime()
        step = 9
        levels = [
            LevelMeta(
                geom=LevelGeom(
                    dx=(1.0, 1.0, 1.0),
                    x0=(0.0, -0.5, -0.5),
                    ref_ratio=1,
                ),
                boxes=[
                    BlockBox((0, 0, 0), (0, 0, 0)),
                    BlockBox((3, 0, 0), (3, 0, 0)),
                ],
            )
        ]
        runmeta = _runmeta_with_step_index(step, levels)
        ds = open_dataset(
            f"memory://flux-sparse-radius-slots-{uri_suffix}",
            runmeta=runmeta,
            step=step,
            level=0,
            runtime=rt,
        )
        for fid in range(1, 10):
            ds.register_field(f"f{fid}", fid)

        state = _one_cell_state(rho=2.0, momx=6.0, energy=21.5, scalar=5.0, bz=1.0)
        for block in (0, 1):
            for fid, values in state.items():
                _set_block_double(ds, step=step, level=0, field=fid, block=block, values=values)

        pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        flux = pipe.flux_surface_integral(
            1,
            momentum=(2, 3, 4),
            energy=5,
            passive_scalar=6,
            magnetic_field=(7, 8, 9),
            radius=radii,
        )
        pipe.run()
        return rt.get_task_chunk_array(
            step=step,
            level=0,
            field=flux.field,
                dtype=np.float64,
            dataset=ds,
            block=0,
        )

    combined = run_flux([0.5, 3.5], uri_suffix="combined")
    first = run_flux([0.5], uri_suffix="first")
    second = run_flux([3.5], uri_suffix="second")

    assert np.allclose(combined[0], first)
    assert np.allclose(combined[1], second)


def test_flux_surface_integral_runtime_single_nonzero_block() -> None:
    rt = Runtime()
    step = 7
    levels = [
        LevelMeta(
            geom=LevelGeom(
                dx=(1.0, 1.0, 1.0),
                x0=(0.0, -0.5, -0.5),
                ref_ratio=1,
            ),
            boxes=[
                BlockBox((2, 0, 0), (2, 0, 0)),
                BlockBox((0, 0, 0), (0, 0, 0)),
            ],
        )
    ]
    runmeta = _runmeta_with_step_index(step, levels)
    ds = open_dataset(
        "memory://flux-single-nonzero-block",
        runmeta=runmeta,
        step=step,
        level=0,
        runtime=rt,
    )
    for fid in range(1, 10):
        ds.register_field(f"f{fid}", fid)

    for fid, values in _one_cell_state(rho=2.0, momx=6.0, energy=21.5, scalar=5.0, bz=1.0).items():
        _set_block_double(ds, step=step, level=0, field=fid, block=1, values=values)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=0.5,
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=flux.field,
        shape=(2, 4),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    assert np.allclose(raw, np.array([[0.0, 0.0, 0.0, 0.0], [6.0, 87.0, 90.0, 15.0]]))


def test_flux_surface_integral_runtime_multiblock_reduction() -> None:
    rt = Runtime()
    step = 5
    levels = [
        LevelMeta(
            geom=LevelGeom(
                dx=(1.0, 1.0, 1.0),
                x0=(-1.0, -0.5, -0.5),
                ref_ratio=1,
            ),
            boxes=[
                BlockBox((0, 0, 0), (0, 0, 0)),
                BlockBox((1, 0, 0), (1, 0, 0)),
            ],
        )
    ]
    runmeta = _runmeta_with_step_index(step, levels)
    ds = open_dataset("memory://flux-two-block", runmeta=runmeta, step=step, level=0, runtime=rt)
    for fid in range(1, 10):
        ds.register_field(f"f{fid}", fid)

    left = _one_cell_state(rho=2.0, momx=-6.0, energy=21.0, scalar=5.0, bz=0.0)
    right = _one_cell_state(rho=2.0, momx=6.0, energy=21.0, scalar=5.0, bz=0.0)
    for fid, values in left.items():
        _set_block_double(ds, step=step, level=0, field=fid, block=0, values=values)
    for fid, values in right.items():
        _set_block_double(ds, step=step, level=0, field=fid, block=1, values=values)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=0.5,
        reduce_fan_in=2,
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=flux.field,
        shape=(2, 4),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    assert np.allclose(raw, np.array([[0.0, 0.0, 0.0, 0.0], [12.0, 174.0, 174.0, 30.0]]))


def test_flux_surface_integral_runtime_outputs_negative_and_positive_bins() -> None:
    rt = Runtime()
    step = 10
    levels = [
        LevelMeta(
            geom=LevelGeom(
                dx=(1.0, 1.0, 1.0),
                x0=(-1.0, -0.5, -0.5),
                ref_ratio=1,
            ),
            boxes=[
                BlockBox((0, 0, 0), (0, 0, 0)),
                BlockBox((1, 0, 0), (1, 0, 0)),
            ],
        )
    ]
    runmeta = _runmeta_with_step_index(step, levels)
    ds = open_dataset("memory://flux-sign-bins", runmeta=runmeta, step=step, level=0, runtime=rt)
    for fid in range(1, 10):
        ds.register_field(f"f{fid}", fid)

    state = _one_cell_state(rho=2.0, momx=6.0, energy=21.0, scalar=5.0, bz=0.0)
    for block in (0, 1):
        for fid, values in state.items():
            _set_block_double(ds, step=step, level=0, field=fid, block=block, values=values)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=0.5,
        reduce_fan_in=2,
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=flux.field,
        shape=(2, 4),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )

    assert np.allclose(raw, np.array([[-6.0, -87.0, -87.0, -15.0], [6.0, 87.0, 87.0, 15.0]]))
    assert np.allclose(raw.sum(axis=0), np.zeros(4))


def test_flux_surface_integral_runtime_outputs_temperature_bins() -> None:
    rt = Runtime()
    step = 10
    levels = [
        LevelMeta(
            geom=LevelGeom(
                dx=(1.0, 1.0, 1.0),
                x0=(-1.0, -0.5, -0.5),
                ref_ratio=1,
            ),
            boxes=[
                BlockBox((0, 0, 0), (0, 0, 0)),
                BlockBox((1, 0, 0), (1, 0, 0)),
            ],
        )
    ]
    runmeta = _runmeta_with_step_index(step, levels)
    ds = open_dataset("memory://flux-temperature-bins", runmeta=runmeta, step=step, level=0, runtime=rt)
    for fid in range(1, 11):
        ds.register_field(f"f{fid}", fid)

    state = _one_cell_state(rho=2.0, momx=6.0, energy=21.0, scalar=5.0, bz=0.0)
    for block in (0, 1):
        for fid, values in state.items():
            _set_block_double(ds, step=step, level=0, field=fid, block=block, values=values)
    _set_block_double(
        ds,
        step=step,
        level=0,
        field=10,
        block=0,
        values=np.array([[[5.0]]], dtype=np.float64),
    )
    _set_block_double(
        ds,
        step=step,
        level=0,
        field=10,
        block=1,
        values=np.array([[[15.0]]], dtype=np.float64),
    )

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=0.5,
        temperature=10,
        temperature_bins=[0.0, 10.0, 20.0],
        reduce_fan_in=2,
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=flux.field,
        shape=(2, 2, 4),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )

    assert np.allclose(raw[0, 0], np.array([-6.0, -87.0, -87.0, -15.0]))
    assert np.allclose(raw[0, 1], np.zeros(4))
    assert np.allclose(raw[1, 0], np.zeros(4))
    assert np.allclose(raw[1, 1], np.array([6.0, 87.0, 87.0, 15.0]))
    assert np.allclose(raw.sum(axis=(0, 1)), np.zeros(4))


def test_flux_surface_integral_runtime_amr_covered_cells_are_excluded() -> None:
    rt = Runtime()
    step = 6
    levels = [
        LevelMeta(
            geom=LevelGeom(
                dx=(1.0, 1.0, 1.0),
                x0=(0.0, -0.5, -0.5),
                ref_ratio=2,
            ),
            boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
        ),
        LevelMeta(
            geom=LevelGeom(
                dx=(0.5, 0.5, 0.5),
                x0=(0.0, -0.5, -0.5),
                ref_ratio=1,
            ),
            boxes=[BlockBox((0, 0, 0), (1, 1, 1))],
        ),
    ]
    runmeta = _runmeta_with_step_index(step, levels)
    ds = open_dataset("memory://flux-amr-mask", runmeta=runmeta, step=step, level=0, runtime=rt)
    for fid in range(1, 10):
        ds.register_field(f"f{fid}", fid)

    coarse = _one_cell_state(rho=2.0, momx=6.0, energy=21.0, scalar=5.0, bz=0.0)
    for fid, values in coarse.items():
        _set_block_double(ds, step=step, level=0, field=fid, block=0, values=values)

    fine_zero = np.zeros((2, 2, 2), dtype=np.float64)
    for fid in range(1, 10):
        _set_block_double(ds, step=step, level=1, field=fid, block=0, values=fine_zero)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.flux_surface_integral(
        1,
        momentum=(2, 3, 4),
        energy=5,
        passive_scalar=6,
        magnetic_field=(7, 8, 9),
        radius=0.5,
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=flux.field,
        shape=(2, 4),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    assert np.allclose(raw, np.zeros((2, 4)))


@pytest.mark.parametrize("momx,momy", [(6.0, 4.0), (-6.0, 4.0), (6.0, -4.0), (0.0, 4.0)])
def test_cylindrical_angular_momentum_uses_surface_lever_arm_and_flux_sign(
    momx: float, momy: float,
) -> None:
    rt = Runtime()
    levels = [LevelMeta(
        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, -0.5, -0.5), ref_ratio=1),
        boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
    )]
    runmeta = _runmeta_with_step_index(0, levels)
    ds = open_dataset("memory://cyl-angular-cell", runmeta=runmeta,
                      step=0, level=0, runtime=rt)
    state = _one_cell_state(rho=2.0, momx=momx, energy=100.0, scalar=1.0, bz=5.0)
    state[3][...] = momy
    state[7][...] = 2.0
    state[8][...] = 3.0
    state[10] = np.full((1, 1, 1), 15.0)
    for fid, values in state.items():
        ds.register_field(f"f{fid}", fid)
        _set_block_double(ds, step=0, level=0, field=fid, block=0, values=values)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.cylindrical_flux_surface_integral(
        1, momentum=(2, 3, 4), energy=5, passive_scalar=6,
        magnetic_field=(7, 8, 9), radius=0.75, height=[0.25, 0.4],
        temperature=10, temperature_bins=[0.0, 10.0, 20.0],
    )
    pipe.run()
    result = rt.get_task_chunk_array(
        step=0, level=0, field=flux.field, block=0, dtype=np.float64, dataset=ds,
    ).reshape(2, 2, 2, 2, 8)
    assert np.all(result[:, :, 0] == 0.0)
    for height_idx, height in enumerate((0.25, 0.4)):
        area = 2.0 * height  # tangent-plane wall strip in this unit-width cell
        advective = 0.75 * momy * momx / 2.0 * area
        maxwell = -0.75 * 3.0 * 2.0 * area
        expected = np.zeros((2, 2))
        expected[0 if advective < 0.0 else 1, 0] = advective
        expected[0, 1] = maxwell
        np.testing.assert_allclose(result[height_idx, :, 1, 1, 4:6], expected)
        # Integral x dA over x>=0, |y|<0.5 in the radius-0.75 disk.
        cap_moment_x = 23.0 / 96.0
        np.testing.assert_allclose(result[height_idx, :, 1, 0, 6], 0.0)
        np.testing.assert_allclose(
            result[height_idx, :, 1, 0, 7],
            [-15.0 * cap_moment_x, 15.0 * cap_moment_x],
        )
        np.testing.assert_allclose(result[height_idx, :, 1, 1, 6:8], 0.0)
        # Radial transport excludes both caps, even with nonzero Bz.
        np.testing.assert_allclose(result[height_idx, :, 1, 0, 4], 0.0)
        np.testing.assert_allclose(result[height_idx, :, 1, 0, 5], 0.0)


def _cylindrical_analytic_profile(
    n: int, *, refined: bool = False, poison_covered: bool = False,
    radial_velocity: float = 0.3, magnetic: bool = True,
) -> np.ndarray:
    """Sample constant cylindrical components on a Cartesian AMR hierarchy."""
    rt = Runtime()
    dx = 3.0 / n
    levels = [LevelMeta(
        geom=LevelGeom(dx=(dx, dx, 0.375), x0=(-1.5, -1.5, -0.75),
                       ref_ratio=2 if refined else 1),
        boxes=[BlockBox((0, 0, 0), (n // 2 - 1, n - 1, 3)),
               BlockBox((n // 2, 0, 0), (n - 1, n - 1, 3))],
    )]
    if refined:
        levels.append(LevelMeta(
            geom=LevelGeom(dx=(dx / 2, dx / 2, 0.1875),
                           x0=(-1.5, -1.5, -0.75), ref_ratio=1),
            boxes=[BlockBox((n, 0, 0), (2 * n - 1, 2 * n - 1, 7))],
        ))
    runmeta = _runmeta_with_step_index(0, levels)
    ds = open_dataset("memory://cyl-angular-profile", runmeta=runmeta,
                      step=0, level=0, runtime=rt)
    for fid in range(1, 10):
        ds.register_field(f"f{fid}", fid)
    for lev, meta in enumerate(levels):
        for block, box in enumerate(meta.boxes):
            axes = [meta.geom.x0[a] +
                    (np.arange(box.lo[a], box.hi[a] + 1) + 0.5) * meta.geom.dx[a]
                    for a in range(3)]
            x, y, z = np.meshgrid(*axes, indexing="ij")
            radius = np.hypot(x, y)
            nx, ny = x / radius, y / radius
            rho = np.full_like(x, 2.0)
            if poison_covered and lev == 0:
                rho[x > 0] *= 100.0
            vx = radial_velocity * nx - 1.7 * ny
            vy = radial_velocity * ny + 1.7 * nx
            bx = (0.4 * nx - 0.6 * ny) if magnetic else np.zeros_like(x)
            by = (0.4 * ny + 0.6 * nx) if magnetic else np.zeros_like(x)
            state = [rho, rho * vx, rho * vy, np.zeros_like(z),
                     10.0 + 0.5 * rho * (vx**2 + vy**2) + 0.5 * (bx**2 + by**2),
                     np.ones_like(x), bx, by, np.zeros_like(z)]
            for fid, values in enumerate(state, 1):
                _set_block_double(ds, step=0, level=lev, field=fid, block=block, values=values)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.cylindrical_flux_surface_integral(
        1, momentum=(2, 3, 4), energy=5, passive_scalar=6,
        magnetic_field=(7, 8, 9), radius=0.8, height=0.37,
    )
    pipe.run()
    result = rt.get_task_chunk_array(
        step=0, level=0, field=flux.field, block=0, dtype=np.float64, dataset=ds,
    ).reshape(2, 2, 8)
    return result[:, 1].sum(axis=0)


def test_cylindrical_angular_momentum_surface_converges() -> None:
    expected = 4.0 * np.pi * 0.37 * 0.8**2 * np.array([2.0 * 0.3 * 1.7, -0.4 * 0.6])
    coarse = _cylindrical_analytic_profile(16)[4:6]
    fine = _cylindrical_analytic_profile(64)[4:6]
    np.testing.assert_allclose(fine, expected, rtol=0.01)
    assert np.linalg.norm(fine - expected) < np.linalg.norm(coarse - expected)


def test_cylindrical_angular_momentum_pure_rotation() -> None:
    values = _cylindrical_analytic_profile(24, radial_velocity=0.0, magnetic=False)
    np.testing.assert_allclose(values[[0, 4, 5]], 0.0, atol=1.0e-12)


def test_cylindrical_angular_momentum_masks_covered_coarse_cells() -> None:
    clean = _cylindrical_analytic_profile(32, refined=True)
    poisoned = _cylindrical_analytic_profile(32, refined=True, poison_covered=True)
    np.testing.assert_allclose(poisoned, clean, rtol=1.0e-12, atol=1.0e-12)
    expected = 4.0 * np.pi * 0.37 * 0.8**2 * np.array([2.0 * 0.3 * 1.7, -0.4 * 0.6])
    np.testing.assert_allclose(clean[4:6], expected, rtol=0.01)


@pytest.mark.parametrize(
    "x0,y0,dx,dy,radius,area,x_moment,y_moment",
    [
        (0.0, 0.0, 1.0, 1.0, 0.5, np.pi / 16.0, 1.0 / 24.0, 1.0 / 24.0),
        (-1.0, 0.0, 1.0, 1.0, 0.5, np.pi / 16.0, -1.0 / 24.0, 1.0 / 24.0),
        (0.0, -1.0, 1.0, 1.0, 0.5, np.pi / 16.0, 1.0 / 24.0, -1.0 / 24.0),
        (-1.0, -1.0, 2.0, 2.0, 0.5, np.pi / 4.0, 0.0, 0.0),
        # Entire block is inside the cylinder: only the endcaps intersect it.
        (0.1, 0.2, 0.2, 0.2, 1.0, 0.04, 0.008, 0.012),
    ],
)
def test_cylindrical_endcap_disk_area_and_lever_moments(
    x0, y0, dx, dy, radius, area, x_moment, y_moment,
) -> None:
    rt = Runtime()
    levels = [LevelMeta(
        geom=LevelGeom(dx=(dx, dy, 1.0), x0=(x0, y0, -0.5), ref_ratio=1),
        boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
    )]
    runmeta = _runmeta_with_step_index(0, levels)
    ds = open_dataset("memory://cap-moments", runmeta=runmeta, runtime=rt, step=0, level=0)
    state = _one_cell_state(rho=2.0, momx=1.0, energy=100.0, scalar=1.0, bz=5.0)
    for fid, value in ((3, 3.0), (4, 4.0), (7, 1.0), (8, 2.0)):
        state[fid][...] = value
    for fid, values in state.items():
        ds.register_field(f"f{fid}", fid)
        _set_block_double(ds, step=0, level=0, field=fid, block=0, values=values)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.cylindrical_flux_surface_integral(
        1, momentum=(2, 3, 4), energy=5, passive_scalar=6,
        magnetic_field=(7, 8, 9), radius=radius, height=0.25,
    )
    pipe.run()
    result = rt.get_task_chunk_array(
        step=0, level=0, field=flux.field, block=0, dtype=np.float64, dataset=ds,
    ).reshape(2, 2, 8)
    np.testing.assert_allclose(result[:, 0, 0], [-4.0 * area, 4.0 * area], atol=1.e-14)
    advective = 2.0 * (3.0 * x_moment - y_moment)
    maxwell = -5.0 * (2.0 * x_moment - y_moment)
    np.testing.assert_allclose(result[:, 0, 6], [-abs(advective), abs(advective)], atol=1.e-14)
    np.testing.assert_allclose(result[:, 0, 7], [-abs(maxwell), abs(maxwell)], atol=1.e-14)


def _vertical_analytic_profile(
    n: int, *, refined: bool = False, poison_covered: bool = False,
    bipolar: bool = True, height: float = 0.375,
) -> np.ndarray:
    """Rigid rotation and a vertical flow on a tiled mesh with optional AMR."""
    rt = Runtime()
    dx = 3.0 / n
    tile = n // 8
    levels = [LevelMeta(
        geom=LevelGeom(dx=(dx, dx, 0.1875), x0=(-1.5, -1.5, -0.75),
                       ref_ratio=2 if refined else 1),
        boxes=[BlockBox((i, j, k), (i + tile - 1, j + tile - 1, k + 3))
               for i in range(0, n, tile) for j in range(0, n, tile) for k in (0, 4)],
    )]
    if refined:
        levels.append(LevelMeta(
            geom=LevelGeom(dx=(dx / 2, dx / 2, 0.09375),
                           x0=(-1.5, -1.5, -0.75), ref_ratio=1),
            boxes=[BlockBox((n, 0, 0), (2*n - 1, 2*n - 1, 15))],
        ))
    runmeta = _runmeta_with_step_index(0, levels)
    ds = open_dataset("memory://cap-profile", runmeta=runmeta, runtime=rt, step=0, level=0)
    for fid in range(1, 10):
        ds.register_field(f"f{fid}", fid)
    for lev, meta in enumerate(levels):
        for block, box in enumerate(meta.boxes):
            axes = [meta.geom.x0[a] +
                    (np.arange(box.lo[a], box.hi[a] + 1) + 0.5) * meta.geom.dx[a]
                    for a in range(3)]
            x, y, z = np.meshgrid(*axes, indexing="ij")
            rho = np.full_like(x, 2.0)
            if poison_covered and lev == 0:
                rho[x > 0] *= 100.0
            sign = np.sign(z) if bipolar else np.ones_like(z)
            vx, vy, vz = -1.7*y, 1.7*x, 0.3*sign
            bx, by, bz = -0.6*y*sign, 0.6*x*sign, np.full_like(z, 0.4)
            state = [rho, rho*vx, rho*vy, rho*vz,
                     10.0 + 0.5*rho*(vx**2 + vy**2 + vz**2) + 0.5*(bx**2 + by**2 + bz**2),
                     np.ones_like(x), bx, by, bz]
            for fid, values in enumerate(state, 1):
                _set_block_double(ds, step=0, level=lev, field=fid, block=block, values=values)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    flux = pipe.cylindrical_flux_surface_integral(
        1, momentum=(2, 3, 4), energy=5, passive_scalar=6,
        magnetic_field=(7, 8, 9), radius=0.8, height=height,
    )
    pipe.run()
    return rt.get_task_chunk_array(
        step=0, level=0, field=flux.field, block=0, dtype=np.float64, dataset=ds,
    ).reshape(2, 2, 8)


def test_cylindrical_vertical_flux_converges_and_counts_grid_aligned_caps_once() -> None:
    expected = np.pi * 0.8**4 * np.array([2.0 * 1.7 * 0.3, -0.6 * 0.4])
    coarse = _vertical_analytic_profile(16)[:, 0, 6:8].sum(axis=0)
    fine = _vertical_analytic_profile(64)
    net = fine[:, 0, 6:8].sum(axis=0)
    np.testing.assert_allclose(net, expected, rtol=0.005)
    assert np.linalg.norm(net - expected) < np.linalg.norm(coarse - expected)
    np.testing.assert_allclose(fine[:, 0, 0].sum(), 2.0 * np.pi * 0.8**2 * 2.0 * 0.3,
                               rtol=1.e-12)
    np.testing.assert_allclose(fine[:, 0, 4:6], 0.0, atol=1.e-14)
    np.testing.assert_allclose(fine[:, 1, 6:8], 0.0, atol=1.e-14)
    assert fine[1, 0, 6] > 0.0  # both caps export prograde angular momentum
    assert fine[0, 0, 7] < 0.0  # magnetic stress transports it in the opposite sense


@pytest.mark.parametrize("height", [0.375, 0.75, 0.37])
def test_cylindrical_vertical_flux_lower_normal_cancels_uniform_throughflow(height) -> None:
    result = _vertical_analytic_profile(32, bipolar=False, height=height)
    mass_flux = np.pi * 0.8**2 * 2.0 * 0.3
    np.testing.assert_allclose(result[:, 0, 0], [-mass_flux, mass_flux], rtol=1.e-12)
    assert result[0, 0, 6] < 0.0
    assert result[1, 0, 6] > 0.0
    np.testing.assert_allclose(result[:, 0, 6:8].sum(axis=0), 0.0, atol=1.e-12)


def test_cylindrical_vertical_flux_masks_partial_amr_coverage() -> None:
    clean = _vertical_analytic_profile(32, refined=True)
    poisoned = _vertical_analytic_profile(32, refined=True, poison_covered=True)
    np.testing.assert_allclose(poisoned, clean, rtol=1.e-12, atol=1.e-12)
    expected = np.pi * 0.8**4 * np.array([2.0 * 1.7 * 0.3, -0.6 * 0.4])
    np.testing.assert_allclose(clean[:, 0, 6:8].sum(axis=0), expected, rtol=0.01)


@pytest.mark.parametrize("axis", [0, 1], ids=["x", "y"])
@pytest.mark.parametrize("side", [-1, 1], ids=["negative", "positive"])
@pytest.mark.parametrize(
    "layout,offset",
    [(layout, offset) for layout in ("single", "split")
     for offset in (-1.e-6, 0.0, 1.e-13, 1.e-6)] + [("amr", 0.0)],
)
def test_cylindrical_wall_grid_face_has_single_owner(axis, side, layout, offset):
    """A coincident tangent patch uses only the cylinder-interior cell."""
    rt = Runtime()
    origin = [-0.5, -0.5, -0.5]
    origin[axis] = 0.0 if side > 0 else -2.0
    hi = [0, 0, 0]
    hi[axis] = 1
    boxes = [BlockBox((0, 0, 0), tuple(hi))]
    if layout == "split":
        boxes = []
        for i in range(2):
            index = [0, 0, 0]
            index[axis] = i
            boxes.append(BlockBox(tuple(index), tuple(index)))
    levels = [LevelMeta(
        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=tuple(origin),
                       ref_ratio=3 if layout == "amr" else 1),
        boxes=boxes,
    )]
    if layout == "amr":
        # Refine the exterior cell. Ratio three retains a fine row whose
        # center lies on the coordinate axis, reproducing the shared patch.
        lo, hi = [0, 0, 0], [2, 2, 2]
        lo[axis], hi[axis] = (3, 5) if side > 0 else (0, 2)
        levels.append(LevelMeta(
            geom=LevelGeom(dx=(1.0 / 3.0,) * 3, x0=tuple(origin), ref_ratio=1),
            boxes=[BlockBox(tuple(lo), tuple(hi))],
        ))
    meta = _runmeta_with_step_index(0, levels)
    ds = open_dataset("memory://wall-face-owner", runmeta=meta, runtime=rt,
                      step=0, level=0)
    for fid in range(1, 10):
        ds.register_field(f"f{fid}", fid)
    normal = np.zeros(3)
    normal[axis] = side
    tangent = np.array([-normal[1], normal[0], 0.0])
    momentum = normal + 2.0 * tangent
    magnetic = 3.0 * normal + 4.0 * tangent
    for lev, level in enumerate(levels):
        for block, box in enumerate(level.boxes):
            shape = tuple(box.hi[a] - box.lo[a] + 1 for a in range(3))
            radial_centers = (
                origin[axis] +
                (np.arange(box.lo[axis], box.hi[axis] + 1) + 0.5)
                * level.geom.dx[axis]
            )
            radial_shape = [1, 1, 1]
            radial_shape[axis] = shape[axis]
            exterior = np.broadcast_to(
                (side * radial_centers > 1.0).reshape(radial_shape), shape
            )
            for fid, value in enumerate(
                [1.0, *momentum, 1000.0, 1.0, *magnetic], 1
            ):
                values = np.full(shape, value)
                if abs(offset) < 1.e-12 and fid in (2, 3, 7, 8):
                    # Distinguish correct interior ownership from arbitrarily
                    # choosing one of two identical states on the face.
                    values[exterior] *= 7.0
                _set_block_double(ds, step=0, level=lev, field=fid, block=block,
                                  values=values)
    pipe = Pipeline(runtime=rt, runmeta=meta, dataset=ds)
    radius = 1.0 + offset
    flux = pipe.cylindrical_flux_surface_integral(
        1, momentum=(2, 3, 4), energy=5, passive_scalar=6,
        magnetic_field=(7, 8, 9), radius=radius, height=0.25,
    )
    pipe.run()
    result = rt.get_task_chunk_array(
        step=0, level=0, field=flux.field, block=0, dtype=np.float64, dataset=ds,
    ).reshape(2, 2, 8)
    # The tangent patch has area 1 * (2 * height) = 0.5.
    expected = np.array([[0.0, 0.0, -6.0 * radius],
                         [0.5, radius, 0.0]])
    np.testing.assert_allclose(result[:, 1, :][:, [0, 4, 5]], expected,
                               rtol=1.e-12, atol=1.e-12)
