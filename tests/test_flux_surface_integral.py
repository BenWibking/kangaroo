from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from analysis import Runtime
from analysis.dataset import open_dataset
from analysis.pipeline import Pipeline
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


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
    arr = np.asarray(values, dtype=np.float64)
    ds._h.set_chunk_ref(step, level, field, 0, block, arr.tobytes(order="C"))


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
        bytes_per_value=8,
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
    assert all(tmpl.output_bytes == [64] for tmpl in accum)

    coarse = [tmpl for tmpl in accum if tmpl.domain.level == 0]
    fine = [tmpl for tmpl in accum if tmpl.domain.level == 1]
    assert coarse
    assert fine
    assert [[0, 0, 0], [0, 1, 1]] in coarse[0].params["covered_boxes"]
    assert fine[0].params["covered_boxes"] == []

    reducers = [tmpl for tmpl in templates if tmpl.kernel == "uniform_slice_reduce"]
    assert reducers
    assert all(tmpl.params["bytes_per_value"] == 8 for tmpl in reducers)
    assert all(tmpl.output_bytes == [64] for tmpl in reducers)
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
        bytes_per_value=8,
    )
    plan = pipe.plan()

    assert flux.radii == (0.25, 0.5, 0.75)
    templates = [tmpl for stage in plan.stages for tmpl in stage.templates]
    accum = [tmpl for tmpl in templates if tmpl.kernel == "flux_surface_integral_accumulate"]
    assert len(accum) == 1
    assert accum[0].params["radii"] == [0.25, 0.5, 0.75]
    assert accum[0].output_bytes == [192]
    reducers = [tmpl for tmpl in templates if tmpl.kernel == "uniform_slice_reduce"]
    assert reducers
    assert all(tmpl.output_bytes == [192] for tmpl in reducers)


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
        bytes_per_value=8,
    )
    plan = pipe.plan()

    assert flux.temperature_bins == (1.0, 10.0, 100.0)
    templates = [tmpl for stage in plan.stages for tmpl in stage.templates]
    accum = [tmpl for tmpl in templates if tmpl.kernel == "flux_surface_integral_accumulate"]
    assert len(accum) == 1
    assert len(accum[0].inputs) == 10
    assert accum[0].params["temperature_bins"] == [1.0, 10.0, 100.0]
    assert accum[0].output_bytes == [128]
    reducers = [tmpl for tmpl in templates if tmpl.kernel == "uniform_slice_reduce"]
    assert reducers
    assert all(tmpl.output_bytes == [128] for tmpl in reducers)


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
            bytes_per_value=8,
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
        bytes_per_value=8,
    )
    plan = pipe.plan()

    templates = [tmpl for stage in plan.stages for tmpl in stage.templates]
    accum = [tmpl for tmpl in templates if tmpl.kernel == "flux_surface_integral_accumulate"]
    assert len(accum) == 2
    by_block = {tmpl.domain.blocks[0]: tmpl for tmpl in accum}
    assert by_block[0].params["radii"] == [0.5]
    assert by_block[0].params["radius_indices"] == [0]
    assert by_block[1].params["radii"] == [3.5]
    assert by_block[1].params["radius_indices"] == [1]
    assert all(tmpl.params["num_radii"] == 2 for tmpl in accum)
    assert all(tmpl.output_bytes == [128] for tmpl in accum)


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
        bytes_per_value=8,
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
    assert reductions[0].params["input_blocks"] == [1]
    assert reductions[0].params["output_blocks"] == [0]
    assert reductions[0].params["group_offsets"] == [0, 1]


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
            bytes_per_value=8,
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
            bytes_per_value=8,
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
            bytes_per_value=8,
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
            bytes_per_value=8,
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
        bytes_per_value=8,
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
        bytes_per_value=8,
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
            bytes_per_value=8,
        )
        pipe.run()
        return rt.get_task_chunk_array(
            step=step,
            level=0,
            field=flux.field,
            shape=(len(radii), 2, 4),
            dtype=np.float64,
            dataset=ds,
            block=0,
        )

    combined = run_flux([0.5, 3.5], uri_suffix="combined")
    first = run_flux([0.5], uri_suffix="first")
    second = run_flux([3.5], uri_suffix="second")

    assert np.allclose(combined[0], first[0])
    assert np.allclose(combined[1], second[0])


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
        bytes_per_value=8,
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
        bytes_per_value=8,
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
        bytes_per_value=8,
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
        bytes_per_value=8,
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
        bytes_per_value=8,
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
